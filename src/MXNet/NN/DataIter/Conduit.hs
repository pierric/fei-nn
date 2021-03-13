{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies          #-}
module MXNet.NN.DataIter.Conduit (
    ConduitData(..),
    Dataset(..),
    imageRecordIter_v1,
    imageRecordIter, mnistIter, csvIter, libSVMIter,
    forEachD_pi
) where

import           Data.Conduit
import qualified Data.Conduit.Combinators as C
import qualified Data.Conduit.List        as CL
import           Data.Conduit.TQueue      (sinkTBQueue)
import           RIO
import           RIO.Prelude              (lift)

import           MXNet.Base
import qualified MXNet.Base.DataIter      as I
import           MXNet.NN.DataIter.Class

data ConduitData m a = ConduitData
    { iter_batch_size :: Maybe Int
    , getConduit      :: ConduitM () a m ()
    }

imageRecordIter_v1 :: (Fullfilled "_ImageRecordIter_v1" () args, DType a, MonadIO m)
    => ArgsHMap "_ImageRecordIter_v1" () args -> ConduitData m (NDArray a, NDArray a)
imageRecordIter_v1 args = ConduitData {
    getConduit = makeIter I._ImageRecordIter_v1 args,
    iter_batch_size = Just (args ! #batch_size)
}

imageRecordIter :: (Fullfilled "_ImageRecordIter" () args, DType a, MonadIO m)
    => ArgsHMap "_ImageRecordIter" () args -> ConduitData m (NDArray a, NDArray a)
imageRecordIter args = ConduitData {
    getConduit = makeIter I._ImageRecordIter args,
    iter_batch_size = Just (args ! #batch_size)
}

mnistIter :: (Fullfilled "_MNISTIter" () args, DType a, MonadIO m)
    => ArgsHMap "_MNISTIter" () args -> ConduitData m (NDArray a, NDArray a)
mnistIter args = ConduitData {
    getConduit = makeIter I._MNISTIter args,
    iter_batch_size = (args !? #batch_size) <|> Just 1
}

csvIter :: (Fullfilled "_CSVIter" () args, DType a, MonadIO m)
    => ArgsHMap "_CSVIter" () args -> ConduitData m (NDArray a, NDArray a)
csvIter args = ConduitData {
    getConduit = makeIter I._CSVIter args,
    iter_batch_size = Just (args ! #batch_size)
}

libSVMIter :: (Fullfilled "_LibSVMIter" () args, DType a, MonadIO m)
    => ArgsHMap "_LibSVMIter" () args -> ConduitData m (NDArray a, NDArray a)
libSVMIter args = ConduitData {
    getConduit = makeIter I._LibSVMIter args,
    iter_batch_size = Just (args ! #batch_size)
}

makeIter :: MonadIO m
    => (args -> IO DataIterHandle) -> args -> ConduitT i (NDArray a, NDArray a) m ()
makeIter creator args = do
    iter <- liftIO (creator args)
    let loop = do valid <- liftIO $ mxDataIterNext iter
                  if valid == 0
                  then liftIO (finalizeDataIterHandle iter)
                  else do
                      yieldM $ liftIO $ do
                          dat <- mxDataIterGetData  iter
                          lbl <- mxDataIterGetLabel iter
                          return (NDArray dat, NDArray lbl)
                      loop
    loop

instance Dataset ConduitData where
    type DatasetMonadConstraint ConduitData m = ()
    fromListD = ConduitData Nothing . CL.sourceList
    zipD d1 d2 = ConduitData Nothing $ getZipSource $ (,) <$> ZipSource (getConduit d1) <*> ZipSource (getConduit d2)
    sizeD d = runConduit (getConduit d .| C.length)
    forEachD d proc = sourceToList $ getConduit d .| CL.mapM proc
    foldD proc unit d = runConduit (getConduit d .| C.foldM proc unit)
    takeD n d = d {getConduit = getConduit d .| C.take n}
    liftD d = d {getConduit = transPipe lift (getConduit d)}

instance DatasetProp ConduitData a where
    batchSizeD = return . iter_batch_size

forEachD_pi size (ConduitData _ conduit) proc = do
    queue <- liftIO $ newTBQueueIO (fromIntegral size)
    stop  <- liftIO $ newTVarIO False

    let indexed  = getZipSource $ (,) <$> ZipSource (CL.sourceList [0..]) <*> ZipSource conduit
        producer = runConduit $ indexed .| sinkTBQueue queue
        consumer = do
            e <- atomically $ do
                    should_stop <- readTVar stop
                    if should_stop
                    then pure Nothing
                    else do
                        e <- tryReadTBQueue queue
                        case e of
                          Nothing -> retrySTM
                          _       -> pure e
            case e of
              Nothing -> return ()
              Just e  -> proc e >> consumer
    withAsyncBound consumer $ \a1 ->
        withAsync producer  $ \a2 -> do
            wait a2
            atomically $ writeTVar stop True
            wait a1
