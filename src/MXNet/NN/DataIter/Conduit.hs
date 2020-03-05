{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module MXNet.NN.DataIter.Conduit (
    ConduitData(..),
    Dataset(..),
    imageRecordIter_v1,
    imageRecordIter, mnistIter, csvIter, libSVMIter
) where

import Data.Conduit
import qualified Data.Conduit.Combinators as C
import qualified Data.Conduit.List as CL
import Control.Applicative
import Control.Monad.IO.Class
import Control.Monad.Trans (lift)

import MXNet.Base
import qualified MXNet.Base.DataIter as I
import MXNet.NN.DataIter.Class

data ConduitData m a = ConduitData {
    iter_batch_size :: Maybe Int,
    getConduit :: ConduitM () a m ()
}

imageRecordIter_v1 :: (Fullfilled "ImageRecordIter_v1" args, DType a, MonadIO m)
    => ArgsHMap "ImageRecordIter_v1" args -> ConduitData m (NDArray a, NDArray a)
imageRecordIter_v1 args = ConduitData {
    getConduit = makeIter I._ImageRecordIter_v1 args,
    iter_batch_size = Just (args ! #batch_size)
}

imageRecordIter :: (Fullfilled "ImageRecordIter" args, DType a, MonadIO m)
    => ArgsHMap "ImageRecordIter" args -> ConduitData m (NDArray a, NDArray a)
imageRecordIter args = ConduitData {
    getConduit = makeIter I._ImageRecordIter args,
    iter_batch_size = Just (args ! #batch_size)
}

mnistIter :: (Fullfilled "MNISTIter" args, DType a, MonadIO m)
    => ArgsHMap "MNISTIter" args -> ConduitData m (NDArray a, NDArray a)
mnistIter args = ConduitData {
    getConduit = makeIter I._MNISTIter args,
    iter_batch_size = (args !? #batch_size) <|> Just 1
}

csvIter :: (Fullfilled "CSVIter" args, DType a, MonadIO m)
    => ArgsHMap "CSVIter" args -> ConduitData m (NDArray a, NDArray a)
csvIter args = ConduitData {
    getConduit = makeIter I._CSVIter args,
    iter_batch_size = Just (args ! #batch_size)
}

libSVMIter :: (Fullfilled "LibSVMIter" args, DType a, MonadIO m)
    => ArgsHMap "LibSVMIter" args -> ConduitData m (NDArray a, NDArray a)
libSVMIter args = ConduitData {
    getConduit = makeIter I._LibSVMIter args,
    iter_batch_size = Just (args ! #batch_size)
}

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
    foldD proc elem d = runConduit (getConduit d .| C.foldM proc elem)
    takeD n d = d {getConduit = getConduit d .| C.take n}
    liftD d = d {getConduit = transPipe lift (getConduit d)}

instance DatasetProp ConduitData a where
    batchSizeD = return . iter_batch_size
