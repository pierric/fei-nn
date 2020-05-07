{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module MXNet.NN.DataIter.Streaming (
    StreamData(..),
    Dataset(..),
    imageRecordIter_v1,
    imageRecordIter, mnistIter, csvIter, libSVMIter
) where

import RIO
import RIO.Prelude (lift)
import Streaming
import Streaming.Prelude (Of(..), yield, length_, toList_)
import qualified Streaming.Prelude as S

import MXNet.Base
import qualified MXNet.Base.DataIter as I
import MXNet.NN.DataIter.Class

data StreamData m a = StreamData {
    iter_batch_size :: Maybe Int,
    getStream :: Stream (Of a) m ()
}

imageRecordIter_v1 :: (Fullfilled "_ImageRecordIter_v1" args, DType a, MonadIO m)
    => ArgsHMap "_ImageRecordIter_v1" args -> StreamData m (NDArray a, NDArray a)
imageRecordIter_v1 args = StreamData {
    getStream = makeIter I._ImageRecordIter_v1 args,
    iter_batch_size = Just (args ! #batch_size)
}

imageRecordIter :: (Fullfilled "_ImageRecordIter" args, DType a, MonadIO m)
    => ArgsHMap "_ImageRecordIter" args -> StreamData m (NDArray a, NDArray a)
imageRecordIter args = StreamData {
    getStream = makeIter I._ImageRecordIter args,
    iter_batch_size = Just (args ! #batch_size)
}

mnistIter :: (Fullfilled "_MNISTIter" args, DType a, MonadIO m)
    => ArgsHMap "_MNISTIter" args -> StreamData m (NDArray a, NDArray a)
mnistIter args = StreamData {
    getStream = makeIter I._MNISTIter args,
    iter_batch_size = (args !? #batch_size) <|> Just 1
}

csvIter :: (Fullfilled "_CSVIter" args, DType a, MonadIO m)
    => ArgsHMap "_CSVIter" args -> StreamData m (NDArray a, NDArray a)
csvIter args = StreamData {
    getStream = makeIter I._CSVIter args,
    iter_batch_size = Just (args ! #batch_size)
}

libSVMIter :: (Fullfilled "_LibSVMIter" args, DType a, MonadIO m)
    => ArgsHMap "_LibSVMIter" args -> StreamData m (NDArray a, NDArray a)
libSVMIter args = StreamData {
    getStream = makeIter I._LibSVMIter args,
    iter_batch_size = Just (args ! #batch_size)
}

makeIter :: MonadIO m
    => (args -> IO DataIterHandle) -> args -> Stream (Of (NDArray a, NDArray a)) m ()
makeIter creator args = do
    iter <- liftIO (creator args)
    let loop = do valid <- liftIO $ mxDataIterNext iter
                  if valid == 0
                  then liftIO (finalizeDataIterHandle iter)
                  else do
                      item <- liftIO $ do
                          dat <- mxDataIterGetData  iter
                          lbl <- mxDataIterGetLabel iter
                          return (NDArray dat, NDArray lbl)
                      yield item
                      loop
    loop

instance Dataset StreamData where
    type DatasetMonadConstraint StreamData m = ()
    fromListD = StreamData Nothing . S.each
    zipD s1 s2 = StreamData Nothing $ S.zip (getStream s1) (getStream s2)
    sizeD = length_ . getStream
    forEachD dat proc = toList_ $ void $ S.mapM proc (getStream dat)
    foldD proc unit dat = S.foldM_ proc (return unit) return (getStream dat)
    takeD n dat = dat { getStream = S.take n (getStream dat) }
    liftD dat = dat { getStream = hoist lift (getStream dat) }

instance DatasetProp StreamData a where
    batchSizeD = return . iter_batch_size
