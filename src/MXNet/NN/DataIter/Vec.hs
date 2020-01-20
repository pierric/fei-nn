{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
module MXNet.NN.DataIter.Vec where

import Data.Vector (Vector)
import qualified Data.Vector as V
import Control.Monad.IO.Class (MonadIO, liftIO)

import MXNet.NN.DataIter.Class
import MXNet.Base (NDArray, DType, ndshape)

newtype DatasetVector (m :: * -> *) a = DatasetVector { _dsv_unwrap :: Vector a }


instance Dataset DatasetVector where
    type DatasetMonadConstraint DatasetVector m = MonadIO m
    fromListD = DatasetVector . V.fromList
    zipD v1 v2 = DatasetVector $ V.zip (_dsv_unwrap v1) (_dsv_unwrap v2)
    sizeD = return . V.length . _dsv_unwrap
    forEachD dat func   = V.toList <$> V.forM (_dsv_unwrap dat) func
    forEachD_i dat func = V.toList <$> V.forM (V.indexed $ _dsv_unwrap dat) func
    foldD func ele = V.foldM' func ele . _dsv_unwrap
    takeD n = DatasetVector . V.take n . _dsv_unwrap
    liftD (DatasetVector x) = DatasetVector x

instance DType a => DatasetProp DatasetVector (NDArray a) where
    batchSizeD (DatasetVector dat) = liftIO $ do
        batch_size : _ <- ndshape $ V.head dat
        return $ Just batch_size

instance DType a => DatasetProp DatasetVector (NDArray a, NDArray a) where
    batchSizeD (DatasetVector dat) = do
        let (arr1, arr2) = V.head dat
        liftIO $ do
            batch_size1 : _ <- ndshape arr1
            batch_size2 : _ <- ndshape arr2
            return $ if batch_size1 /= batch_size2
                        then Nothing
                        else Just batch_size1
