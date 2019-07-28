{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
module MXNet.NN.DataIter.Vec where

import Data.Vector (Vector)
import qualified Data.Vector as V
import Control.Monad (when)
import Control.Monad.IO.Class (MonadIO, liftIO)
-- import Control.Monad.Trans.Resource (MonadThrow(..))

import MXNet.NN.DataIter.Class
import MXNet.NN.Types
import MXNet.Base (NDArray, DType, ndshape)

newtype DatasetVector a = DatasetVector { _dsv_unwrap :: Vector a }


type instance DatasetConstraint DatasetVector m = MonadIO m

instance Dataset DatasetVector where
    fromListD = DatasetVector . V.fromList
    zipD v1 v2 = DatasetVector $ V.zip (_dsv_unwrap v1) (_dsv_unwrap v2)
    sizeD = return . V.length . _dsv_unwrap
    forEachD dat func   = V.toList <$> V.forM (_dsv_unwrap dat) func
    forEachD_i dat func = V.toList <$> V.forM (V.indexed $ _dsv_unwrap dat) func
    foldD func ele = V.foldM' func ele . _dsv_unwrap
    takeD n = return . V.toList . V.take n . _dsv_unwrap

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
