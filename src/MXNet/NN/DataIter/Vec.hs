{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module MXNet.NN.DataIter.Vec where

import           RIO
import           RIO.List                 (headMaybe)
import qualified RIO.Vector.Boxed         as V
import qualified RIO.Vector.Boxed.Partial as V (head)

import           MXNet.Base               (DType, NDArray, ndshape)
import           MXNet.NN.DataIter.Class

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
    batchSizeD (DatasetVector dat) = liftIO $ headMaybe <$> ndshape (V.head dat)

instance DType a => DatasetProp DatasetVector (NDArray a, NDArray a) where
    batchSizeD (DatasetVector dat) = do
        let (arr1, arr2) = V.head dat
        liftIO $ do
            s1 <- ndshape arr1
            s2 <- ndshape arr2
            return $ case (headMaybe s1, headMaybe s2) of
              (Just b1, Just b2) | b1 == b2 -> Just b1
              _                             -> Nothing
