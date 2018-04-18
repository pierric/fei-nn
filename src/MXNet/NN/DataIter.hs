{-# LANGUAGE FlexibleInstances #-}
module MXNet.NN.DataIter where

import Control.Monad (void)
import MXNet.Core.Base
import qualified Streaming.Prelude as SR
import qualified Data.Vector as Vec

class Dataset d where
    type EffectM d :: * -> *
    type DatType d :: *
    size :: d -> EffectM d Int
    forEach :: d -> ((Int,Int) -> NDArray (DatType d) -> NDArray (DatType d) -> EffectM d a) -> EffectM d [a]

instance Monad m => Dataset (SR.Stream (SR.Of (NDArray t, NDArray t)) m Int) where
    type EffectM (SR.Stream (SR.Of (NDArray t, NDArray t)) m Int) = m
    type DatType (SR.Stream (SR.Of (NDArray t, NDArray t)) m Int) = t
    size dat = SR.effects dat
    forEach dat proc = do
        let index = SR.enumFrom (1 :: Int)
        t <- size dat
        SR.toList_ $ void $ SR.mapM (\(i,(x,y)) -> proc (i,t) x y) (SR.zip index dat)

instance Dataset (Vec.Vector (NDArray t, NDArray t)) where
    type EffectM (Vec.Vector (NDArray t, NDArray t)) = IO
    type DatType (Vec.Vector (NDArray t, NDArray t)) = t
    size dat = return $ Vec.length dat
    forEach dat proc = do
        let t = Vec.length dat
        ret <- Vec.imapM (\i (x,y) -> proc (i,t) x y) dat
        return $ Vec.toList ret