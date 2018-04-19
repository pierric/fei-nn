{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE DataKinds #-}
module MXNet.NN.DataIter where

import Control.Monad (void)
import GHC.Exts (Constraint)
import MXNet.Core.Base
import qualified Streaming.Prelude as SR
import qualified Data.Vector as Vec

import MXNet.NN.Types (TrainM)

type family DatasetConstraint d (m :: * -> *) :: Constraint

class Dataset d where
    type DatType d :: *
    size :: DatasetConstraint d m => d -> TrainM (DatType d) m Int
    forEach :: DatasetConstraint d m => d -> ((Int,Int) -> NDArray (DatType d) -> NDArray (DatType d) -> TrainM (DatType d) m a) -> TrainM (DatType d) m [a]

type instance DatasetConstraint (SR.Stream (SR.Of (NDArray t, NDArray t)) (TrainM t m1) Int) m2 = m1 ~ m2

instance Monad m => Dataset (SR.Stream (SR.Of (NDArray t, NDArray t)) (TrainM t m) Int) where
    type DatType (SR.Stream (SR.Of (NDArray t, NDArray t)) (TrainM t m) Int) = t
    size dat = SR.effects dat
    forEach dat proc = do
        let index = SR.enumFrom (1 :: Int)
        t <- size dat
        SR.toList_ $ void $ SR.mapM (\(i,(x,y)) -> proc (i,t) x y) (SR.zip index dat)

type instance DatasetConstraint (Vec.Vector (NDArray t, NDArray t)) m2 = Monad m2
instance Dataset (Vec.Vector (NDArray t, NDArray t)) where
    type DatType (Vec.Vector (NDArray t, NDArray t)) = t
    size dat = return $ Vec.length dat
    forEach dat proc = do
        let t = Vec.length dat
        ret <- Vec.imapM (\i (x,y) -> proc (i,t) x y) dat
        return $ Vec.toList ret