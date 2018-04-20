{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE DataKinds #-}
module MXNet.NN.DataIter where

import Control.Monad (void)
import Control.Monad.IO.Class
import GHC.Exts (Constraint)
import MXNet.Core.Base
import qualified Streaming.Prelude as SR
import qualified Data.Vector as Vec

import MXNet.NN.Types (TrainM)
import MXNet.NN.LazyVec

type family DatasetConstraint d (m :: * -> *) :: Constraint

class Dataset d where
    type DatType d :: *
    forEach :: DatasetConstraint d m => d -> ((Int,Int) -> NDArray (DatType d) -> NDArray (DatType d) -> TrainM (DatType d) m a) -> TrainM (DatType d) m [a]

type instance DatasetConstraint (SR.Stream (SR.Of (NDArray t, NDArray t)) (TrainM t m1) Int) m2 = m1 ~ m2

instance Monad m => Dataset (SR.Stream (SR.Of (NDArray t, NDArray t)) (TrainM t m) Int) where
    type DatType (SR.Stream (SR.Of (NDArray t, NDArray t)) (TrainM t m) Int) = t
    forEach dat proc = do
        let index = SR.enumFrom (1 :: Int)
        t <- SR.effects dat
        SR.toList_ $ void $ SR.mapM (\(i,(x,y)) -> proc (i,t) x y) (SR.zip index dat)

type instance DatasetConstraint (LVec (NDArray t, NDArray t)) m2 = MonadIO m2
instance Dataset (LVec (NDArray t, NDArray t)) where
    type DatType (LVec (NDArray t, NDArray t)) = t
    forEach dat proc = do
        vec <- liftIO $ toVec dat
        let t = Vec.length vec
        ret <- Vec.imapM (\i (x,y) -> proc (i+1,t) x y) vec
        return $ Vec.toList ret
