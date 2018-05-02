module MXNet.NN.DataIter.Class where

import MXNet.Core.Base (NDArray)
import GHC.Exts (Constraint)
import MXNet.NN.Types (TrainM)

type family DatasetConstraint d (m :: * -> *) :: Constraint

class Dataset d where
    type DatType d :: *
    size :: DatasetConstraint d m => d -> TrainM (DatType d) m Int
    forEach :: DatasetConstraint d m => d -> (Int -> NDArray (DatType d) -> NDArray (DatType d) -> TrainM (DatType d) m a) -> TrainM (DatType d) m [a]
    forEach' :: DatasetConstraint d m => d -> ((Int,Int) -> NDArray (DatType d) -> NDArray (DatType d) -> TrainM (DatType d) m a) -> TrainM (DatType d) m [a]

