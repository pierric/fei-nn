{-# Language MultiParamTypeClasses, FunctionalDependencies #-}
{-# LANGUAGE FlexibleInstances #-}
module MXNet.NN.DataIter.Class where

import GHC.Exts (Constraint)
import Control.Monad (foldM)

-- | Constraints on Dataset and the monad where the operation shall be ran.
type family DatasetConstraint (d :: * -> *) (m :: * -> *) :: Constraint

-- | Abstract Dataset type class.
-- Available instances include 'LVec' and mxnet data-iters in package <https://github.com/pierric/mxnet-dataiter mxnet-dataiter>
class Dataset (d :: * -> *) where
    -- | Create Dataset from `[]`.
    -- note that depending on the instance, it may or may not work with infinitive list.
    fromListD   :: [e] -> d e
    -- | Zip two Datasets
    zipD        :: d e1 -> d e2 -> d (e1, e2)
    -- | Get number of elements
    sizeD       :: (DatasetConstraint d m, Monad m) => d e -> m Int
    -- | Apply a function on each element of Dataset
    forEachD    :: (DatasetConstraint d m, Monad m) => d e -> (e -> m a) -> m [a]

    -- | Apply a function on each element of Dataset together with the element's index. 
    -- Note that the default implmentation assumes the Dataset can be created from a infinitive list.
    forEachD_i  :: (DatasetConstraint d m, Monad m) => d e -> ((Int, e) -> m a) -> m [a]
    forEachD_i  dat = forEachD (zipD (fromListD [1..]) dat)

    -- | Apply a function on each element of Dataset together with the total number of elements and the element's index.
    forEachD_ni :: (DatasetConstraint d m, Monad m) => d e -> (((Int, Int), e) -> m a) -> m [a]
    forEachD_ni dat proc = do 
        n <- sizeD dat
        forEachD ((fromListD (replicate n n) `zipD` fromListD [1..n]) `zipD` dat) proc

    foldD :: (DatasetConstraint d m, Monad m) => d e -> a -> (a -> e -> m a) -> m a

type instance DatasetConstraint [] m = ()
instance Dataset [] where
    fromListD = id
    zipD = zip
    sizeD = return . length
    forEachD = flip mapM
    foldD list ele func = foldM func ele list