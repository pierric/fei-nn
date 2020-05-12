{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
module MXNet.NN.DataIter.Class where

import GHC.Exts (Constraint)
import RIO
import RIO.Prelude.Types (MonadTrans)

-- Available instances include 'LVec' and mxnet data-iters in package <https://github.com/pierric/mxnet-dataiter mxnet-dataiter>
class Dataset (d :: (* -> *) -> * -> *) where
    type DatasetMonadConstraint d (m :: * -> *) :: Constraint
    -- | Create Dataset from `[]`.
    -- note that depending on the instance, it may or may not work with infinitive list.
    fromListD   :: (Monad m, DatasetMonadConstraint d m) => [e] -> d m e
    -- | Zip two Datasets
    zipD        :: (Monad m, DatasetMonadConstraint d m) => d m e1 -> d m e2 -> d m (e1, e2)
    -- | Get number of elements
    sizeD       :: (Monad m, DatasetMonadConstraint d m) => d m e -> m Int
    -- | Apply a function on each element of Dataset
    forEachD    :: (Monad m, DatasetMonadConstraint d m) => d m e -> (e -> m a) -> m [a]

    -- | Apply a function on each element of Dataset together with the element's index.
    -- Note that the default implmentation assumes the Dataset can be created from a infinitive list.
    forEachD_i  :: (Monad m, DatasetMonadConstraint d m) => d m e -> ((Int, e) -> m a) -> m [a]
    forEachD_i  dat = forEachD (zipD (fromListD [1..]) dat)

    -- | Apply a function on each element of Dataset together with the total number of elements and the element's index.
    forEachD_ni :: (Monad m, DatasetMonadConstraint d m) => d m e -> (((Int, Int), e) -> m a) -> m [a]
    forEachD_ni dat proc = do
        n <- sizeD dat
        forEachD ((fromListD (replicate n n) `zipD` fromListD [1..n]) `zipD` dat) proc

    foldD :: (Monad m, DatasetMonadConstraint d m) => (a -> e -> m a) -> a -> d m e -> m a

    takeD :: (Monad m, DatasetMonadConstraint d m) => Int -> d m e -> d m e

    -- | Lift from one monad into another
    liftD :: (MonadTrans t, Monad m, DatasetMonadConstraint d m) => d m a -> d (t m) a


class Dataset d => DatasetProp (d :: (* -> *) -> * -> *) e where
    -- | Get the batch size of the dataset
    batchSizeD :: (Monad m, DatasetMonadConstraint d m) => d m e -> m (Maybe Int)
