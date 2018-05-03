{-# Language MultiParamTypeClasses, FunctionalDependencies #-}
{-# LANGUAGE FlexibleInstances #-}
module MXNet.NN.DataIter.Class where

import GHC.Exts (Constraint)

type family DatasetConstraint (d :: * -> *) (m :: * -> *) :: Constraint

class Dataset (d :: * -> *) where
    fromListD   :: [e] -> d e
    zipD        :: d e1 -> d e2 -> d (e1, e2)
    sizeD       :: (DatasetConstraint d m, Monad m) => d e -> m Int
    forEachD    :: (DatasetConstraint d m, Monad m) => d e -> (e -> m a) -> m [a]

    forEachD_i  :: (DatasetConstraint d m, Monad m) => d e -> ((Int, e) -> m a) -> m [a]
    forEachD_i  dat = forEachD (zipD (fromListD [1..]) dat)

    forEachD_ni :: (DatasetConstraint d m, Monad m) => d e -> (((Int, Int), e) -> m a) -> m [a]
    forEachD_ni dat proc = do 
        n <- sizeD dat
        forEachD ((fromListD (repeat n) `zipD` fromListD [1..]) `zipD` dat) proc