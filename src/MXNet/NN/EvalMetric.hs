{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
module MXNet.NN.EvalMetric where

import Data.IORef
import qualified Data.HashMap.Strict as M
import Control.Monad.Trans.Resource (MonadThrow(..))
import Control.Monad
import Control.Monad.IO.Class (MonadIO, liftIO)
import Text.Printf (printf)
import qualified Data.Vector.Storable as SV
import Control.Lens (makeLenses, use, (^.))
import Control.Monad.State.Strict (lift)
import MXNet.Core.Base (DType, context, ndshape, nil)
import qualified MXNet.Core.Base.NDArray as A
import qualified MXNet.Core.Base.Internal.TH.NDArray as A
import MXNet.NN.Types

-- | Metric data
data BaseMetric dytpe = BaseMetric {
    _metric_name :: String,
    _metric_labelname :: [String],
    _metric_instance :: IORef Int,
    _metric_sum :: IORef dytpe
}
makeLenses ''BaseMetric

-- | create a new metric data
newBaseMetric :: (DType dtype, MonadIO m) => String -> [String] -> m (BaseMetric dtype)
newBaseMetric name labels = do
    a <- liftIO $ newIORef 0
    b <- liftIO $ newIORef 0
    return $ BaseMetric name labels a b

-- | reset all information
resetBaseMetric :: (DType dtype, MonadIO m) => BaseMetric dtype -> m ()
resetBaseMetric metric = liftIO $ do
    writeIORef (_metric_sum metric) 0
    writeIORef (_metric_instance metric) 0

-- -- | get the metric
getBaseMetric :: (DType dtype, MonadIO m) => BaseMetric dtype -> m Float
getBaseMetric metric = do
    s <- liftIO $ readIORef (_metric_sum metric)
    n <- liftIO $ readIORef (_metric_instance metric)
    return $ realToFrac s / fromIntegral n

-- | Abstract Evaluation type class
class DType (MetricDType metric) => EvalMetricMethod metric where
    type MetricDType metric :: *
    evaluate :: MonadIO m
             => metric                                         -- evaluation metric
             -> M.HashMap String (A.NDArray (MetricDType metric))  -- network bindings
             -> [A.NDArray (MetricDType metric)]                   -- output of the network
             -> TrainM (MetricDType metric) m ()
    format   :: MonadIO m => metric -> TrainM (MetricDType metric) m String

-- | Basic evluation - cross entropy 
data CrossEntropy a = CrossEntropy (BaseMetric a)

metricCE :: (DType dtype, MonadIO m) => [String] -> m (CrossEntropy dtype)
metricCE labels = CrossEntropy <$> newBaseMetric "CrossEntropy" labels

instance DType a => EvalMetricMethod (CrossEntropy a) where
    type MetricDType (CrossEntropy a) = a
    -- | evaluate the log-loss. 
    -- preds is of shape (batch_size, num_category), each element along the second dimension gives the probability of the category.
    -- label is of shape (batch_size,), each element gives the category number.
    evaluate (CrossEntropy metric) bindings outputs = do
        let labels = map (bindings M.!) (metric ^. metric_labelname)
        liftIO $ zipWithM_ each outputs labels 
      where
        each preds label = do
            (n1, shp1) <- A.ndshape preds
            (n2, shp2) <- A.ndshape label
            when (n1 /= 2 || n2 /= 1 || head shp1 /= head shp2) (throwM $ MismatchedShapeInEval shp1 shp2)
            -- before call pick, we have to make sure preds and label 
            -- are in the same context
            preds_may_copy <- do
                c1 <- context preds
                c2 <- context label
                if c1 == c2 
                    then return preds
                    else do
                        (_, preds_shap) <- ndshape preds
                        preds_copy <- A.makeEmptyNDArray preds_shap c2 False
                        A._copyto' (A.getHandle preds) [A.getHandle preds_copy] :: IO ()
                        return preds_copy
            predprj <- A.pick (A.getHandle preds_may_copy) (A.getHandle label) nil
            predlog <- A.log predprj
            loss    <- A.sum predlog nil >>= A.items . A.NDArray
            modifyIORef (_metric_sum metric) (+ (negate $ loss SV.! 0))
            modifyIORef (_metric_instance metric) (+ head shp1)
    format (CrossEntropy metric) = liftIO $ do
        e <- getBaseMetric metric 
        return $ printf "<%s: %0.3f>" (_metric_name metric) e

-- | Learning rate
data DumpLearningRate a = DumpLearningRate

metricLR :: (DType dtype, MonadIO m) => m (DumpLearningRate dtype)
metricLR = return DumpLearningRate

instance DType a => EvalMetricMethod (DumpLearningRate a) where
    type MetricDType (DumpLearningRate a) = a
    evaluate _ _ _ = return ()
    format _ = do
        lr <- lift $ use stat_last_lr
        return $ printf "<lr: %0.6f>" lr

data ListOfMetric dt a where
    MNil :: ListOfMetric dt '[]
    (:+) :: (EvalMetricMethod a, MetricDType a ~ dt) => a -> ListOfMetric dt as -> ListOfMetric dt (a ': as)

instance DType dt => EvalMetricMethod (ListOfMetric dt '[]) where
    type MetricDType (ListOfMetric dt '[]) = dt
    evaluate _ _ _ = return ()
    format _ = return ""

instance (DType dt, MetricDType a ~ MetricDType (ListOfMetric dt as), EvalMetricMethod a, EvalMetricMethod (ListOfMetric dt as)) => EvalMetricMethod (ListOfMetric dt (a ': as)) where
    type MetricDType (ListOfMetric dt (a ': as)) = dt
    evaluate (a :+ as) bindings outputs = do
        evaluate a  bindings outputs
        evaluate as bindings outputs
    format (a :+ as) = do
        s1 <- format a
        s2 <- format as
        return $ s1 ++ " " ++ s2

infixr 9 :+
infixr 9 #+
infixr 9 ##

(#+) :: (Monad m, EvalMetricMethod a, MetricDType a ~ dt) 
     => m a -> m (ListOfMetric dt as) -> m (ListOfMetric dt (a ': as))
(#+) = liftM2 (:+)

(##) :: (Monad m, EvalMetricMethod a, EvalMetricMethod b, MetricDType a ~ MetricDType b) 
     => m a -> m b -> m (ListOfMetric (MetricDType a) ('[a, b]))
(##) a b = a #+ b #+ (return MNil)
