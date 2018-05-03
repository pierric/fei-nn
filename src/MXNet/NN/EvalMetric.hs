{-# LANGUAGE TemplateHaskell #-}
module MXNet.NN.EvalMetric where

import Data.IORef
import Control.Exception.Base (Exception)
import Control.Monad.Trans.Resource (MonadThrow(..))
import Data.Typeable (Typeable)
import Control.Monad
import Control.Monad.IO.Class (MonadIO, liftIO)
import Text.Printf (printf)
import qualified Data.Vector.Storable as SV
import Control.Lens (makeLenses)
import MXNet.Core.Base
import qualified MXNet.Core.Base.NDArray as A
import qualified MXNet.Core.Base.Internal.TH.NDArray as A

-- | Metric data
data Metric dytpe method = Metric {
    _metric_name :: String,
    _metric_labelname :: [String],
    _metric_instance :: IORef Int,
    _metric_sum :: IORef dytpe
}
makeLenses ''Metric

-- | create a new metric data
newMetric :: (DType dtype, MonadIO m) => method -> String -> [String] -> m (Metric dtype method)
newMetric _ name labels = do
    a <- liftIO $ newIORef 0
    b <- liftIO $ newIORef 0
    return $ Metric name labels a b

-- | reset all information
resetMetric :: (DType dtype, MonadIO m) => Metric dtype method -> m ()
resetMetric metric = liftIO $ do
    writeIORef (_metric_sum metric) 0
    writeIORef (_metric_instance metric) 0

-- | get the metric
getMetric :: (DType dtype, MonadIO m) => Metric dtype method -> m Float
getMetric metric = do
    s <- liftIO $ readIORef (_metric_sum metric)
    n <- liftIO $ readIORef (_metric_instance metric)
    return $ realToFrac s / fromIntegral n

-- | format the metric as string
formatMetric :: (DType dtype, MonadIO m) => Metric dtype method -> m String
formatMetric metric = do
    e <- getMetric metric 
    return $ printf "<%s: %0.3f>" (_metric_name metric) e

-- | Abstract Evaluation type class
class EvalMetricMethod method where
    evaluate :: DType dtype => Metric dtype method -> A.NDArray dtype -> A.NDArray dtype -> IO ()

-- | Basic evluation - cross entropy 
data CrossEntropy = CrossEntropy
instance EvalMetricMethod CrossEntropy where
    -- | evaluate the log-loss. 
    -- preds is of shape (batch_size, num_category), each element along the second dimension gives the probability of the category.
    -- label is of shape (batch_size,), each element gives the category number.
    evaluate metric preds label = do
        (n1, shp1) <- A.ndshape preds
        (n2, shp2) <- A.ndshape label
        when (n1 /= 2 || n2 /= 1 || head shp1 /= head shp2) (throwM InvalidInput)
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

-- | Possible exceptions in evaluation.
data EvalMetricExc = InvalidInput
    deriving (Show, Typeable)
instance Exception EvalMetricExc

