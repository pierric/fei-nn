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

import MXNet.Base
import qualified MXNet.Base.Operators.NDArray as A
import MXNet.NN.Types

-- | Metric data
data BaseMetric a = BaseMetric {
    _metric_name :: String,
    _metric_labelname :: [String],
    _metric_instance :: IORef Int,
    _metric_sum :: IORef a
}
makeLenses ''BaseMetric

-- | create a new metric data
newBaseMetric :: (Num a, MonadIO m) => String -> [String] -> m (BaseMetric a)
newBaseMetric name labels = do
    a <- liftIO $ newIORef 0
    b <- liftIO $ newIORef 0
    return $ BaseMetric name labels a b

-- | reset all information
resetBaseMetric :: (Num a, MonadIO m) => BaseMetric a -> m ()
resetBaseMetric metric = liftIO $ do
    writeIORef (_metric_sum metric) 0
    writeIORef (_metric_instance metric) 0

-- -- | get the metric
getBaseMetric :: MonadIO m => BaseMetric a -> m (a, Int)
getBaseMetric metric = do
    s <- liftIO $ readIORef (_metric_sum metric)
    n <- liftIO $ readIORef (_metric_instance metric)
    return (s, n)

-- | Abstract Evaluation type class
class EvalMetricMethod metric where
    evaluate :: (MonadIO m, DType a)
             => metric a                        -- evaluation metric
             -> M.HashMap String (NDArray a)  -- network bindings
             -> [NDArray a]                   -- output of the network
             -> TrainM a m ()
    format   :: (MonadIO m, DType a) => metric a -> TrainM a m String


-- | Basic evaluation - accuracy
data Accuracy a = Accuracy (BaseMetric Int)

mACC :: (DType dtype, MonadIO m) => [String] -> m (Accuracy dtype)
mACC labels = Accuracy <$> newBaseMetric "Accuracy" labels

instance EvalMetricMethod Accuracy where
    evaluate (Accuracy metric) bindings outputs = do
        let labels = map (bindings M.!) (metric ^. metric_labelname)
        liftIO $ zipWithM_ each outputs labels 
      where
        each preds@(NDArray preds_hdl) label = do
            [pred_cat_hdl] <- A.argmax (#data := preds_hdl .& #axis := Just 1 .& Nil)
            pred_cat <- toVector (NDArray pred_cat_hdl)
            real_cat <- toVector label

            batch_size:_ <- ndshape preds
            let correct = SV.length $ SV.filter id $ SV.zipWith (==) pred_cat real_cat
            modifyIORef (_metric_sum metric) (+ correct)
            modifyIORef (_metric_instance metric) (+ batch_size)
    format (Accuracy metric) = liftIO $ do
        (s, n) <- getBaseMetric metric 
        return $ printf "<%s: %0.2f>" (_metric_name metric) (100 * fromIntegral s / fromIntegral n :: Float)

-- | Basic evaluation - cross entropy 
data CrossEntropy a = CrossEntropy (BaseMetric a)

mCE :: (DType dtype, MonadIO m) => [String] -> m (CrossEntropy dtype)
mCE labels = CrossEntropy <$> newBaseMetric "CrossEntropy" labels


copyTo :: DType a => NDArray a -> NDArray a -> IO ()
copyTo (NDArray dst) (NDArray src) = A._copyto_upd [dst] (#data := src .& Nil)

instance EvalMetricMethod CrossEntropy where
    -- | evaluate the log-loss. 
    -- preds is of shape (batch_size, num_category), each element along the second dimension gives the probability of the category.
    -- label is of shape (batch_size,), each element gives the category number.
    evaluate (CrossEntropy metric) bindings outputs = do
        let labels = map (bindings M.!) (metric ^. metric_labelname)
        liftIO $ zipWithM_ each outputs labels 
      where
        each preds label@(NDArray labelHandle) = do
            shp1 <- ndshape preds
            shp2 <- ndshape label
            when (length shp1 /= 2 || length shp2 /= 1 || head shp1 /= head shp2) (throwM $ MismatchedShapeInEval shp1 shp2)
            -- before call pick, we have to make sure preds and label 
            -- are in the same context
            NDArray preds_may_copy <- do
                c1 <- context preds
                c2 <- context label
                if c1 == c2 
                    then return preds
                    else do
                        preds_shap <- ndshape preds
                        preds_copy <- makeEmptyNDArray preds_shap c2
                        copyTo preds_copy preds
                        return preds_copy
            [predprj] <- A.pick (#data := preds_may_copy .& #index := labelHandle .& Nil)
            [predlog] <- A.log (#data := predprj .& Nil)
            loss      <- A.sum (#data := predlog .& Nil) >>= toVector . NDArray . head
            modifyIORef (_metric_sum metric) (+ (negate $ loss SV.! 0))
            modifyIORef (_metric_instance metric) (+ head shp1)
    format (CrossEntropy metric) = liftIO $ do
        (s, n) <- getBaseMetric metric 
        return $ printf "<%s: %0.3f>" (_metric_name metric) (realToFrac s / fromIntegral n :: Float)

-- | Learning rate
data DumpLearningRate a = DumpLearningRate

mLR :: (DType dtype, MonadIO m) => m (DumpLearningRate dtype)
mLR = return DumpLearningRate

instance EvalMetricMethod DumpLearningRate where
    evaluate _ _ _ = return ()
    format _ = do
        lr <- lift $ use stat_last_lr
        return $ printf "<lr: %0.6f>" lr

data ListOfMetric a dt where
    MNil :: ListOfMetric '[] dt
    (:+) :: (EvalMetricMethod a) => a dt -> ListOfMetric as dt -> ListOfMetric (a ': as) dt

instance EvalMetricMethod (ListOfMetric '[]) where
    evaluate _ _ _ = return ()
    format _ = return ""

instance (EvalMetricMethod a, EvalMetricMethod (ListOfMetric as)) => EvalMetricMethod (ListOfMetric (a ': as)) where
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

(#+) :: (Monad m, EvalMetricMethod a, DType dt) 
     => m (a dt) -> m (ListOfMetric as dt) -> m (ListOfMetric (a ': as) dt)
(#+) = liftM2 (:+)

(##) :: (Monad m, EvalMetricMethod a, EvalMetricMethod b, DType dt) 
     => m (a dt) -> m (b dt) -> m (ListOfMetric ('[a, b]) dt)
(##) a b = a #+ b #+ (return MNil)
