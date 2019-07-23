{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
module MXNet.NN.EvalMetric where

import Data.IORef
-- import Data.Dynamic
import qualified Data.HashMap.Strict as M
import Control.Monad.Trans.Resource (MonadThrow(..))
import Control.Monad
import Control.Monad.IO.Class (MonadIO, liftIO)
import Text.Printf (printf)
import qualified Data.Vector.Storable as SV

import MXNet.Base
import qualified MXNet.Base.Operators.NDArray as A
import MXNet.NN.Types 

-- | Abstract Evaluation type class
class EvalMetricMethod metric where
    data MetricData metric a
    newMetric :: (MonadIO m, DType a) 
             => String                        -- phase name
             -> metric a                      -- tag
             -> m (MetricData metric a)
    evaluate :: (MonadIO m, DType a)
             => MetricData metric a           -- evaluation metric
             -> M.HashMap String (NDArray a)  -- network bindings
             -> NDArray a                     -- output of the network
             -> m (M.HashMap String Double)
    format   :: (MonadIO m, DType a) => MetricData metric a -> m String


-- | Basic evaluation - accuracy
data Accuracy a = Accuracy String

instance EvalMetricMethod Accuracy where
    data MetricData Accuracy a = AccuracyData String String (IORef Int) (IORef Int) 
    newMetric phase (Accuracy label) = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ AccuracyData phase label a b
    evaluate (AccuracyData phase label cntRef sumRef) bindings output = do
        liftIO $ compute output (bindings M.! label)
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        let acc = fromIntegral s / fromIntegral n
        return $ M.singleton (phase ++ "_acc") acc
      where
        compute preds@(NDArray preds_hdl) lbl = do
            [pred_cat_hdl] <- A.argmax (#data := preds_hdl .& #axis := Just 1 .& Nil)
            pred_cat <- toVector (NDArray pred_cat_hdl)
            real_cat <- toVector lbl

            batch_size:_ <- ndshape preds
            let correct = SV.length $ SV.filter id $ SV.zipWith (==) pred_cat real_cat
            modifyIORef sumRef (+ correct)
            modifyIORef cntRef (+ batch_size)
    format (AccuracyData _ _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return $ printf "<Accuracy: %0.2f>" (100 * fromIntegral s / fromIntegral n :: Float)

-- | Basic evaluation - cross entropy 
data CrossEntropy a = CrossEntropy String

copyTo :: DType a => NDArray a -> NDArray a -> IO ()
copyTo (NDArray dst) (NDArray src) = A._copyto_upd [dst] (#data := src .& Nil)

instance EvalMetricMethod CrossEntropy where
    data MetricData CrossEntropy a = CrossEntropyData String String (IORef Int) (IORef Float)
    newMetric phase (CrossEntropy label) = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ CrossEntropyData phase label a b
    -- | evaluate the log-loss. 
    -- preds is of shape (batch_size, num_category), each element along the second dimension gives the probability of the category.
    -- label is of shape (batch_size,), each element gives the category number.
    evaluate (CrossEntropyData phase label cntRef sumRef) bindings output = do
        liftIO $ compute output (bindings M.! label) 
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        let loss = realToFrac s / fromIntegral n
        return $ M.singleton (phase ++ "_loss") loss
      where
        compute preds lbl@(NDArray labelHandle) = do
            shp1 <- ndshape preds
            shp2 <- ndshape lbl
            when (length shp1 /= 2 || length shp2 /= 1 || head shp1 /= head shp2) (throwM $ MismatchedShapeInEval shp1 shp2)
            -- before call pick, we have to make sure preds and label 
            -- are in the same context
            NDArray preds_may_copy <- do
                c1 <- context preds
                c2 <- context lbl
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
            modifyIORef sumRef (+ (negate $ loss SV.! 0))
            modifyIORef cntRef (+ head shp1)
    format (CrossEntropyData _ _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return $ printf "<CrossEntropy: %0.3f>" (realToFrac s / fromIntegral n :: Float)

data ListOfMetric ms a where
    MNil :: ListOfMetric '[] a
    (:*) :: (EvalMetricMethod m) => m a -> ListOfMetric ms a -> ListOfMetric (m ': ms) a

instance EvalMetricMethod (ListOfMetric '[]) where
    data MetricData (ListOfMetric '[]) a = MNilData
    newMetric _ _ = return MNilData
    evaluate _ _ _ = return M.empty
    format _ = return ""

instance (EvalMetricMethod m, EvalMetricMethod (ListOfMetric ms)) => EvalMetricMethod (ListOfMetric (m ': ms)) where
    data MetricData (ListOfMetric (m ': ms)) a = MCompositeData (MetricData m a) (MetricData (ListOfMetric ms) a)
    newMetric phase (a :* as) = MCompositeData <$> (newMetric phase a) <*> (newMetric phase as)
    evaluate (MCompositeData a as) bindings output = do
        m1 <- evaluate a  bindings output
        m2 <- evaluate as bindings output
        return $ M.union m1 m2
    format (MCompositeData a as) = do
        s1 <- format a
        s2 <- format as
        return $ s1 ++ " " ++ s2

infixr 9 :*
