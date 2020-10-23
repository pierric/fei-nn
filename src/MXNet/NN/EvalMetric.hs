{-# LANGUAGE DataKinds         #-}
{-# LANGUAGE FlexibleContexts  #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RecordWildCards   #-}
{-# LANGUAGE TemplateHaskell   #-}
{-# LANGUAGE TypeOperators     #-}
module MXNet.NN.EvalMetric where

import           Formatting                  (fixed, int, sformat, stext, (%))
import           RIO
import qualified RIO.HashMap                 as M
import qualified RIO.HashMap.Partial         as M ((!))
import qualified RIO.NonEmpty                as RNE
import qualified RIO.Text                    as T
import qualified RIO.Vector.Storable         as SV
import qualified RIO.Vector.Storable.Partial as SV (head)

import           MXNet.Base
import           MXNet.Base.Operators.Tensor (_norm)
import           MXNet.NN.Layer
import           MXNet.NN.Types

-- | Abstract Evaluation type class
class EvalMetricMethod metric where
    data MetricData metric a
    newMetric :: (MonadIO m, FloatDType a, HasCallStack)
             => Text                          -- phase name
             -> metric a                      -- tag
             -> m (MetricData metric a)
    evalMetric :: (MonadIO m, FloatDType a, HasCallStack)
             => MetricData metric a           -- evaluation metric
             -> M.HashMap Text (NDArray a)    -- network bindings
             -> [NDArray a]                   -- output of the network
             -> m (M.HashMap Text Double)
    formatMetric :: (MonadIO m, FloatDType a, HasCallStack) => MetricData metric a -> m Text


-- | Basic evaluation - accuracy
data AccuracyPredType = PredByThreshold Float
    | PredByArgmax
    | PredByArgmaxAt Int
data Accuracy a = Accuracy
    { _mtr_acc_name :: Text
    , _mtr_acc_type :: AccuracyPredType
    , _mtr_acc_min_value :: Float
    , _mtr_acc_get_prob :: M.HashMap Text (NDArray a) -> [NDArray a] -> NDArray a
    , _mtr_acc_get_gt :: M.HashMap Text (NDArray a) -> [NDArray a] -> NDArray a
    }

instance EvalMetricMethod Accuracy where
    data MetricData Accuracy a = AccuracyPriv (Accuracy a) Text (IORef Int) (IORef Int)
    newMetric phase conf = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ AccuracyPriv conf phase a b
    evalMetric (AccuracyPriv Accuracy{..} phase cntRef sumRef) bindings outputs = liftIO $ do
        out <- toCPU $ _mtr_acc_get_prob bindings outputs
        lbl <- toCPU $ _mtr_acc_get_gt bindings outputs

        out <- case _mtr_acc_type of
          PredByThreshold thr -> geqScalar thr out
          PredByArgmax        -> argmax out (Just (-1))False
          PredByArgmaxAt axis -> argmax out (Just axis) False

        valid   <- geqScalar _mtr_acc_min_value lbl
        correct <- and_ valid =<< eq_ out lbl
        num_correct <- SV.head <$> (toVector =<< sum_ correct Nothing False)
        num_valid   <- SV.head <$> (toVector =<< sum_ valid Nothing False)

        modifyIORef sumRef (+ floor num_correct)
        modifyIORef cntRef (+ floor num_valid)

        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        let acc = fromIntegral s / fromIntegral n
        return $ M.singleton ((sformat (stext % "_" % stext % "_acc") phase _mtr_acc_name)) acc
    formatMetric (AccuracyPriv Accuracy{..} _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return $ sformat
            ("<" % stext % "-Acc" % ": " % fixed 4 % ">")
            _mtr_acc_name
            (100 * fromIntegral s / fromIntegral n :: Float)

-- | Basic evaluation - vector norm
data Norm a = Norm
    { _mtr_norm_name :: Text
    , _mtr_norm_ord :: Int
    , _mtr_norm_get_array :: M.HashMap Text (NDArray a) -> [NDArray a] -> NDArray a
    }

instance EvalMetricMethod Norm where
    data MetricData Norm a = NormPriv (Norm a) Text (IORef Int) (IORef Double)
    newMetric phase conf = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ NormPriv conf phase a b

    evalMetric (NormPriv Norm{..} phase cntRef sumRef) bindings preds = liftIO $ do
        array <- toCPU $ _mtr_norm_get_array bindings preds

        norm <- prim _norm (#data := array .& #ord := _mtr_norm_ord .& Nil)
        norm <- SV.head <$> toVector norm
        batch_size :| _ <- ndshape array

        modifyIORef' sumRef (+ realToFrac norm)
        modifyIORef' cntRef (+ batch_size)

        s <- readIORef sumRef
        n <- readIORef cntRef
        let val = s / fromIntegral n
        return $ M.singleton (sformat (stext % "_" % stext % "_norm") phase _mtr_norm_name) val

    formatMetric (NormPriv Norm{..} _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return $ sformat ("<" % stext % "-L" % int % ": " % fixed 4 % ">")
                         _mtr_norm_name
                         _mtr_norm_ord
                         (realToFrac s / fromIntegral n :: Float)

-- | Basic evaluation - cross entropy
data CrossEntropy a = CrossEntropy
    { _mtr_ce_name     :: Text
    , _mtr_ce_gt_clsid :: Bool
    , _mtr_ce_get_prob :: M.HashMap Text (NDArray a) -> [NDArray a] -> NDArray a
    , _mtr_ce_get_gt   :: M.HashMap Text (NDArray a) -> [NDArray a] -> NDArray a
    }

instance EvalMetricMethod CrossEntropy where
    data MetricData CrossEntropy a = CrossEntropyPriv (CrossEntropy a) Text (IORef Int) (IORef Double)
    newMetric phase conf = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ CrossEntropyPriv conf phase a b

    evalMetric (CrossEntropyPriv CrossEntropy{..} phase cntRef sumRef) bindings preds = liftIO $ do
        prob <- toCPU $ _mtr_ce_get_prob  bindings preds
        gt   <- toCPU $ _mtr_ce_get_gt    bindings preds

        (loss, num_valid) <-
            if _mtr_ce_gt_clsid
            then do
                -- when the gt labels are class id,
                --  prob: (a, ..., b, num_classes)
                --  gt: (a,..,b)
                -- dim(prob) = dim(gt) + 1
                -- The last dimension serves as the prob dist.
                -- We pickup the prob at the label specified class.
                cls_prob  <- log2_ =<< addScalar 1e-5 =<< pickI gt prob
                weights   <- geqScalar 0 gt
                cls_prob  <- mul_ cls_prob weights
                nloss     <- toVector =<< sum_ cls_prob Nothing False
                num_valid <- toVector =<< sum_ weights Nothing False
                return (negate (SV.head nloss), SV.head num_valid)
            else do
                -- when the gt are onehot vector
                --   prob: (a, .., b, num_classes)
                --   gt:   (a, .., b, num_classes)
                -- dim(prob) == dim(gt)
                term1     <- mul_ gt =<< log2_ =<< addScalar 1e-5 prob
                a         <- log2_   =<< addScalar 1e-5 =<< rsubScalar 1 prob
                term2     <- mul_ a  =<< rsubScalar 1 gt
                weights   <- geqScalar 0 gt
                nloss     <- mul_ weights =<< add_ term1 term2
                nloss     <- toVector =<< sum_ nloss Nothing False
                num_valid <- toVector =<< sum_ weights Nothing False
                return (negate (SV.head nloss), SV.head num_valid)

        modifyIORef' sumRef (+ realToFrac loss)
        modifyIORef' cntRef (+ floor num_valid)

        s <- readIORef sumRef
        n <- readIORef cntRef
        let val = s / fromIntegral n
        return $ M.singleton (sformat (stext % "_" % stext % "_ce") phase _mtr_ce_name) val

    formatMetric (CrossEntropyPriv CrossEntropy{..} _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return $ sformat ("<" % stext % "-CE" % ": " % fixed 4 % ">")
                         _mtr_ce_name
                         (realToFrac s / fromIntegral n :: Float)


data ListOfMetric ms a where
    MNil :: ListOfMetric '[] a
    (:*) :: (EvalMetricMethod m) => m a -> ListOfMetric ms a -> ListOfMetric (m ': ms) a

instance EvalMetricMethod (ListOfMetric '[]) where
    data MetricData (ListOfMetric '[]) a = MNilData
    newMetric _ _ = return MNilData
    evalMetric _ _ _ = return M.empty
    formatMetric _ = return ""

instance (EvalMetricMethod m, EvalMetricMethod (ListOfMetric ms)) => EvalMetricMethod (ListOfMetric (m ': ms)) where
    data MetricData (ListOfMetric (m ': ms)) a = MCompositeData (MetricData m a) (MetricData (ListOfMetric ms) a)
    newMetric phase (a :* as) = MCompositeData <$> (newMetric phase a) <*> (newMetric phase as)
    evalMetric (MCompositeData a as) bindings output = do
        m1 <- evalMetric a  bindings output
        m2 <- evalMetric as bindings output
        return $ M.union m1 m2
    formatMetric (MCompositeData a as) = do
        s1 <- formatMetric a
        s2 <- formatMetric as
        return $ T.concat [s1, " ", s2]

infixr 9 :*
