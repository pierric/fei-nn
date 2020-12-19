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
    metricUpdate :: (MonadIO m, FloatDType a, HasCallStack)
                 => MetricData metric a           -- evaluation metric
                 -> M.HashMap Text (NDArray a)    -- network bindings
                 -> [NDArray a]                   -- output of the network
                 -> m (M.HashMap Text Double)
    metricName   :: MetricData metric a -> Text
    metricValue  :: (MonadIO m, FloatDType a, HasCallStack) => MetricData metric a -> m Double
    metricFormat :: (MonadIO m, FloatDType a, HasCallStack) => MetricData metric a -> m Text
    metricFormat m = do
        name  <- pure (metricName m)
        value <- metricValue m
        return $ sformat ("<" % stext % ": " % fixed 4 % ">") name value


-- | Basic evaluation - accuracy
data AccuracyPredType = PredByThreshold Float
    | PredByArgmax
    | PredByArgmaxAt Int
data Accuracy a = Accuracy
    { _mtr_acc_name :: Maybe Text
    , _mtr_acc_type :: AccuracyPredType
    , _mtr_acc_min_value :: Float  -- | to filter values less than min
    , _mtr_acc_get_prob :: M.HashMap Text (NDArray a) -> [NDArray a] -> NDArray a
    , _mtr_acc_get_gt :: M.HashMap Text (NDArray a) -> [NDArray a] -> NDArray a
    }

instance EvalMetricMethod Accuracy where
    data MetricData Accuracy a = AccuracyPriv (Accuracy a) Text (IORef Int) (IORef Int)
    newMetric phase conf = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ AccuracyPriv conf phase a b

    metricUpdate mtr@(AccuracyPriv Accuracy{..} phase cntRef sumRef) bindings outputs = liftIO $ do
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

        value <- metricValue mtr
        return $ M.singleton (metricName mtr) value

    metricName (AccuracyPriv Accuracy{..} phase _ _) =
        let name = fromMaybe "acc" _mtr_acc_name
         in sformat (stext % "_" % stext) phase name

    metricValue (AccuracyPriv _ _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return (100 * fromIntegral s / fromIntegral n)

-- | Basic evaluation - vector norm
data Norm a = Norm
    { _mtr_norm_name :: Maybe Text
    , _mtr_norm_ord :: Int
    , _mtr_norm_get_array :: M.HashMap Text (NDArray a) -> [NDArray a] -> NDArray a
    }

instance EvalMetricMethod Norm where
    data MetricData Norm a = NormPriv (Norm a) Text (IORef Int) (IORef Double)
    newMetric phase conf = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ NormPriv conf phase a b

    metricUpdate mtr@(NormPriv Norm{..} phase cntRef sumRef) bindings preds = liftIO $ do
        array <- toCPU $ _mtr_norm_get_array bindings preds

        norm <- prim _norm (#data := array .& #ord := _mtr_norm_ord .& Nil)
        norm <- SV.head <$> toVector norm
        batch_size :| _ <- ndshape array

        modifyIORef' sumRef (+ realToFrac norm)
        modifyIORef' cntRef (+ batch_size)

        value <- metricValue mtr
        return $ M.singleton (metricName mtr) value

    metricName (NormPriv Norm{..} phase _ _) =
        let lk = sformat ("_L" % int) _mtr_norm_ord
            name = fromMaybe lk _mtr_norm_name
         in sformat (stext % "_" % stext) phase name

    metricValue (NormPriv _ _ cntRef sumRef) = liftIO $ do
        s <- readIORef sumRef
        n <- readIORef cntRef
        return $ realToFrac s / fromIntegral n

-- | Basic evaluation - cross entropy
data CrossEntropy a = CrossEntropy
    { _mtr_ce_name     :: Maybe Text
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

    metricUpdate mtr@(CrossEntropyPriv CrossEntropy{..} phase cntRef sumRef) bindings preds = liftIO $ do
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

        value <- metricValue mtr
        return $ M.singleton (metricName mtr) value

    metricName (CrossEntropyPriv CrossEntropy{..} phase _ _) =
        let name = fromMaybe "ce" _mtr_ce_name
         in sformat (stext % "_" % stext) phase name

    metricValue (CrossEntropyPriv _ _ cntRef sumRef) = liftIO $ do
        s <- readIORef sumRef
        n <- readIORef cntRef
        return $ realToFrac s / fromIntegral n


data ListOfMetric ms a where
    MNil :: ListOfMetric '[] a
    (:*) :: (EvalMetricMethod m) => m a -> ListOfMetric ms a -> ListOfMetric (m ': ms) a

instance EvalMetricMethod (ListOfMetric '[]) where
    data MetricData (ListOfMetric '[]) a = MNilData
    newMetric _ _  = return MNilData
    metricName  _  = error "Empty metric"
    metricValue _  = error "Empty metric"
    metricUpdate _ _ _ = return M.empty
    metricFormat _     = return ""

instance (EvalMetricMethod m, EvalMetricMethod (ListOfMetric ms)) => EvalMetricMethod (ListOfMetric (m ': ms)) where
    data MetricData (ListOfMetric (m ': ms)) a = MCompositeData (MetricData m a) (MetricData (ListOfMetric ms) a)
    newMetric phase (a :* as) = MCompositeData <$> (newMetric phase a) <*> (newMetric phase as)
    metricUpdate (MCompositeData a as) bindings output = do
        m1 <- metricUpdate a  bindings output
        m2 <- metricUpdate as bindings output
        return $ M.union m1 m2
    metricName _  = error "List of metrics"
    metricValue _ = error "List of metrics"
    metricFormat (MCompositeData a as) = do
        s1 <- metricFormat a
        s2 <- metricFormat as
        return $ T.concat [s1, " ", s2]

infixr 9 :*

class MetricsToList ms where
    metricsToList :: (MonadIO m, FloatDType a) => MetricData (ListOfMetric ms) a -> m [(Text, Double)]

instance MetricsToList '[] where
    metricsToList MNilData = return []

instance (EvalMetricMethod m, MetricsToList n) => MetricsToList (m ': n) where
    metricsToList (MCompositeData a b) = do
        n <- pure $ metricName a
        v <- metricValue a
        w <- metricsToList b
        return $ (n, v) : w
