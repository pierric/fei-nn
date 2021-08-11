{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE InstanceSigs        #-}
{-# LANGUAGE RecordWildCards     #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeOperators       #-}
module MXNet.NN.EvalMetric where

import           Control.Lens
import           Formatting                   (fixed, int, sformat, stext, (%))
import           RIO                          hiding (view)
import qualified RIO.HashMap                  as M
import           RIO.List                     (headMaybe)
import           RIO.State                    (execState)
import qualified RIO.Text                     as T
import qualified RIO.Vector.Storable          as SV
import qualified RIO.Vector.Storable.Partial  as SV (head)

import           MXNet.Base
import           MXNet.Base.Operators.Tensor  (_norm)
import           MXNet.Base.Tensor.Functional

-- | Keep a double value and its running mean
data RunningValue = RunningValue
    { _rv_accumlated :: (Int, Double)
    , _rv_last       :: (Int, Double)
    }

makeLenses ''RunningValue

-- | Create a new 'RunningValue'
newRunningValueRef :: MonadIO m => m (IORef RunningValue)
newRunningValueRef = liftIO $ newIORef $ RunningValue (0, 0) (0, 0)

-- | Update the 'RunningValue'
updateRunningValueRef :: MonadIO m
                      => IORef RunningValue
                      -> Maybe Int
                      -> Double
                      -> m ()
updateRunningValueRef ref a b = liftIO $ modifyIORef ref $ execState $ do
    let a' = fromMaybe 1 a
    rv_accumlated . _1 += a'
    rv_accumlated . _2 += b
    rv_last .= (a', b)

-- | Get the running mean of the value
getRunningValueSmoothed :: MonadIO m => IORef RunningValue -> m Double
getRunningValueSmoothed ref = liftIO $ do
    (n, v) <- readIORef ref <&> view rv_accumlated
    return $ v / (fromIntegral n + 1e-5)

-- | Get the last value
getRunningValueLast :: MonadIO m => IORef RunningValue -> m Double
getRunningValueLast ref = liftIO $ do
    (n, v) <- readIORef ref <&> view rv_last
    return $ v / (fromIntegral n + 1e-5)

-- | Abstract Evaluation type class
class EvalMetricMethod metric where
    data MetricData metric a
    newMetric :: (MonadIO m, FloatDType a, HasCallStack)
              => Text                          -- phase name
              -> metric a                      -- tag
              -> m (MetricData metric a)
    metricUpdate :: forall a m . (MonadIO m, FloatDType a, HasCallStack)
                 => MetricData metric a           -- evaluation metric
                 -> M.HashMap Text (NDArray a)    -- network bindings
                 -> [NDArray a]                   -- output of the network
                 -> m (M.HashMap Text Double)
    metricName   :: MetricData metric a -> Text
    metricValue  :: (MonadIO m, FloatDType a, HasCallStack) => MetricData metric a -> m Double
    metricValueLast :: (MonadIO m, FloatDType a, HasCallStack) => MetricData metric a -> m Double
    metricFormat :: (MonadIO m, FloatDType a, HasCallStack) => MetricData metric a -> m Text
    metricFormat m = do
        name  <- pure (metricName m)
        value <- metricValue m
        return $ sformat ("<" % stext % ": " % fixed 4 % ">") name value

-- | Basic metric for loss. This metric will sum up the value of the loss tensor.
data Loss a = Loss
    { _mtr_loss_name     :: Maybe Text
    , _mtr_loss_get_loss :: [NDArray a] -> NDArray a
    }

instance EvalMetricMethod Loss where
    data MetricData Loss a = LossP (Loss a) Text (IORef RunningValue)
    newMetric phase conf = LossP conf phase <$> newRunningValueRef

    metricUpdate mtr@(LossP Loss{..} _ ref) bindings outputs = liftIO $ do
        out <- toCPU $ _mtr_loss_get_loss outputs
        value <- realToFrac . SV.sum <$> toVector out
        updateRunningValueRef ref Nothing value

        value_smoothed <- getRunningValueSmoothed ref
        return $ M.singleton (metricName mtr) value_smoothed

    metricName (LossP Loss{..} phase _) =
        let name = fromMaybe "loss" _mtr_loss_name
         in sformat (stext % "_" % stext) phase name

    metricValue (LossP _ _ ref) = getRunningValueSmoothed ref
    metricValueLast (LossP _ _ ref) = getRunningValueLast ref

-- | Basic evaluation - accuracy
data AccuracyPredType = PredByThreshold Double
    | PredByArgmax
    | PredByArgmaxAt Int
data Accuracy a = Accuracy
    { _mtr_acc_name :: Maybe Text
    , _mtr_acc_type :: AccuracyPredType
    , _mtr_acc_min_value :: Double -- | to filter values less than min
    , _mtr_acc_get_prob :: M.HashMap Text (NDArray a) -> [NDArray a] -> NDArray a
    , _mtr_acc_get_gt :: M.HashMap Text (NDArray a) -> [NDArray a] -> NDArray a
    }

instance EvalMetricMethod Accuracy where
    data MetricData Accuracy a = AccuracyP (Accuracy a) Text (IORef RunningValue)
    newMetric phase conf = AccuracyP conf phase <$> newRunningValueRef

    metricUpdate :: forall a m . (MonadIO m, FloatDType a, HasCallStack)
                 => MetricData Accuracy a
                 -> M.HashMap Text (NDArray a)
                 -> [NDArray a]
                 -> m (M.HashMap Text Double)
    metricUpdate mtr@(AccuracyP Accuracy{..} phase ref) bindings outputs = liftIO $ do
        out <- toCPU $ _mtr_acc_get_prob bindings outputs
        lbl <- toCPU $ _mtr_acc_get_gt bindings outputs

        out <- case _mtr_acc_type of
                 PredByThreshold thr -> geqScalar thr out >>= castToFloat @(DTypeName a)
                 PredByArgmax        -> argmax out (Just (-1))False  >>= castToFloat @(DTypeName a)
                 PredByArgmaxAt axis -> argmax out (Just axis) False >>= castToFloat @(DTypeName a)

        valid   <- geqScalar _mtr_acc_min_value lbl
        correct <- eq_ out lbl >>= and_ valid >>= castToFloat @(DTypeName a)
        num_correct <- SV.head <$> (toVector =<< sum_ correct Nothing False)
        valid   <- castToFloat @(DTypeName a) valid
        num_valid   <- SV.head <$> (toVector =<< sum_ valid Nothing False)

        updateRunningValueRef ref (Just $ floor num_valid) (realToFrac num_correct)

        value_smoothed <- (100 *) <$> getRunningValueSmoothed ref
        return $ M.singleton (metricName mtr) value_smoothed

    metricName (AccuracyP Accuracy{..} phase _) =
        let name = fromMaybe "acc" _mtr_acc_name
         in sformat (stext % "_" % stext) phase name

    metricValue (AccuracyP _ _ ref) = (100 *) <$> getRunningValueSmoothed ref
    metricValueLast (AccuracyP _ _ ref) = (100 *) <$> getRunningValueLast ref

-- | Basic evaluation - vector norm
data Norm a = Norm
    { _mtr_norm_name :: Maybe Text
    , _mtr_norm_ord :: Int
    , _mtr_norm_get_array :: M.HashMap Text (NDArray a) -> [NDArray a] -> NDArray a
    }

instance EvalMetricMethod Norm where
    data MetricData Norm a = NormP (Norm a) Text (IORef RunningValue)
    newMetric phase conf = NormP conf phase <$> newRunningValueRef

    metricUpdate mtr@(NormP Norm{..} phase ref) bindings preds = liftIO $ do
        array <- toCPU $ _mtr_norm_get_array bindings preds
        shape <- ndshape array

        case headMaybe shape of
          Nothing -> throwString (T.unpack (metricName mtr) ++ " is scalar.")
          Just batch_size -> do
              norm  <- prim _norm (#data := array .& #ord := _mtr_norm_ord .& Nil)
              norm  <- SV.head <$> toVector norm
              updateRunningValueRef ref (Just batch_size) (realToFrac norm)

              value_smoothed <- getRunningValueSmoothed ref
              return $ M.singleton (metricName mtr) value_smoothed

    metricName (NormP Norm{..} phase _) =
        let lk = sformat ("_L" % int) _mtr_norm_ord
            name = fromMaybe lk _mtr_norm_name
         in sformat (stext % "_" % stext) phase name

    metricValue (NormP _ _ ref) = getRunningValueSmoothed ref
    metricValueLast (NormP _ _ ref) = getRunningValueLast ref

-- | Basic evaluation - cross entropy
data CrossEntropy a = CrossEntropy
    { _mtr_ce_name     :: Maybe Text
    , _mtr_ce_gt_clsid :: Bool
    , _mtr_ce_get_prob :: M.HashMap Text (NDArray a) -> [NDArray a] -> NDArray a
    , _mtr_ce_get_gt   :: M.HashMap Text (NDArray a) -> [NDArray a] -> NDArray a
    }

instance EvalMetricMethod CrossEntropy where
    data MetricData CrossEntropy a = CrossEntropyP (CrossEntropy a) Text (IORef RunningValue)
    newMetric phase conf = CrossEntropyP conf phase <$> newRunningValueRef

    metricUpdate :: forall a m . (MonadIO m, FloatDType a, HasCallStack)
                 => MetricData CrossEntropy a
                 -> M.HashMap Text (NDArray a)
                 -> [NDArray a]
                 -> m (M.HashMap Text Double)
    metricUpdate mtr@(CrossEntropyP CrossEntropy{..} phase ref) bindings preds = liftIO $ do
        prob <- toCPU $ _mtr_ce_get_prob bindings preds
        gt   <- toCPU $ _mtr_ce_get_gt   bindings preds

        (loss, num_valid) <-
                if _mtr_ce_gt_clsid
                then do
                    -- when the gt labels are class id,
                    --  prob: (a, ..., b, num_classes)
                    --  gt: (a,..,b)
                    -- dim(prob) = dim(gt) + 1
                    -- The last dimension serves as the prob dist.
                    -- We pickup the prob at the label specified class.
                    cls_prob  <- log2_ =<< addScalar 1e-5 =<< pick (Just (-1)) gt prob
                    weights   <- geqScalar 0 gt >>= castToFloat @(DTypeName a)
                    cls_prob  <- mul_ cls_prob weights
                    nloss     <- toVector =<< sum_ cls_prob Nothing False
                    num_valid <- toVector =<< sum_ weights  Nothing False
                    return (negate (SV.head nloss), SV.head num_valid)
                else do
                    -- when the gt are onehot vector
                    --   prob: (a, .., b, num_classes)
                    --   gt:   (a, .., b, num_classes)
                    -- dim(prob) == dim(gt)
                    term1     <- mul_ gt =<< log2_ =<< addScalar 1e-5 prob
                    a         <- log2_   =<< addScalar 1e-5 =<< rsubScalar 1 prob
                    term2     <- mul_ a  =<< rsubScalar 1 gt
                    weights   <- geqScalar 0 gt >>= castToFloat @(DTypeName a)
                    nloss     <- mul_ weights =<< add_ term1 term2
                    nloss     <- toVector =<< sum_ nloss Nothing False
                    num_valid <- toVector =<< sum_ weights Nothing False
                    return (negate (SV.head nloss), SV.head num_valid)

        updateRunningValueRef ref (Just $ floor num_valid) (realToFrac loss)

        value_smoothed <- getRunningValueSmoothed ref
        return $ M.singleton (metricName mtr) value_smoothed

    metricName (CrossEntropyP CrossEntropy{..} phase _) =
        let name = fromMaybe "ce" _mtr_ce_name
         in sformat (stext % "_" % stext) phase name

    metricValue (CrossEntropyP _ _ ref) = getRunningValueSmoothed ref
    metricValueLast (CrossEntropyP _ _ ref) = getRunningValueLast ref


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
        v <- metricValueLast a
        w <- metricsToList b
        return $ (n, v) : w
