module MXNet.NN.Module where

import RIO hiding (evaluate)
import qualified RIO.HashMap as M
import qualified RIO.HashSet as S
import qualified RIO.NonEmpty as RNE ((<|))
import RIO.List (zipWith3)
import Control.Lens.Setter ((.=), (+=), (%=))
import Control.Lens (use, (^?!), ix)
import Formatting (sformat, (%), int, stext)

import MXNet.Base
import qualified MXNet.Base.Operators.NDArray as A
import MXNet.NN.Types
import MXNet.NN.TaggedState (Tagged(..), untag)
import MXNet.NN.Optimizer (Optimizer, optimize)
import MXNet.NN.DataIter.Class (Dataset(..), DatasetProp(..))
import MXNet.NN.EvalMetric (EvalMetricMethod(..), MetricData)


data UnkownShapeOrScalar = UnkownShapeOrScalar Text
    deriving (Typeable, Show)
instance Exception UnkownShapeOrScalar

initialize :: forall tag dty. DType dty => Symbol dty -> Config dty -> IO (TaggedModuleState dty tag)
initialize symbol config = do
     -- give a initial batch_size = 1 for the placeholders
    let spec1 = M.map (shapeCons 1) $ M.difference input_shapes initializers
        spec2 = initializers
        dinit = config ^. cfg_default_initializer
        cxt   = config ^. cfg_context
        -- remove any input or label from the fixed set
        fixed = (config ^. cfg_fixed_params) `S.difference`
                (S.fromList $ M.keys input_shapes ++ label_names)

    (args, _, auxs, r) <- inferShape symbol (M.toList spec1)
    arg_with_shp <- M.fromList <$> mapM checkTensorShape args
    aux_with_shp <- M.fromList <$> mapM checkTensorShape auxs
    ---------------------
    -- important! labels should be merged into placeholders,
    -- otherwise the labels are considered to have gradient.
    ---------------------
    let lbl_with_shp = M.filterWithKey (\k _ -> k `elem` label_names) arg_with_shp
        phl_with_shp = M.map _shape_nonempty spec1 `M.union` lbl_with_shp
    placeholders <- mapM (flip makeEmptyNDArray cxt) phl_with_shp

    arg_tensors <- M.traverseWithKey (initI placeholders fixed spec2 dinit) arg_with_shp
    aux_tensors <- M.traverseWithKey (initA dinit) aux_with_shp

    let params = arg_tensors `M.union` aux_tensors
    executor <- bind symbol cxt params True

    return $ Tagged $ ModuleState {
        _mod_symbol       = symbol,
        _mod_input_shapes = phl_with_shp,
        _mod_params       = params,
        _mod_context      = cxt,
        _mod_executor     = executor,
        _mod_statistics   = Statistics 0 0,
        _mod_scores       = M.empty,
        _mod_fixed_args   = fixed
    }
  where
    input_shapes = config ^. cfg_data
    label_names  = config ^. cfg_label
    initializers = config ^. cfg_initializers
    -- initialize input symbols.
    -- placeholders are backed by empty NDArray,
    -- other input symbols are initialized by an initializer.
    initI placeholder fixed spec2 dinit inp shp =
        case M.lookup inp placeholder of
            Just in_arg -> do
                return $ ParameterV in_arg
            Nothing -> do
                arg_in <- case M.lookup inp spec2 of
                    Just cinit -> cinit inp shp (_cfg_context config)
                    Nothing    -> dinit inp shp (_cfg_context config)
                if S.member inp fixed
                    then return $ ParameterF arg_in
                    else do
                        arg_gr <- makeEmptyNDArray shp (_cfg_context config)
                        return $ ParameterG arg_in arg_gr
    -- initialize auxiliary symbols.
    initA dinit aux shp = do
        arg_aux <- dinit aux shp (_cfg_context config)
        return $ ParameterA arg_aux

    checkTensorShape (name, SScalar)   = throwIO $ UnkownShapeOrScalar name
    checkTensorShape (name, STensor s) = return (name, s)

bind :: DType dty => Symbol dty -> Context -> M.HashMap Text (Parameter dty) -> Bool -> IO (Executor dty)
bind symbol context params trainable = do
    argnames <- listArguments symbol
    auxnames <- listAuxiliaryStates symbol
    -- sanity check
    assert (S.fromList (M.keys params) == S.fromList (argnames ++ auxnames)) (return ())
    -- the parameters to bind should be arranged in the same order as the argnames
    let num_args = length argnames
        arg_all  = map ((params ^?!) . ix) argnames
        arg_in   = map (\case {
                    ParameterV t -> t;
                    ParameterF t -> t;
                    ParameterG t _ -> t;
                    ParameterA _ -> error "auxiliary parameter shouldn't occur"
                    }) arg_all
        arg_gr_w_req = if trainable then map (\case {ParameterG _ t -> Just (t, 1); _ -> Nothing}) arg_all
                                    else replicate num_args Nothing
        aux_arg_aux  = map (_param_aux . (params ^?!) . ix) auxnames
    execBind symbol context arg_in arg_gr_w_req aux_arg_aux

adapt :: (DType dty, MonadIO m) => M.HashMap Text (NDArray dty) -> Module tag dty m ()
adapt inputs = do
    symbol  <- use $ untag . mod_symbol
    exec    <- use $ untag . mod_executor
    context <- use $ untag . mod_context
    fixed   <- use $ untag . mod_fixed_args
    -- shapes  <- M.toList <$> use mod_input_shapes
    -- shapes' <- lift $ mapM (runKleisli $ second $ Kleisli ndshape) $ M.toList inputs
    shapes  <- use $ untag . mod_input_shapes
    shapes' <- liftIO $ mapM ndshape inputs

    -- reshape the executor (and arg, aux arrays)
    when (shapes /= shapes') $ do
        (args, grads, auxs, exec) <- liftIO $ execReshapeEx exec True True context (M.toList shapes')
        arg_names <- liftIO $ listArguments symbol
        aux_names <- liftIO $ listAuxiliaryStates symbol
        let buildArg key a Nothing  | S.member key fixed = (key, ParameterF a)
            buildArg key a Nothing  | otherwise = (key, ParameterV a)
            buildArg key a (Just b) = (key, ParameterG a b)
            buildAux key a = (key, ParameterA a)
            arg_ndarrs = M.fromList $ zipWith3 buildArg arg_names args grads
            aux_ndarrs = M.fromList $ zipWith buildAux aux_names auxs
        untag . mod_executor .= exec
        untag . mod_input_shapes .= shapes'
        untag . mod_params   .= M.union arg_ndarrs aux_ndarrs

    -- copy the ndarray
    targets <- use $ untag . mod_params
    forM_ (M.toList inputs) $ \ (k, src) -> liftIO $ do
        case M.lookup k targets of
          Just (ParameterV dst) -> A._copyto_upd [unNDArray dst] (#data := unNDArray src .& Nil)
          _ -> return ()

forwardOnly :: forall tag dty m. (DType dty, MonadIO m) => M.HashMap Text (NDArray dty) -> Module tag dty m [NDArray dty]
forwardOnly inputs = do
    adapt inputs
    exec <- use $ untag . mod_executor
    liftIO $ do
        execForward exec False
        execGetOutputs exec

fit :: forall tag dty m. (DType dty, MonadIO m) => M.HashMap Text (NDArray dty) -> Module tag dty m ()
fit inputs = do
    adapt inputs
    exec <- use $ untag . mod_executor
    liftIO $ do
        execForward exec True
        execBackward exec []

update :: forall tag dty opt m any. (Optimizer opt, DType dty, MonadIO m) => opt dty -> M.HashMap Text any -> Module tag dty m ()
update opt blacklist = do
    params <- use (untag . mod_params)
    forM_ (M.toList params) $ \case
        (k, ParameterG weig grad) | not (M.member k blacklist) -> do
            optimize opt k weig grad
        _ -> return ()
    untag . mod_statistics . stat_num_upd += 1


fitAndEval :: forall tag dty opt mtr m. (DType dty, Optimizer opt, EvalMetricMethod mtr, MonadIO m)
           => opt dty -> M.HashMap Text (NDArray dty) -> MetricData mtr dty -> Module tag dty m ()
fitAndEval opt datAndLbl metric = do
    fit datAndLbl
    update opt M.empty
    exec <- use (untag . mod_executor)
    out  <- liftIO $ execGetOutputs exec
    eval_results <- evaluate metric datAndLbl out
    untag . mod_scores %= M.union eval_results


fitDataset :: (Dataset d, DatasetProp d e, DType a,
        MonadIO m, MonadThrow m, MonadReader env m, HasLogFunc env,
        HasCallStack,
        DatasetMonadConstraint d (Module t a m),
        Optimizer opt, EvalMetricMethod mtr)
    => d (Module t a m) e
    -> d (Module t a m) e
    -> ([Text] -> e -> M.HashMap Text (NDArray a))
    -> opt a
    -> mtr a
    -> Int
    -> Module t a m ()
fitDataset trainDataset valDataset make_binding opt metric epochs = do
    -- callbacks <- use sess_callbacks

    vars <- M.keys <$> use (untag . mod_input_shapes)

    total     <- sizeD trainDataset
    -- batchSize <- batchSizeD trainDataset >>= maybe (throwM DatasetOfUnknownBatchSize) return

    lift . logInfo $ display ("[Train]" :: Text)
    forM_ [1..epochs] $ \epochInd -> do
        trainMetricData <- newMetric "train" metric

        lift . logInfo . display $ sformat ("epoch " % int) epochInd
        -- forM_ callbacks (begOfEpoch epochInd total)

        void $ forEachD_i trainDataset $ \(i, item) -> do
            -- forM_ callbacks (begOfBatch i batchSize)
            let binding = make_binding vars item
            fitAndEval opt binding trainMetricData
            eval <- format trainMetricData
            lift . logInfo . display $ sformat (int % int % stext) i total eval
            -- forM_ callbacks (endOfBatch i batchSize)

        -- forM_ callbacks (endOfEpoch epochInd total)

        lift . logInfo $ display ("[Validate]" :: Text)
        valMetricData <- newMetric "val" metric
        void $ forEachD valDataset $ \item -> do
            let binding = make_binding vars item
            -- TODO: it is bad to pass labels to forwardOnly
            out <- forwardOnly binding
            evaluate valMetricData binding out
        eval <- format valMetricData
        lift . logInfo $ display eval
        --
        -- forM_ callbacks (endOfVal epochInd total)
