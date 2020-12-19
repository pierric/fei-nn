module MXNet.NN.Module where

import           Control.Lens            (has, ix, use, (%=), (+=), (.=), (^?!))
import           Formatting              (int, sformat, stext, (%))
import           GHC.TypeLits            (KnownSymbol)
import           RIO
import qualified RIO.HashMap             as M
import qualified RIO.HashSet             as S
import           RIO.List                (zipWith3)
import qualified RIO.NonEmpty            as NE
import qualified RIO.Text                as T
import qualified RIO.Vector.Storable     as VS

import           MXNet.Base
import           MXNet.NN.DataIter.Class (Dataset (..), DatasetProp (..))
import           MXNet.NN.EvalMetric     (EvalMetricMethod (..), MetricData)
import           MXNet.NN.Layer          (copy)
import           MXNet.NN.Optimizer      (Optimizer, optimize)
import           MXNet.NN.Session        (withSession)
import           MXNet.NN.TaggedState    (Tagged (..), untag)
import           MXNet.NN.Types


data InitError = UnknownShape Text
    | ShouldNotBeScalar Text
    | ShapeNotAgree Text (NonEmpty Int) (NonEmpty Int)
    | BadShapeValue Text Text
    | BadInitValue Text Text
    deriving (Typeable, Show)
instance Exception InitError


initialize :: forall tag dty. (HasCallStack, FloatDType dty) => SymbolHandle -> Config dty -> IO (TaggedModuleState dty tag)
initialize symbol config = do
     -- give a initial batch_size = 1 for the placeholders
    let spec1 = M.difference input_shapes initializers
        spec2 = initializers
        dinit = config ^. cfg_default_initializer
        cxt   = config ^. cfg_context
        -- remove any input or label from the fixed set
        fixed = (config ^. cfg_fixed_params) `S.difference`
                (S.fromList $ M.keys input_shapes ++ label_names)

    -- although it is possible to set __shape__ to certain symbols, we
    -- don't use the information for shape inference right now.

    (args, _, auxs, _) <- inferShape symbol (M.toList spec1)
    arg_with_shp <- M.fromList <$> mapM checkTensorShape args
    aux_with_shp <- M.fromList <$> mapM checkTensorShape auxs
    let all_shapes = arg_with_shp `M.union` aux_with_shp

    attrs <- mxSymbolListAttr symbol
    let node_with_init = M.filter (has $ ix "__init__") attrs
    ---------------------
    -- important! labels should be merged into placeholders,
    -- otherwise the labels are considered to have gradient.
    ---------------------
    let lbl_with_shp = M.filterWithKey (\k _ -> k `elem` label_names) arg_with_shp
        phl_with_shp = M.map _shape_nonempty spec1 `M.union` lbl_with_shp
    placeholders <- mapM (flip makeEmptyNDArray cxt) phl_with_shp

    arg_init    <- M.traverseWithKey (initN arg_with_shp aux_with_shp fixed) node_with_init
    arg_tensors <- M.traverseWithKey (initI placeholders fixed spec2 dinit) arg_with_shp
    aux_tensors <- M.traverseWithKey (initA dinit) aux_with_shp

    let params = arg_init `M.union` arg_tensors `M.union` aux_tensors
    --forM (M.toList params) $ \(k, p) -> do
    --    o <- case p of
    --      ParameterV a -> fmap (\s -> ("V", [s])) $ ndshape a
    --      ParameterF a -> fmap (\s -> ("F", [s])) $ ndshape a
    --      ParameterG a b -> liftM2 (\s t -> ("V", [s, t])) (ndshape a) (ndshape b)
    --      ParameterA a -> fmap (\s -> ("A", [s])) $ ndshape a
    --    traceShowM ("  param", k, o)

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
    context      = config ^. cfg_context
    -- initialize input symbols.
    -- placeholders are backed by empty NDArray,
    -- other input symbols are initialized by an initializer.
    initI placeholders fixed spec2 dinit inp shp = do
        case M.lookup inp placeholders of
            Just in_arg -> do
                return $ ParameterV in_arg
            Nothing -> do
                arg_in <- case M.lookup inp spec2 of
                    Just cinit -> cinit inp shp context
                    Nothing    -> dinit inp shp context
                if S.member inp fixed
                    then return $ ParameterF arg_in
                    else do
                        arg_gr <- makeEmptyNDArray shp context
                        return $ ParameterG arg_in arg_gr
    -- initialize auxiliary symbols.
    initA dinit aux shp = do
        arg_aux <- dinit aux shp context
        return $ ParameterA arg_aux

    -- initialize symbol with __init__ attribute
    initN args auxs fixed name attrs = do
        let shp1 = M.lookup name (M.union args auxs)
            shp2 = M.lookup "__shape__" attrs

        shp2 <- case shp2 of
                  Nothing -> pure Nothing
                  Just s -> case readMaybe (T.unpack s) >>= NE.nonEmpty of
                      Nothing -> throwM $ BadShapeValue name s
                      t       -> pure t

        shape <- case (shp1, shp2) of
          (Nothing, Nothing) -> throwM $ UnknownShape name
          (Just shp1, Just shp2) | shp1 /= shp2 -> throwM $ ShapeNotAgree name shp1 shp2
          (Just shp1, _) -> pure shp1
          (_, Just shp2) -> pure shp2

        let value_text = attrs ^. ix "__init__"
        array <- case T.unpack value_text of
                   "[\"zero\", {}]" -> zeros shape >>= flip toContext context
                   "[\"one\", {}]"  -> ones  shape >>= flip toContext context
                   v -> case readMaybe v of
                          Nothing -> throwM $ BadInitValue name value_text
                          Just v  -> makeNDArray shape context $ VS.fromList v

        case (S.member name fixed, M.member name args, M.member name auxs) of
          (True, _, _) -> return $ ParameterF array
          (False, False, True) -> return $ ParameterA array
          (False, True, False) -> do
              grad <- makeEmptyNDArray shape context
              return $ ParameterG array grad

    checkTensorShape (name, SScalar)   = throwIO $ ShouldNotBeScalar name
    checkTensorShape (name, STensor s) = return (name, s)


bind :: (HasCallStack, FloatDType dty) => SymbolHandle -> Context -> M.HashMap Text (Parameter dty) -> Bool -> IO (Executor dty)
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


adapt :: (FloatDType dty, MonadIO m) => M.HashMap Text (NDArray dty) -> Module tag dty m ()
adapt inputs = do
    symbol  <- use $ untag . mod_symbol
    exec    <- use $ untag . mod_executor
    context <- use $ untag . mod_context
    fixed   <- use $ untag . mod_fixed_args
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

        -- it should be safe to free the old executor
        old_executor <- use (untag . mod_executor)
        liftIO $ execFree old_executor

        untag . mod_executor .= exec
        untag . mod_input_shapes .= shapes'
        untag . mod_params   .= M.union arg_ndarrs aux_ndarrs

    -- copy the ndarray
    targets <- use $ untag . mod_params
    forM_ (M.toList inputs) $ \ (k, src) -> liftIO $ do
        case M.lookup k targets of
          Just (ParameterV dst) -> void $ copy src dst
          _                     -> return ()

forwardOnly :: (FloatDType dty, MonadIO m) => M.HashMap Text (NDArray dty) -> Module tag dty m [NDArray dty]
forwardOnly inputs = do
    adapt inputs
    exec <- use $ untag . mod_executor
    liftIO $ do
        execForward exec False
        execGetOutputs exec

fit :: (FloatDType dty, MonadIO m) => M.HashMap Text (NDArray dty) -> Module tag dty m ()
fit inputs = do
    adapt inputs
    exec <- use $ untag . mod_executor
    liftIO $ do
        execForward exec True
        execBackward exec []

update :: (Optimizer opt, FloatDType dty, MonadIO m) => opt dty -> M.HashMap Text any -> Module tag dty m ()
update opt blacklist = do
    params <- use $ untag . mod_params
    forM_ (M.toList params) $ \case
        (k, ParameterG weig grad) | not (M.member k blacklist) -> do
            optimize opt k weig grad
        _ -> return ()
    untag . mod_statistics . stat_num_upd += 1


fitAndEval :: (FloatDType dty, Optimizer opt, EvalMetricMethod mtr, MonadIO m)
    => opt dty -> M.HashMap Text (NDArray dty) -> MetricData mtr dty -> Module tag dty m ()
fitAndEval opt datAndLbl metric = do
    fit datAndLbl
    update opt M.empty
    exec <- use $ untag . mod_executor
    out  <- liftIO $ execGetOutputs exec
    eval_results <- metricUpdate metric datAndLbl out
    untag . mod_scores %= M.union eval_results


fitDataset :: (KnownSymbol tag, Dataset d, DatasetProp d e, FloatDType a,
        MonadIO m, MonadThrow m, MonadReader env m, HasLogFunc env,
        HasCallStack,
        DatasetMonadConstraint d m,
        Optimizer opt, EvalMetricMethod mtr)
    => TaggedModuleState a tag
    -> d m e
    -> d m e
    -> ([Text] -> e -> M.HashMap Text (NDArray a))
    -> opt a
    -> mtr a
    -> Int
    -> m ()
fitDataset sess trainDataset valDataset make_binding opt metric epochs = do
    -- callbacks <- use sess_callbacks

    sess_mvar <- newMVar sess
    let variables = M.keys $ sess ^. untag . mod_input_shapes
    total <- sizeD trainDataset
    -- batchSize <- batchSizeD trainDataset >>= maybe (throwM DatasetOfUnknownBatchSize) return

    logInfo $ display ("[Train]" :: Text)
    forM_ [1..epochs] $ \epochInd -> do
        trainMetricData <- newMetric "train" metric

        logInfo . display $ sformat ("epoch " % int) epochInd
        -- forM_ callbacks (begOfEpoch epochInd total)

        void $ forEachD_i trainDataset $ \(i, item) -> withSession sess_mvar  $ do
            -- forM_ callbacks (begOfBatch i batchSize)
            let binding = make_binding variables  item
            fitAndEval opt binding trainMetricData
            eval <- metricFormat trainMetricData
            logInfo . display $ sformat (int % int % stext) i total eval
            -- forM_ callbacks (endOfBatch i batchSize)

        -- forM_ callbacks (endOfEpoch epochInd total)

        logInfo $ display ("[Validate]" :: Text)
        valMetricData <- newMetric "val" metric
        void $ forEachD valDataset $ \item -> withSession sess_mvar  $ do
            let binding = make_binding variables  item
            -- TODO: it is bad to pass labels to forwardOnly
            out <- forwardOnly binding
            metricUpdate valMetricData binding out
        eval <- metricFormat valMetricData
        logInfo $ display eval
        --
        -- forM_ callbacks (endOfVal epochInd total)
