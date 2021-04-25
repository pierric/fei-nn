module MXNet.NN.Module where

import           Control.Lens            (has, ix, use, (%=), (+=), (.=), (^?!))
import           Formatting              (int, sformat, stext, (%))
import           GHC.TypeLits            (KnownSymbol)
import           RIO
import qualified RIO.HashMap             as M
import qualified RIO.HashSet             as S
import           RIO.List                (zip3, zipWith3)
import qualified RIO.NonEmpty            as NE
import qualified RIO.Text                as T
import qualified RIO.Vector.Storable     as VS

import           MXNet.Base
import           MXNet.NN.DataIter.Class (Dataset (..), DatasetProp (..))
import           MXNet.NN.EvalMetric     (EvalMetricMethod (..), MetricData)
import qualified MXNet.NN.Initializer    as I
import           MXNet.NN.Optimizer      (Optimizer, optimize)
import           MXNet.NN.Session        (withSession)
import           MXNet.NN.TaggedState    (Tagged (..), untag)
import           MXNet.NN.Types


data InitError = UnknownShape Text
    | ShouldNotBeScalar Text
    | ShapeNotAgree Text [Int] [Int]
    | BadShapeValue Text Text
    | BadInitValue Text Text
    deriving (Typeable, Show)
instance Exception InitError


initialize :: forall tag dty. (HasCallStack, FloatDType dty) => Symbol dty -> Config dty -> IO (TaggedModuleState dty tag)
initialize symbol config = do
    attrs    <- listAttrs symbol
    argnames <- listArguments symbol

    let dinit = config ^. cfg_default_initializer
        cxt   = config ^. cfg_context
        node_with_init = M.mapMaybe (M.lookup "__init__") attrs
        no_grad = (config ^. cfg_fixed_params) `S.union`
                  (S.fromList $ M.keys input_shapes ++ label_names)
        fixed   = no_grad `S.difference` (S.fromList $ M.keys input_shapes ++ label_names)
        rtypes = M.mapMaybe (M.lookup "__grad_req__" >=> parseRType) attrs `M.union`
                 M.map (const ReqNull) (S.toMap no_grad) `M.union`
                 M.fromList [(n, ReqWrite) | n <- argnames]
        shapes = M.mapMaybe (M.lookup "__shape__" >=> parseShape) attrs `M.union` input_shapes
        dtypes = M.mapMaybe (M.lookup "__dtype__" >=> parseDType) attrs
        stypes = M.mapMaybe (M.lookup "__storage_type__" >=> parseSType) attrs

    (params, executor) <- simpleBind symbol cxt rtypes shapes dtypes stypes

    M.traverseWithKey (initN params) node_with_init
    M.traverseWithKey (initD dinit initializers) params

    return $ Tagged $ ModuleState {
        _mod_symbol       = symbol,
        _mod_input_shapes = input_shapes,
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

    initD dinit cinit name param = do
        let init_op = case M.lookup name cinit of
                        Just x  -> x
                        Nothing -> dinit

        case param of
          ParameterG a g -> init_op name a >> I.zeros name g
          ParameterA a   -> dinit name a
          _              -> return ()

    -- initialize symbol with __init__ attribute
    initN :: DType a => HashMap Text (Parameter a) -> Text -> Text -> IO ()
    initN params name init_value = do
        let do_init a = case T.unpack init_value of
                          "[\"zero\", {}]" -> I.zeros name a
                          "[\"one\", {}]"  -> I.ones name a
                          v -> case readMaybe v :: Maybe Float of
                                 Just v  -> I.constant v name a
                                 _ -> case readMaybe v of
                                        Just v  -> I.vector (VS.fromList v) name a
                                        Nothing -> throwM $ BadInitValue name init_value
         in case M.lookup name params of
              Nothing -> let msg = sformat ("'" % stext % "' is not in param list, but has __init__ defined?") name
                          in error $ T.unpack msg
              Just (ParameterV a) -> do_init a
              Just (ParameterA a) -> do_init a
              _ ->  error $ T.unpack name ++ " has an __init__ property, " ++
                            "but it is neither a variable nor an auxiliary state."

    parseShape :: Text -> Maybe [Int]
    parseShape = readMaybe . T.unpack

    parseRType :: Text -> Maybe ReqType
    parseRType "null"    = Just ReqNull
    parseRType "write"   = Just ReqWrite
    parseRType "add"     = Just ReqAdd
    parseRType "inplace" = Just ReqInplace
    parseRType _         = Nothing

    parseDType :: Text -> Maybe Int
    parseDType "float32" = Just 0
    parseDType "float64" = Just 1
    parseDType "uint8"   = Just 3
    parseDType "int32"   = Just 4
    parseDType "int64"   = Just 6
    parseDType _         = Nothing

    parseSType :: Text -> Maybe Int
    parseSType "default" = Just 0
    parseSType _         = Nothing

bind :: (HasCallStack, FloatDType dty)
     => Symbol dty -> Context -> M.HashMap Text (Parameter dty) -> Bool -> IO (Executor dty)
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
                    ParameterG t _ -> t;
                    ParameterA _ -> error "auxiliary parameter shouldn't occur"
                    }) arg_all
        arg_gr_w_req = if trainable then map (\case {ParameterG _ t -> Just (t, ReqWrite); _ -> Nothing}) arg_all
                                    else replicate num_args Nothing
        aux_arg_aux  = map (_param_aux . (params ^?!) . ix) auxnames
    execBind symbol context arg_in arg_gr_w_req aux_arg_aux

simpleBind :: (HasCallStack, FloatDType dty)
           => Symbol dty
           -> Context
           -> HashMap Text ReqType
           -> HashMap Text Shape
           -> HashMap Text Int
           -> HashMap Text Int
           -> IO (M.HashMap Text (Parameter dty), Executor dty)
simpleBind symbol context rtypes shapes dtypes stypes = do
    argnames <- listArguments symbol
    auxnames <- listAuxiliaryStates symbol
    (args, grads, auxs, exec) <- execSimpleBind symbol context rtypes shapes dtypes stypes

    let build1 n a Nothing  = traceShow ("V", n) (n, ParameterV a)
        build1 n a (Just g) = traceShow ("G", n) (n, ParameterG a g)
        param1 = zipWith3 build1 argnames args grads
        param2 = zipWith (\n a -> (n, ParameterA a)) auxnames auxs
    return $ (M.fromList $ param1 ++ param2, exec)

adapt :: (HasCallStack, FloatDType dty, MonadIO m) => M.HashMap Text (NDArray dty) -> Module tag dty m ()
adapt inputs = do
    symbol  <- use $ untag . mod_symbol
    exec    <- use $ untag . mod_executor
    context <- use $ untag . mod_context
    fixed   <- use $ untag . mod_fixed_args
    shapes  <- use $ untag . mod_input_shapes
    shapes' <- liftIO $ mapM ndshape inputs

    -- reshape the executor (and arg, aux arrays)
    when (shapes /= shapes') $ do
        (args, grads, auxs, exec) <- liftIO $ do
            void $ mxSetIsNumpyShape NpyShapeOff
            ret <- execReshapeEx exec True True context (M.toList shapes')
            void $ mxSetIsNumpyShape NpyShapeGL
            return ret
        arg_names <- liftIO $ listArguments symbol
        aux_names <- liftIO $ listAuxiliaryStates symbol
        let buildArg key a Nothing  = (key, ParameterV a)
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

forwardOnly :: (HasCallStack, FloatDType dty, MonadIO m) => M.HashMap Text (NDArray dty) -> Module tag dty m [NDArray dty]
forwardOnly inputs = do
    adapt inputs
    exec <- use $ untag . mod_executor
    liftIO $ do
        execForward exec False
        execGetOutputs exec

fit :: (HasCallStack, FloatDType dty, MonadIO m) => M.HashMap Text (NDArray dty) -> Module tag dty m ()
fit inputs = do
    adapt inputs
    exec <- use $ untag . mod_executor
    liftIO $ do
        execForward exec True
        execBackward exec []

update :: (HasCallStack, Optimizer opt, FloatDType dty, MonadIO m) => opt dty -> M.HashMap Text any -> Module tag dty m ()
update opt blacklist = do
    params <- use $ untag . mod_params
    forM_ (M.toList params) $ \case
        (k, ParameterG weig grad) | not (M.member k blacklist) -> do
            optimize opt k weig grad
        _ -> return ()
    untag . mod_statistics . stat_num_upd += 1


fitAndEval :: (HasCallStack, FloatDType dty, Optimizer opt, EvalMetricMethod mtr, MonadIO m)
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
