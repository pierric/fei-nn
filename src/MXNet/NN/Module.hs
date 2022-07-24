{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.NN.Module where

import           Control.Lens                 (ix, use, (%=), (+=), (.=), (^?!))
import           Formatting                   (int, sformat, stext, (%))
import           GHC.TypeLits                 (KnownSymbol)
import           RIO
import qualified RIO.HashMap                  as M
import qualified RIO.HashSet                  as S
import           RIO.List                     (zipWith3)
import qualified RIO.Text                     as T

import           MXNet.Base
import           MXNet.Base.Tensor.Functional (copy)
import           MXNet.NN.DataIter.Class      (Dataset (..), DatasetProp (..))
import           MXNet.NN.EvalMetric          (EvalMetricMethod (..),
                                               MetricData)
import           MXNet.NN.Initializer
import           MXNet.NN.Optimizer           (Optimizer, optimize)
import           MXNet.NN.Session             (withSession)
import           MXNet.NN.TaggedState         (Tagged (..), untag)
import           MXNet.NN.Types


data InitError = UnknownShape Text
    | ShouldNotBeScalar Text
    | ShapeNotAgree Text [Int] [Int]
    | BadShapeValue Text Text
    | BadInitValue Text Text
    deriving (Typeable, Show)
instance Exception InitError


-- | initialiez the module
--
-- Due to restriction of the random operations of mxnet, we have to restrict the
-- dtype to BasicFloatDTypes.
--
initialize :: forall tag dty. (HasCallStack, FloatDType dty, InEnum (DTypeName dty) BasicFloatDTypes)
           => Symbol dty -> Config dty -> IO (TaggedModuleState dty tag)
initialize symbol config = do
    attrs    <- listAttrs symbol
    argnames <- listArguments symbol

    let dinit = config ^. cfg_default_initializer
        cxt   = config ^. cfg_context
        node_with_init = M.mapMaybe (M.lookup "__init__") attrs
        no_grad = (config ^. cfg_fixed_params) `S.union`
                  S.fromList (M.keys input_shapes ++ label_names)
        fixed   = no_grad `S.difference` S.fromList (M.keys input_shapes ++ label_names)
        rtypes = M.mapMaybe (M.lookup "__grad_req__" >=> parseRType) attrs `M.union`
                 M.map (const ReqNull) (S.toMap no_grad) `M.union`
                 M.fromList [(n, ReqWrite) | n <- argnames]
        shapes = M.mapMaybe (M.lookup "__shape__" >=> parseShape) attrs `M.union` input_shapes
        dtypes = M.mapMaybe (M.lookup "__dtype__" >=> parseDType) attrs
        stypes = M.mapMaybe (M.lookup "__storage_type__" >=> parseSType) attrs

    (params, executor) <- simpleBind symbol cxt rtypes shapes dtypes stypes [] Nothing

    let inits1 = concat $ M.mapWithKey (initN params) node_with_init
        inits2 = concat $ M.mapWithKey (initD dinit initializers) params
    forM_ (inits1 ++ inits2) (\(SomeInitializer n, t, a) -> initNDArray n t a)

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
    -- initD :: DType a
    --       => SomeInitializer a
    --       -> HashMap Text (SomeInitializer a)
    --       -> Text
    --       -> Parameter a
    --       -> [(SomeInitializer a, Text, NDArray a)]
    initD dinit _ name (ParameterA a)       = [(dinit, name, a)]
    initD dinit cinit name (ParameterG a g) = let uinit = case M.lookup name cinit of
                                                            Just uinit -> uinit
                                                            Nothing    -> dinit
                                               in [(uinit, name, a), (SomeInitializer InitZeros, name, g)]
    initD _ _ _ _ = []

    -- initialize symbol with __init__ attribute
    -- initN :: forall a . (FloatDType a, InEnum (DTypeName a) BasicFloatDTypes)
    --       => HashMap Text (Parameter a) -> Text -> Text -> [(SomeInitializer a, Text, NDArray a)]
    initN params name init_value =
        -- NOTE: it seems not possible to create an arbitrary initializer from a str, even knowing that
        -- it derives the Read class. We fall back to the predefined two options, SimpleInit or RandomInit.
        let initS = SomeInitializer <$> (readMaybe $ T.unpack init_value :: Maybe (SimpleInit dty))
            initR = SomeInitializer <$> (readMaybe $ T.unpack init_value :: Maybe (RandomInit dty))
            param = M.lookup name params
         in case (initS <|> initR, param) of
              (_, Nothing) -> let msg = sformat ("'" % stext % "' is not in param list, but has __init__ defined?") name
                               in error $ T.unpack msg
              (Nothing, _) -> let msg = sformat ("'" % stext % "' has an __init__ property, but it is neither a variable nor an auxiliary state.") name
                               in error $ T.unpack msg
              (Just init, Just (ParameterV a)) -> [(init, name, a)]
              (Just init, Just (ParameterA a)) -> [(init, name, a)]
              (_, _) -> error "Shouldn't happen"

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
           -> [Text]
           -> Maybe (Executor dty)
           -> IO (M.HashMap Text (Parameter dty), Executor dty)
simpleBind symbol context rtypes shapes dtypes stypes shared_arg_names shared_exec = do
    argnames <- listArguments symbol
    auxnames <- listAuxiliaryStates symbol
    (args, grads, auxs, exec) <- execSimpleBindWithShared
                                    symbol context rtypes shapes dtypes stypes
                                    shared_arg_names shared_exec

    let build1 n a Nothing  = (n, ParameterV a)
        build1 n a (Just g) = (n, ParameterG a g)
        param1 = zipWith3 build1 argnames args grads
        param2 = zipWith (\n a -> (n, ParameterA a)) auxnames auxs
    return $ (M.fromList $ param1 ++ param2, exec)

adapt :: (HasCallStack, FloatDType dty, MonadIO m) => M.HashMap Text (NDArray dty) -> Module tag dty m ()
adapt inputs = do
    symbol  <- use $ untag . mod_symbol
    exec    <- use $ untag . mod_executor
    context <- use $ untag . mod_context
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

withSharedParameters :: (HasCallStack, FloatDType dty, MonadIO m, MonadThrow m)
                    => Symbol dty
                    -> HashMap Text [Int]
                    -> ((HashMap Text (NDArray dty) -> IO [NDArray dty]) -> IO a)
                    -> Module tag dty m a
withSharedParameters symbol input_shapes proc = do
    attrs <- liftIO $ listAttrs symbol

    params <- use (untag . mod_params) <&> M.filter hasGradient
    cxt    <- use (untag . mod_context)
    exec   <- use $ untag . mod_executor

    let shared_arg_names = S.toList $ S.intersection (mapToSet params) (mapToSet attrs)
        shapes = M.mapMaybe (M.lookup "__shape__" >=> parseShape) attrs `M.union` input_shapes
        rtypes = M.map (const ReqNull) attrs
        dtypes = M.mapMaybe (M.lookup "__dtype__" >=> parseDType) attrs
        stypes = M.mapMaybe (M.lookup "__storage_type__" >=> parseSType) attrs

    liftIO $ do
        (params', executor) <- simpleBind symbol cxt rtypes shapes dtypes stypes
                                          shared_arg_names (Just exec)

        let forward inputs = do
                forM_ (M.toList inputs) $ \ (k, src) -> do
                    case M.lookup k params' of
                      Just (ParameterV dst) -> void $ copy src dst
                      _                     -> return ()

                execForward executor False
                execGetOutputs executor
        proc forward

    where
        mapToSet = S.fromMap . M.map (const ())

-- forwardOnlyShared :: (HasCallStack, FloatDType dty, MonadIO m, MonadThrow m)
--                   => Symbol dty
--                   -> M.HashMap Text (NDArray dty)
--                   -> Module tag dty m [NDArray dty]
-- forwardOnlyShared symbol inputs = do
--     attrs <- liftIO $ listAttrs symbol
--     input_shapes <- liftIO $ mapM ndshape inputs
--
--     params <- use (untag . mod_params)
--     cxt    <- use (untag . mod_context)
--     exec   <- use $ untag . mod_executor
--
--     let rtypes = M.map (const ReqNull) attrs
--         -- rtypes = M.mapMaybe (M.lookup "__grad_req__" >=> parseRType) attrs
--         shapes = M.mapMaybe (M.lookup "__shape__" >=> parseShape) attrs `M.union` input_shapes
--         dtypes = M.mapMaybe (M.lookup "__dtype__" >=> parseDType) attrs
--         stypes = M.mapMaybe (M.lookup "__storage_type__" >=> parseSType) attrs
--         mapToSet = S.fromMap . M.map (const ())
--         shared_arg_names = S.toList $ mapToSet params `S.difference` mapToSet inputs
--
--     liftIO $ do
--         (params', executor) <- simpleBind symbol cxt rtypes shapes dtypes stypes
--                                          shared_arg_names (Just exec)
--         forM_ (M.toList inputs) $ \ (k, src) -> do
--             case M.lookup k params' of
--               Just (ParameterV dst) -> void $ copy src dst
--               _                     -> return ()
--
--         execForward executor False
--         execGetOutputs executor

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
        --

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
