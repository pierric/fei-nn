module MXNet.NN.Module where

import qualified Data.HashMap.Strict as M
import qualified Data.HashSet as S
import Control.Arrow (second, Kleisli(..))
import Control.Lens.Setter ((.=), (+=))
import Control.Lens.Getter (use, (^.))
import Control.Monad (when, forM_)
import Control.Monad.Trans (lift)
import Control.Exception (assert)

import MXNet.Base (
    DType, Context,
    Symbol, listArguments, listAuxiliaryStates, inferShape,
    NDArray, ndshape, makeEmptyNDArray,
    Executor, execForward, execBackward, execGetOutputs, execReshapeEx, execBind,
    waitAll)
import MXNet.NN.Types
import MXNet.NN.Session
import MXNet.NN.TaggedState (Tagged(..), untag)
import MXNet.NN.Optimizer (Optimizer, optimize)

initialize :: forall tag dty. DType dty => Symbol dty -> Config dty -> IO (TaggedModuleState dty tag)
initialize symbol config = do
     -- give a initial batch_size = 1 for the placeholders
    let spec1 = M.map (1:) $ M.difference input_shapes initializers
        spec2 = initializers
        dinit = config ^. cfg_default_initializer
        cxt   = config ^. cfg_context

    (args, _, auxs, _) <- inferShape symbol (M.toList spec1)
    let arg_with_shp = M.fromList args
        aux_with_shp = M.fromList auxs
    ---------------------
    -- important! labels should be merged into placeholders,
    -- otherwise the labels are considered to have gradient.
    ---------------------
    let lbl_with_shp = M.filterWithKey (\k v -> k `elem` label_names) arg_with_shp
    placeholders <- mapM (flip makeEmptyNDArray cxt) $ M.union spec1 lbl_with_shp

    arg_tensors <- M.traverseWithKey (initI placeholders spec2 dinit) arg_with_shp
    aux_tensors <- M.traverseWithKey (initA dinit) aux_with_shp

    let params = arg_tensors `M.union` aux_tensors
    executor <- bind symbol cxt params True

    return $ Tagged $ ModuleState {
        _mod_symbol       = symbol,
        _mod_input_shapes = M.union input_shapes lbl_with_shp,
        _mod_params       = params,
        _mod_context      = cxt,
        _mod_executor     = executor,
        _mod_statistics    = Statistics 0 0
    }
  where
    input_shapes = config ^. cfg_data
    label_names  = config ^. cfg_label
    initializers = config ^. cfg_initializers
    -- initialize input symbols.
    -- placeholders are backed by empty NDArray,
    -- other input symbols are initialized by an initializer.
    initI placeholder spec2 dinit inp shp =
        case M.lookup inp placeholder of
            Just in_arg -> do
                return $ ParameterI in_arg Nothing
            Nothing -> do
                arg_in <- case M.lookup inp spec2 of
                    Just cinit -> cinit inp shp (_cfg_context config)
                    Nothing    -> dinit inp shp (_cfg_context config)
                arg_gr <- makeEmptyNDArray shp (_cfg_context config)
                return $ ParameterI arg_in (Just arg_gr)
    -- initialize auxiliary symbols.
    initA dinit aux shp = do
        arg_aux <- dinit aux shp (_cfg_context config)
        return $ ParameterA arg_aux

bind :: DType dty => Symbol dty -> Context -> M.HashMap String (Parameter dty) -> Bool -> IO (Executor dty)
bind symbol context params trainable = do
    argnames <- listArguments symbol
    auxnames <- listAuxiliaryStates symbol 
    -- sanity check
    assert (M.keysSet params == S.fromList (argnames ++ auxnames)) (return ())
    -- the parameters to bind should be arranged in the same order as the argnames
    let num_args = length argnames
        arg_in   = map (_param_in . (params M.!)) argnames
        arg_gr_w_req = if trainable then flip map argnames $ \name -> do
                                            a <- _param_grad (params M.! name)
                                            return (a, 1)
                                    else replicate num_args Nothing
        aux_arg_aux  = map (_param_aux . (params M.!)) auxnames
    execBind symbol context arg_in arg_gr_w_req aux_arg_aux

adapt :: DType dty => M.HashMap String (NDArray dty) -> Module tag dty ()
adapt inputs = do
    symbol  <- use $ untag . mod_symbol
    exec    <- use $ untag . mod_executor
    context <- use $ untag . mod_context
    -- shapes  <- M.toList <$> use mod_input_shapes
    -- shapes' <- lift $ mapM (runKleisli $ second $ Kleisli ndshape) $ M.toList inputs
    shapes  <- use $ untag . mod_input_shapes
    shapes' <- lift $ mapM ndshape inputs

    when (shapes /= shapes') $ do
        (args, grads, auxs, exec) <- lift $ execReshapeEx exec True True context (M.toList shapes')
        arg_names <- lift $ listArguments symbol
        aux_names <- lift $ listAuxiliaryStates symbol
        let arg_ndarrs = M.fromList $ zip arg_names $ map (uncurry ParameterI) (zip args grads)
            aux_ndarrs = M.fromList $ zip aux_names $ map ParameterA auxs
        untag . mod_executor .= exec
        untag . mod_input_shapes   .= shapes'
        untag . mod_params   .= M.union arg_ndarrs aux_ndarrs

forwardOnly :: forall tag dty. DType dty => M.HashMap String (NDArray dty) -> Module tag dty [NDArray dty]
forwardOnly inputs = do
    adapt inputs
    exec <- use $ untag . mod_executor
    lift $ do
        execForward exec False
        execGetOutputs exec

fit :: forall tag dty. DType dty => M.HashMap String (NDArray dty) -> Module tag dty ()
fit inputs = do
    adapt inputs
    exec <- use $ untag . mod_executor
    lift $ do
        execForward exec True
        execBackward exec []

update :: forall tag dty opt any. (Optimizer opt, DType dty) => opt dty -> M.HashMap String any -> Module tag dty ()
update opt blacklist = do
    params <- use (untag . mod_params)
    forM_ (M.toList params) $ \case
        (k, ParameterI weig (Just grad)) | not (M.member k blacklist) -> do
            optimize opt k weig grad
        _ -> return ()
    untag . mod_statistics . stat_num_upd += 1
    lift waitAll


