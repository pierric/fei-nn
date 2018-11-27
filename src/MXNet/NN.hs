{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
module MXNet.NN (
    Parameter(..),
    Config(..),
    Session(..),
    Exc(..),
    Initializer,
    TrainM,
    train,
    inferShape,
    initialize,
    fit, fitAndEval,
    forwardOnly,
    getContext,
    sess_param,
    sess_context,
    module MXNet.NN.Optimizer,
    module MXNet.NN.LrScheduler,
    module MXNet.NN.EvalMetric,
    module MXNet.NN.Initializer,
    module MXNet.NN.Layer,
) where

import MXNet.Base
import qualified MXNet.Base.Operators.NDArray as A

import qualified Data.HashMap.Strict as M
import qualified Control.Monad.State.Strict as ST
import Data.Maybe (isJust, fromJust, maybe)
import Control.Monad (when)
import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Monad.Trans.Resource (MonadThrow(..))
import Control.Lens (traverseOf, use, (+=))

import MXNet.NN.Types
import MXNet.NN.Optimizer
import MXNet.NN.EvalMetric
import MXNet.NN.Layer
import MXNet.NN.Initializer
import MXNet.NN.LrScheduler

-- | Execute the 'TrainM' monad
train :: (DType a, Monad m) => Session a -> TrainM a m r -> m r
train sess proc = ST.evalStateT (ST.evalStateT proc sess) (Statistics 0 0)

-- | infer the shapes of input and auxiliary symbols in a symbolic neural network
inferShape :: DType a => Symbol a -> M.HashMap String (NDArray a) -> IO (M.HashMap String [Int], M.HashMap String [Int])
inferShape (Symbol sym) known = do
    let (names, vals) = unzip $ M.toList known
    shapes <- mapM ndshape vals
    let arg_ind = scanl (+) 0 $ map length shapes
        arg_shp = concat shapes
    (inp_shp, _, aux_shp, complete) <- mxSymbolInferShape sym names arg_ind arg_shp
    when (not complete)  $ throwM InferredShapeInComplete
    inps <- mxSymbolListArguments sym
    auxs <- mxSymbolListAuxiliaryStates sym
    return (M.fromList $ zip inps inp_shp, M.fromList $ zip auxs aux_shp)

-- | initialize all parameters
initialize :: DType a => Symbol a -> Config a -> IO (Session a)
initialize sym config = do
    let spec1 = M.difference (_cfg_placeholders config) (_cfg_initializers config)
        spec2 = _cfg_initializers config
        dinit = _cfg_default_initializer config
        cxt   = _cfg_context config
    placeholder  <- mapM (\shp -> makeEmptyNDArray shp cxt) spec1
    (inp_with_shp, aux_with_shp) <- inferShape sym placeholder
    inp_args <- M.traverseWithKey (initI placeholder spec2 dinit) inp_with_shp
    aux_args <- M.traverseWithKey (initA dinit) aux_with_shp
    return $ Session (inp_args `M.union` aux_args) cxt
  where
    -- initialize input symbols.
    -- placeholders are backed by empty NDArray,
    -- other input symbols are initialized by an initializer.
    initI placeholder spec2 dinit inp shp = do
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

-- | bind the symbolic network with actual parameters
bind :: (DType a, MonadIO m, MonadThrow m) => Symbol a -> M.HashMap String (Maybe (NDArray a)) -> Bool -> TrainM a m (Executor a)
bind net dat train_ = do
    Context{..} <- use sess_context

    (inp_shps, aux_shps) <- liftIO $ inferShape net (M.map fromJust $ M.filter isJust dat)
    modifyT . traverseOf sess_param $ M.traverseWithKey $ \k p -> do
        case p of 
          ParameterI {} -> do 
            let ishp = inp_shps M.! k
            case M.lookup k dat of
                -- if the name is given in the binding data, we check its consistency.
                Just a  -> liftIO $ ensure_consistency (maybe (Right ishp) Left a) p
                -- if the name is missing in the binding data, we check the infered shape
                -- matches both the _param_in and _param_grad
                Nothing -> do
                    pshp1 <- liftIO $ ndshape (_param_in p)
                    when (ishp /= pshp1 ) (throwM $ MismatchedShapeOfSym (k ++ "[i]") ishp pshp1)
                    case (train_, _param_grad p) of
                        (True, Just ndarray) -> do
                            pshp2 <- liftIO $ ndshape ndarray
                            when (ishp /= pshp2) (throwM $ MismatchedShapeOfSym (k ++ "[t]") ishp pshp2)
                        _ -> return ()
                    return p
          ParameterA {} -> do
            let ishp = aux_shps M.! k
            pshp1 <- liftIO $ ndshape (_param_aux p)
            when (ishp /= pshp1 ) (throwM $ MismatchedShapeOfSym (k ++ "[i]") ishp pshp1)
            return p

    args <- use sess_param
    exec_handle <- liftIO $ do
        names <- mxSymbolListArguments (unSymbol net)
        -- the parameters to bind should be arranged in the same order as the names
        let arg_in  = map (unNDArray . _param_in) $ map (args M.!) names
            arg_gr  = if train_ 
                        then map (fmap unNDArray . _param_grad) $ map (args M.!) names
                        else replicate (M.size args) Nothing
            arg_gr_req = replicate (M.size args) 1

        auxnames <- mxSymbolListAuxiliaryStates (unSymbol net)
        let aux_arg_aux = map (unNDArray . _param_aux) $ map (args M.!) auxnames
        mxExecutorBind (unSymbol net) _device_type _device_id
                        arg_in arg_gr arg_gr_req 
                        aux_arg_aux
    return $ Executor exec_handle
  where
    -- make sure the _param_in can be used in the inference and backpropagation
    -- + user data input can be in a different context w.r.t. session configuration
    --   + copy inp with the right context
    -- + batch size can be different from the initial configuration, or at the time 
    --   to swap training and inference
    --   + create one and copy it
    -- + for inferenceOnly, labels' NDArray can be uninitialized. 
    --   + just create one
    ensure_consistency :: DType a => Either (NDArray a) [Int] -> Parameter a -> IO (Parameter a)
    ensure_consistency (Left a) p = do
        src_cxt <- context a
        src_shp <- ndshape a
        dst_cxt <- context (_param_in p)
        dst_shp <- ndshape (_param_in p)
        case (src_cxt == dst_cxt, src_shp == dst_shp) of
            (True , True) -> return $ p {_param_in = a}
            (False, True) -> do
                A._copyto_upd [unNDArray (_param_in p)] (#data := unNDArray a .& Nil)
                return p
            _ -> do
                a_copy <- makeEmptyNDArray src_shp dst_cxt
                A._copyto_upd [unNDArray a_copy] (#data := unNDArray a .& Nil)
                return $! p {_param_in = a_copy}    
    ensure_consistency (Right src_shp) p = do
        dst_cxt <- context (_param_in p)
        dst_shp <- ndshape (_param_in p)
        if src_shp == dst_shp 
            then return p
            else do
                dummy <- makeEmptyNDArray src_shp dst_cxt
                return $! p {_param_in = dummy}

-- | single step train. Must provide all the placeholders.
fit :: (DType a, MonadIO m, MonadThrow m, Optimizer opt) 
    => opt a -> Symbol a -> M.HashMap String (NDArray a) -> TrainM a m ()
fit opt net datAndLbl = do
    exec <- bind net (M.map Just datAndLbl) True
    liftIO $ do 
        mxExecutorForward (unExecutor exec) True
        mxExecutorBackward (unExecutor exec) []
        -- forward/backward are asynchronised operation in mxnet, in a
        -- sense that only opcodes are pushed onto an internal execution 
        -- stack, and there is a executor running in a separate thread.
        -- It is possible that an OOM of CPU memory occurs, if 'fit' are 
        -- called so fast that too many opcodes and data on the stack, 
        -- as described in issue #1
        mxNDArrayWaitAll
    updateParameters opt datAndLbl

-- | single step train. Must provide all the placeholders.
--   After fitting, it also update the evaluation metric.
fitAndEval :: (DType a, MonadIO m, MonadThrow m, Optimizer opt, EvalMetricMethod mtr)
           => opt a -> Symbol a -> M.HashMap String (NDArray a) -> mtr a -> TrainM a m ()
fitAndEval opt net datAndLbl metric = do
     Executor exec  <- bind net (M.map Just datAndLbl) True
     preds <- liftIO $ do 
         mxExecutorForward exec True
         mxExecutorBackward exec []
         mxNDArrayWaitAll
         map NDArray <$> mxExecutorOutputs exec
     updateParameters opt datAndLbl
     evaluate metric datAndLbl preds

updateParameters :: (MonadIO m, Optimizer opt, DType dtype) 
                 => opt dtype -> M.HashMap String any -> TrainM dtype m ()
updateParameters opt blacklist = do
    modifyT . traverseOf sess_param  $ M.traverseWithKey $ \ k v -> do
        case v of 
          ParameterI {} -> do 
            case (M.member k blacklist, _param_grad v) of
              (False, Just grad) -> do
                        new_in <- optimize opt k (_param_in v) grad
                        -- must evaluate the new parameter to WHNF
                        -- otherwise, the old _param_in is retained.
                        -- if context is GPU, then OOM will soon 
                        -- occur, as described in issue #2
                        return $! v {_param_in = new_in}
              _ -> return v
          ParameterA {} -> return v
    ST.lift (stat_num_upd += 1)

-- | forward only. Must provide all the placeholders, setting the data to @Just xx@, and set label to @Nothing@.
-- 
-- Note that the batch size here can be different from that in the training phase.
forwardOnly :: (DType a, MonadIO m, MonadThrow m) => Symbol a -> M.HashMap String (Maybe (NDArray a)) -> TrainM a m [NDArray a]
forwardOnly net dat = do
    Executor exec <- bind net dat False
    liftIO $ do
        mxExecutorForward exec False
        -- for the same reason in 'fit'.
        mxNDArrayWaitAll
        map NDArray <$> mxExecutorOutputs exec

getContext :: Monad m => TrainM a m Context
getContext = use sess_context

-- | modify the state within the inner monad
-- 
-- thanks to lens, we can modify the first field of the state with following 
-- combinator:
-- 
-- modifyT . traverseOf _1
--  :: (Field1 s s a b, Monad m) => (a -> m b) -> StateT s m ()
modifyT :: Monad m => (s -> m s) -> ST.StateT s m ()
modifyT func = do
    s0 <- ST.get
    s1 <- ST.lift $ func s0
    ST.put s1

