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
    CallbackClass(..), Callback(..),
    train,
    inferShape,
    initialize,
    fit, fit_, fitAndEval, fitDataset,
    forwardOnly,
    getContext,
    sess_param, sess_context, sess_callbacks,
    module MXNet.NN.Optimizer,
    module MXNet.NN.LrScheduler,
    module MXNet.NN.EvalMetric,
    module MXNet.NN.Initializer,
    module MXNet.NN.Layer,
    module MXNet.NN.Callback,
) where

import qualified Data.HashMap.Strict as M
import qualified Control.Monad.State.Strict as ST
import Data.Maybe (isJust, fromJust, maybe)
import Data.Foldable (forM_)
import Data.Time.Clock (getCurrentTime, diffUTCTime)
import Control.Monad (when, void)
import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Monad.Trans.Resource (MonadThrow(..))
import Control.Lens (traverseOf, use, (+=), (%=), (.=))
import System.IO (hFlush, stdout)
import System.Mem
import Text.Printf
-- import Control.Lens.Tuple

import MXNet.Base
import qualified MXNet.Base.Operators.NDArray as A

import MXNet.NN.Types
import MXNet.NN.Optimizer
import MXNet.NN.EvalMetric
import MXNet.NN.Layer
import MXNet.NN.Initializer
import MXNet.NN.LrScheduler
import MXNet.NN.DataIter.Class
import MXNet.NN.Callback

-- | Execute the 'TrainM' monad
train :: (DType a, Monad m) => Session a -> TrainM a m r -> m r
train sess proc = ST.evalStateT (ST.evalStateT proc sess) (Statistics 0 0)

-- inferShape' :: DType a => Symbol a -> M.HashMap String [Int] -> IO (M.HashMap String [Int], M.HashMap String [Int])
-- inferShape' (Symbol sym) known = do
--     let (names, shapes) = unzip $ M.toList known
--     let arg_ind = scanl (+) 0 $ map length shapes
--         arg_shp = concat shapes
--     (inp_shp, _, aux_shp, complete) <- mxSymbolInferShape sym names arg_ind arg_shp
--     when (not complete)  $ throwM InferredShapeInComplete
--     inps <- mxSymbolListArguments sym
--     auxs <- mxSymbolListAuxiliaryStates sym
--     return (M.fromList $ zip inps inp_shp, M.fromList $ zip auxs aux_shp)

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
    return $ Session {
        _sess_placeholders = _cfg_placeholders config,
        _sess_param = inp_args `M.union` aux_args,
        _sess_context = cxt,
        _sess_callbacks = []
    }
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

-- bind' :: (DType a, MonadIO m, MonadThrow m) => Symbol a -> M.HashMap String [Int] -> Bool -> TrainM a m (Executor a)
-- bind' net shapes train_ = do
--     Context{..} <- use sess_context
--     (inp_shps, aux_shps) <- liftIO $ inferShape' net shapes
--     modifyT . traverseOf sess_param $ M.traverseWithKey $ \k p -> do
--         case p of 
--             ParameterI {} -> do
--                 let ishp = inp_shps M.! k
--                 param_shp <- liftIO $ ndshape (_param_in p)
--                 param_in_new <- liftIO $ ensure_shape ishp (_param_in p)
--                 param_grad_new <- 
--                     if train_ then 
--                         liftIO $ mapM (ensure_shape ishp) (_param_grad p)
--                     else
--                         return Nothing
--                 return $! p {_param_in = param_in_new, _param_grad = param_grad_new}
--             ParameterA {} -> do
--                 let ishp = aux_shps M.! k
--                 ashp <- liftIO $ ndshape (_param_aux p)
--                 when (ishp /= ashp ) (throwM $ MismatchedShapeOfSym (k ++ "[i]") ishp ashp)
--                 return p
--     sess_placeholders .= shapes
--     args <- use sess_param
--     exec_handle <- liftIO $ do
--         names <- mxSymbolListArguments (unSymbol net)
--         -- the parameters to bind should be arranged in the same order as the names
--         let num_args = length names
--             arg_in  = map (unNDArray . _param_in) $ map (args M.!) names
--             arg_gr  = if train_ 
--                         then map (fmap unNDArray . _param_grad) $ map (args M.!) names
--                         else replicate num_args Nothing
--             arg_gr_req = replicate num_args (if train_ then 1 else 0)

--         auxnames <- mxSymbolListAuxiliaryStates (unSymbol net)
--         let aux_arg_aux = map (unNDArray . _param_aux) $ map (args M.!) auxnames
--         mxExecutorBind (unSymbol net) _device_type _device_id
--                         arg_in arg_gr arg_gr_req 
--                         aux_arg_aux
--     return $ Executor exec_handle
--   where
--     ensure_shape :: DType a => [Int] -> NDArray a -> IO (NDArray a)
--     ensure_shape src_shp ndarray = do
--         dst_cxt <- context ndarray
--         dst_shp <- ndshape ndarray
--         if src_shp == dst_shp 
--             then return ndarray
--             else makeEmptyNDArray src_shp dst_cxt

-- setPlaceholders :: (DType a, MonadIO m, MonadThrow m) => M.HashMap String (NDArray a) -> TrainM a m ()
-- setPlaceholders values = do
--     params <- use sess_param
--     placeHolders <- use sess_placeholders
--     forM_ (M.toList values) $ \ (key, ndarray_src) -> liftIO $ do
--         case M.lookup key params of
--             Nothing -> throwM $ NotAParameter key
--             Just param -> do
--                 let ndarray_dst = _param_in param
--                 shp_dst <- ndshape ndarray_dst
--                 shp_src <- ndshape ndarray_src
--                 case M.lookup key placeHolders of
--                     Just shp_plc | shp_src /= shp_plc -> throwM $ MismatchedShapeOfSym key shp_src shp_plc
--                     _ -> return ()
--                 when (shp_src /= shp_dst) $ throwM $ MismatchedShapeOfSym key shp_src shp_dst
--                 A._copyto_upd [unNDArray ndarray_dst] (#data := unNDArray ndarray_src .& Nil)

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
        let num_args = length names
            arg_in  = map (unNDArray . _param_in) $ map (args M.!) names
            arg_gr  = if train_ 
                        then map (fmap unNDArray . _param_grad) $ map (args M.!) names
                        else replicate num_args Nothing
            arg_gr_req = replicate num_args (if train_ then 1 else 0)

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
    => opt a -> Symbol a -> M.HashMap String (NDArray a) -> TrainM a m (Executor a)
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
    updateParameters opt datAndLbl
    liftIO performGC
    return exec

-- | single step train. Must provide all the placeholders.
fit_ :: (DType a, MonadIO m, MonadThrow m, Optimizer opt) 
     => opt a -> Symbol a -> M.HashMap String (NDArray a) -> TrainM a m ()
fit_ opt net datAndLbl = void $ fit opt net datAndLbl

-- | single step train. Must provide all the placeholders.
--   After fitting, it also update the evaluation metric.
fitAndEval :: (DType a, MonadIO m, MonadThrow m, Optimizer opt, EvalMetricMethod mtr)
           => opt a -> Symbol a -> M.HashMap String (NDArray a) -> mtr a -> TrainM a m ()
fitAndEval opt net datAndLbl metric = do
    Executor exec  <- fit opt net datAndLbl
    preds <- liftIO $ map NDArray <$> mxExecutorOutputs exec
    evaluate metric datAndLbl preds
    liftIO performGC

fitDataset :: (Dataset d, DType a, DataItem e a,
               MonadIO m, MonadThrow m, DatasetConstraint d (TrainM a m),
               Optimizer opt, EvalMetricMethod mtr)
    => opt a -> Symbol a 
    -> [String]
    -> d e
    -> mtr a
    -> TrainM a m ()
fitDataset opt net varnames dataset metric = do
    callbacks <- use sess_callbacks
    shapes <- use sess_placeholders
    total <- sizeD dataset
    [example0] <- takeD 1 dataset
    batchSize <- batchSizeD example0
    forM_ callbacks (begOfEpoch total)

    void $ forEachD_i dataset $ \(i, e) -> do
        forM_ callbacks (begOfBatch i batchSize)
        placeHolders <- makePlaceholderMapD e varnames
        fitAndEval opt net placeHolders metric
        eval <- format metric
        liftIO $ putStr $ printf "\r\ESC[K%d/%d %s" i total eval
        forM_ callbacks (endOfBatch i batchSize)
        liftIO $ hFlush stdout
        liftIO performGC

    forM_ callbacks (endOfEpoch total)
    liftIO $ hFlush stdout
-- fitDataset :: (Dataset d, DType a, DataItem e a,
--                MonadIO m, MonadThrow m, DatasetConstraint d (TrainM a m),
--                Optimizer opt, EvalMetricMethod mtr)
--     => opt a -> Symbol a 
--     -> [String]
--     -> d e
--     -> mtr a
--     -> TrainM a m ()
-- fitDataset opt net varnames dataset metric = do
--     callbacks <- use sess_callbacks
--     shapes <- use sess_placeholders
--     total <- sizeD dataset
--     [example0] <- takeD 1 dataset
--     batchSize <- batchSizeD example0
--     -- assuming the data shape axis-0 is the batch-size
--     Executor exec <- bind' net (M.map (\shp -> batchSize:tail shp) shapes) True

--     forM_ callbacks (begOfEpoch total)
--     void $ forEachD_i dataset $ \(i, e) -> do
--         forM_ callbacks (begOfBatch i batchSize)

--         t1 <- liftIO getCurrentTime

--         placeHolders <- makePlaceholderMapD e varnames
--         setPlaceholders placeHolders

--         t2 <- liftIO getCurrentTime
--         sess_prof . _1 += diffUTCTime t2 t1

--         liftIO $ do 
--             mxExecutorForward exec True
--             mxExecutorBackward exec []

--         t3 <- liftIO getCurrentTime
--         sess_prof . _2 += diffUTCTime t3 t2

--         updateParameters opt placeHolders

--         t4 <- liftIO getCurrentTime
--         sess_prof . _3 += diffUTCTime t4 t3

--         preds <- liftIO $ map NDArray <$> mxExecutorOutputs exec

--         t5 <- liftIO getCurrentTime
--         sess_prof . _4 += diffUTCTime t5 t4

--         evaluate metric placeHolders preds
--         eval <- format metric

--         t6 <- liftIO getCurrentTime
--         sess_prof . _5 += diffUTCTime t6 t5

--         liftIO $ putStr $ "\r\ESC[K" ++ show i ++ "/" ++ show total ++ " " ++ eval
--         forM_ callbacks (endOfBatch i batchSize)
--         liftIO $ hFlush stdout
--         liftIO performGC

--         t7 <- liftIO getCurrentTime
--         sess_prof . _6 += diffUTCTime t7 t6
--     forM_ callbacks (endOfEpoch total)
--     liftIO $ hFlush stdout

updateParameters :: (MonadIO m, Optimizer opt, DType dtype) 
                 => opt dtype -> M.HashMap String any -> TrainM dtype m ()
updateParameters opt blacklist = do
    params <- use sess_param
    forM_ (M.toList params) $ \ (k, v) -> do
        case (v, M.member k blacklist, _param_grad v) of 
          (ParameterI {}, False, Just grad) -> ST.lift $ optimize opt k (_param_in v) grad
          _ -> return ()
    ST.lift (stat_num_upd += 1)
    -- waitParams

-- | forward only. Must provide all the placeholders, setting the data to @Just xx@, and set label to @Nothing@.
-- 
-- Note that the batch size here can be different from that in the training phase.
forwardOnly :: (DType a, MonadIO m, MonadThrow m) => Symbol a -> M.HashMap String (Maybe (NDArray a)) -> TrainM a m [NDArray a]
forwardOnly net dat = do
    Executor exec <- bind net dat False
    liftIO $ mxExecutorForward exec False
    liftIO $ map NDArray <$> mxExecutorOutputs exec

waitParams :: (MonadIO m, DType a) => TrainM a m ()
waitParams = do
    params <- use sess_param
    forM_ params (\param -> 
        case param of 
            ParameterA arr1 -> 
                wait arr1
            ParameterI arr1 Nothing -> 
                wait arr1
            ParameterI arr1 (Just arr2) -> do
                wait arr1
                wait arr2
        )

wait :: (MonadIO m, DType a) => NDArray a -> TrainM a m ()
wait (NDArray hdl) = liftIO $ mxNDArrayWaitToRead hdl

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

