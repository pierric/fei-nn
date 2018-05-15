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
    module MXNet.NN.Optimizer
) where

import MXNet.Core.Base hiding (bind, context, (^.))
import MXNet.Core.Base.Internal
import qualified MXNet.Core.Base.NDArray as A
import qualified MXNet.Core.Base.Symbol as S
import qualified MXNet.Core.Base.Executor as E
import qualified MXNet.Core.Types.Internal as MXI
import qualified MXNet.Core.Base.Internal.TH.NDArray as MXI
import qualified Data.HashMap.Strict as M
import qualified Control.Monad.State.Strict as ST
import Data.Maybe (isJust, fromJust, maybe)
import Control.Monad (when, zipWithM_)
import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Monad.Trans.Resource (MonadThrow(..))
import Control.Lens (traverseOf, use, (^.))

import MXNet.NN.Types
import MXNet.NN.Optimizer
import MXNet.NN.EvalMetric

-- | Execute the 'TrainM' monad
train :: (DType a, Monad m) => Session a -> TrainM a m r -> m r
train = flip ST.evalStateT

-- | infer the shapes of all the symbols in a symbolic neural network
inferShape :: DType a => Symbol a -> M.HashMap String (NDArray a) -> IO (M.HashMap String [Int])
inferShape sym known = do
    let (names, vals) = unzip $ M.toList known
    shapes <- mapM ndshape vals
    let arg_ind = scanl (+) 0 $ map fst shapes
        arg_shp = concat $ map snd shapes
    (inp_shp, _, _) <- mxSymbolInferShape (S.getHandle sym) names arg_ind arg_shp
    inps <- listInputs sym
    return $ M.fromList $ zip inps inp_shp

-- | initialize all parameters
initialize :: DType a => Symbol a -> Config a -> IO (Session a)
initialize sym config = do
    let spec1 = M.difference (_cfg_placeholders config) (_cfg_initializers config)
        spec2 = _cfg_initializers config
        dinit = _cfg_default_initializer config
        cxt   = _cfg_context config
    placeholder  <- mapM (\shp -> makeEmptyNDArray shp cxt False) spec1
    inp_with_shp <- inferShape sym placeholder
    args <- M.traverseWithKey (init_with_random_normal placeholder spec2 dinit) inp_with_shp
    return $ Session args cxt
  where
    init_with_random_normal placeholder spec2 dinit inp shp = do
        case M.lookup inp placeholder of
            Just in_arg -> do
                nullarg <- MXI.nullNDArrayHandle
                return $ Parameter in_arg (A.NDArray nullarg)
            Nothing -> do
                arg_in <- case M.lookup inp spec2 of
                    Just cinit -> cinit shp (_cfg_context config)
                    Nothing    -> dinit shp (_cfg_context config)
                arg_gr <- makeEmptyNDArray shp (_cfg_context config) False
                return $ Parameter arg_in arg_gr

-- | bind the symbolic network with actual parameters
bind :: (DType a, MonadIO m, MonadThrow m) => Symbol a -> M.HashMap String (Maybe (NDArray a)) -> Bool -> TrainM a m (Executor a)
bind net dat train_ = do
    Context{..} <- use sess_context

    shps <- liftIO $ inferShape net (M.map fromJust $ M.filter isJust dat)
    modifyT . traverseOf sess_param $ M.traverseWithKey $ \k p -> do
        let ishp = shps M.! k
        case M.lookup k dat of
            Just a  -> liftIO $ update_param (maybe (Right ishp) Left a) p
            Nothing -> do
                (_, pshp1) <- liftIO $ ndshape (_param_in p)
                when (ishp /= pshp1 ) (throwM $ MismatchedShapeOfSym k)
                when train_ $ do
                    (_, pshp2) <- liftIO $ ndshape (_param_grad p)
                    when (ishp /= pshp2) (throwM $ MismatchedShapeOfSym k)
                return p

    args <- use sess_param
    exec_handle <- liftIO $ do
        names <- listInputs net
        nullarg <- MXI.nullNDArrayHandle
        -- the parameters to bind should be arranged in the same order as the names
        let arg_num = fromIntegral (M.size args)
            arg_in  = map (A.getHandle . _param_in) $ map (args M.!) names
            arg_gr  = if train_ 
                        then map (A.getHandle . _param_grad) $ map (args M.!) names
                        else replicate (M.size args) nullarg
            arg_gr_req = replicate (M.size args) 1

        checked $ mxExecutorBind (S.getHandle net) deviceType deviceId
                                            arg_num arg_in arg_gr arg_gr_req 
                                            0 []
    return $ E.Executor exec_handle
  where
    update_param :: DType a => Either (NDArray a) [Int] -> Parameter a -> IO (Parameter a)
    update_param (Left a) p = do
        src_cxt <- A.context a
        src_shp <- snd <$> A.ndshape a
        dst_cxt <- A.context (_param_in p)
        dst_shp <- snd <$> A.ndshape (_param_in p)
        case (src_cxt == dst_cxt, src_shp == dst_shp) of
            (True , True) -> return $ p {_param_in = a}
            (False, True) -> do
                MXI._copyto' (A.getHandle a) [A.getHandle (_param_in p)] :: IO ()
                return p
            _ -> do
                a_copy <- makeEmptyNDArray src_shp dst_cxt False
                MXI._copyto' (A.getHandle a) [A.getHandle a_copy] :: IO ()
                return $! p {_param_in = a_copy}    
    update_param (Right src_shp) p = do
        dst_cxt <- A.context (_param_in p)
        dst_shp <- snd <$> A.ndshape (_param_in p)
        if src_shp == dst_shp 
            then return p
            else do
                dummy <- makeEmptyNDArray src_shp dst_cxt False
                return $! p {_param_in = dummy}

-- | single step train. Must provide all the placeholders.
fit :: (DType a, MonadIO m, MonadThrow m, Optimizer opt, OptArgsCst opt g) 
    => opt a g -> Symbol a -> M.HashMap String (NDArray a) -> TrainM a m ()
fit opt net datAndLbl = do
    exec <- bind net (M.map Just datAndLbl) True
    liftIO $ do 
        checked $ mxExecutorForward (E.getHandle exec) 1
        checked $ mxExecutorBackward (E.getHandle exec) 0 []
        -- forward/backward are asynchronised operation in mxnet, in a
        -- sense that only opcodes are pushed onto an internal execution 
        -- stack, and there is a executor running in a separate thread.
        -- It is possible that an OOM of CPU memory occurs, if 'fit' are 
        -- called so fast that too many opcodes and data on the stack, 
        -- as described in issue #1
        checked $ mxNDArrayWaitAll
    updateParameters opt datAndLbl

-- | single step train. Must provide all the placeholders.
--   After fitting, it also update the evaluation metric.
fitAndEval :: (DType a, MonadIO m, MonadThrow m, Optimizer opt, OptArgsCst opt g, EvalMetricMethod mth)
           => opt a g -> Symbol a -> M.HashMap String (NDArray a) -> Metric a mth -> TrainM a m ()
fitAndEval opt net datAndLbl metric = do
     exec  <- bind net (M.map Just datAndLbl) True
     preds <- liftIO $ do 
         checked $ mxExecutorForward (E.getHandle exec) 1
         checked $ mxExecutorBackward (E.getHandle exec) 0 []
         checked $ mxNDArrayWaitAll
         getOutputs exec
     updateParameters opt datAndLbl
     let labels = map (datAndLbl M.!) (metric ^. metric_labelname)
     liftIO $ zipWithM_ (evaluate metric) preds labels

updateParameters :: (MonadIO m, Optimizer opt, OptArgsCst opt args, DType dtype) 
                 => opt dtype args -> M.HashMap String any -> TrainM dtype m ()
updateParameters opt blacklist = do
    modifyT . traverseOf sess_param  $ M.traverseWithKey $ \ k v -> do
        if (not $ M.member k blacklist)
            then do new_in <- liftIO $ optimize opt k (_param_in v) (_param_grad v) 
                    -- must evaluate the new parameter to WHNF
                    -- otherwise, the old _param_in is retained.
                    -- if context is GPU, then OOM will soon 
                    -- occur, as described in issue #2
                    return $! v {_param_in = new_in}
            else return v

-- | forward only. Must provide all the placeholders, setting the data to @Just xx@, and set label to @Nothing@.
-- 
-- Note that the batch size here can be different from that in the training phase.
forwardOnly :: (DType a, MonadIO m, MonadThrow m) => Symbol a -> M.HashMap String (Maybe (NDArray a)) -> TrainM a m [NDArray a]
forwardOnly net dat = do
    exec <- bind net dat False
    liftIO $ do
        checked $ mxExecutorForward (E.getHandle exec) 0
        -- for the same reason in 'fit'.
        checked $ mxNDArrayWaitAll
        getOutputs exec

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

