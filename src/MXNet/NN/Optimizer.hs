{-# LANGUAGE CPP                  #-}
{-# LANGUAGE ConstraintKinds      #-}
{-# LANGUAGE OverloadedLists      #-}
{-# LANGUAGE UndecidableInstances #-}
module MXNet.NN.Optimizer where

import           Control.Lens                (use, (.=))
import           GHC.Exts                    (Constraint)
import           GHC.TypeLits
import           RIO
import qualified RIO.HashMap                 as M
import           RIO.State

import           MXNet.Base                  hiding (Symbol)
import qualified MXNet.Base.Operators.Tensor as T
import           MXNet.NN.LrScheduler        (LrScheduler (..))
import           MXNet.NN.TaggedState        (untag)
import           MXNet.NN.Types              (TaggedModuleState, mod_statistics,
                                              stat_last_lr, stat_num_upd)

-- | Abstract Optimizer type class
class Optimizer (opt :: * -> *) where
    data OptimizerTag opt :: *
    -- | Specific required arguments
    -- data ReqArgs opt :: *
    -- | Specific optional arguments
    -- type OptArgsList opt :: [KV *]
    -- | make the optimizer
    makeOptimizer :: (DType dtype, LrScheduler sch, OptimizerCst opt dtype args, MonadIO m)
                  => OptimizerTag opt -> sch
                  -> ArgsHMap (OptimizerSym opt) '(NDArray, dtype) args
                  -> m (opt dtype)
    -- | run the optimizer with the input & expected tensor
    optimize :: (DType dtype, MonadState (TaggedModuleState dtype t) m, MonadIO m)
             => opt dtype                            -- optimizer
             -> Text                                 -- symbol name to optimize
             -> NDArray dtype                        -- parameter
             -> NDArray dtype                        -- gradient
             -> m ()

type family OptimizerSym (opt :: * -> *) :: Symbol
type family OptimizerCst (opt :: * -> *) dt (args :: [*]) :: Constraint

-- | SGD optimizer
data SGD_Opt dtype where
    SGD_Opt :: (LrScheduler sch, OptimizerCst SGD_Opt dtype args)
            => sch -> ArgsHMap (OptimizerSym SGD_Opt) '(NDArray, dtype) args
            -> SGD_Opt dtype

type instance OptimizerSym SGD_Opt = "_sgd_update"
-- 1.0.0 type instance OptimizerCst SGD_Opt dt args = HasArgs (OptimizerSym SGD_Opt) args '["wd", "rescale_grad", "clip_gradient"]
type instance OptimizerCst SGD_Opt dt args =
    HasArgs (OptimizerSym SGD_Opt) '(NDArray, dt) args '["wd", "rescale_grad", "clip_gradient", "lazy_update"]

instance Optimizer SGD_Opt where
    data OptimizerTag SGD_Opt = SGD
    makeOptimizer SGD sch args = return $ SGD_Opt sch args
    optimize (SGD_Opt sch args) _ weight gradient = do
        nup <- use $ untag . mod_statistics . stat_num_upd
        let lr = getLR sch nup
        untag . mod_statistics . stat_last_lr .= lr
        liftIO $ void $ T._sgd_update (#weight := weight
                                    .& #grad   := gradient
                                    .& #lr     := lr .& args)
            (Just [weight])

-- | SGD with momentum optimizer
data SGD_Mom_Opt dtype where
    SGD_Mom_Opt :: (LrScheduler sch, OptimizerCst SGD_Mom_Opt dtype args)
                => sch -> ArgsHMap (OptimizerSym SGD_Mom_Opt) '(NDArray, dtype) args
                -> (IORef (M.HashMap Text (NDArray dtype)))
                -> SGD_Mom_Opt dtype

type instance OptimizerSym SGD_Mom_Opt = "_sgd_mom_update"
-- 1.0.0 type instance OptimizerCst SGD_Mom_Opt dt args = HasArgs (OptimizerSym SGD_Mom_Opt) args '["momentum", "wd", "rescale_grad", "clip_gradient"]
type instance OptimizerCst SGD_Mom_Opt dt args =
    HasArgs (OptimizerSym SGD_Mom_Opt) '(NDArray, dt) args '["momentum", "wd", "rescale_grad", "clip_gradient", "lazy_update"]

instance Optimizer SGD_Mom_Opt where
    data OptimizerTag SGD_Mom_Opt = SGD'Mom
    makeOptimizer SGD'Mom sch args = do
        empty <- newIORef M.empty
        return $ SGD_Mom_Opt sch args empty

    optimize (SGD_Mom_Opt sch args emaref) symbol weight gradient = do
        nup <- use $ untag . mod_statistics . stat_num_upd
        let lr = getLR sch nup
        untag . mod_statistics . stat_last_lr .= lr
        liftIO $ do
            ema <- readIORef emaref
            momentum <- case M.lookup symbol ema of
                Nothing    -> do
                    mom <- prim T._zeros_like (#data := weight .& Nil)
                    writeIORef emaref (M.insert symbol mom ema)
                    return mom
                Just a -> return a
            -- let norm x = prim T._norm (#data := x .& #ord := 1 .& Nil)
            -- [w0] <- toVector =<< norm weight
            -- [m0] <- toVector =<< norm momentum
            -- [g0] <- toVector =<< norm gradient
            void $ T._sgd_mom_update (#weight := weight
                                   .& #grad   := gradient
                                   .& #mom    := momentum
                                   .& #lr     := lr .& args)
                (Just [weight])
            -- [w1] <- toVector =<< norm weight
            -- [m1] <- toVector =<< norm momentum
            -- traceShowM ("opt", symbol, w0, m0, g0, w1, m1)

-- | ADAM optmizer
data ADAM_Opt dtype where
    ADAM_Opt :: (LrScheduler sch, OptimizerCst ADAM_Opt dtype args)
            => sch -> ArgsHMap (OptimizerSym ADAM_Opt) '(NDArray, dtype) args
            -> IORef (M.HashMap Text (NDArray dtype, NDArray dtype))
            -> ADAM_Opt dtype

type instance OptimizerSym ADAM_Opt = "_adam_update"
-- 1.0.0 type instance OptimizerCst ADAM_Opt dt args = HasArgs (OptimizerSym ADAM_Opt) args '["beta1", "beta2", "epsilon", "wd", "rescale_grad", "clip_gradient"]
type instance OptimizerCst ADAM_Opt dt args =
    HasArgs (OptimizerSym ADAM_Opt) '(NDArray, dt) args '["beta1", "beta2", "epsilon", "wd", "rescale_grad", "clip_gradient", "lazy_update"]

instance Optimizer ADAM_Opt where
    data OptimizerTag ADAM_Opt = ADAM
    makeOptimizer ADAM sch args = do
        empty <- newIORef M.empty
        return $ ADAM_Opt sch args empty

    optimize (ADAM_Opt sch args emaref) symbol weight gradient = do
        nup <- use $ untag . mod_statistics . stat_num_upd
        let lr = getLR sch nup
        untag . mod_statistics . stat_last_lr .= lr
        liftIO $ do
            ema <- readIORef emaref
            (moving_avg, moving_var) <- case M.lookup symbol ema of
                Nothing    -> do
                    avg <- prim T._zeros_like (#data := weight .& Nil)
                    var <- prim T._ones_like  (#data := weight .& Nil)
                    writeIORef emaref (M.insert symbol (avg, var) ema)
                    return (avg, var)
                Just (a, v) -> return (a, v)
            void $ T._adam_update (#weight := weight
                                .& #grad   := gradient
                                .& #mean   := moving_avg
                                .& #var    := moving_var
                                .& #lr     := lr .& args)
                (Just [weight])


#if MXNET_VERSION >= 10600

data ADAMW_Opt dtype where
    ADAMW_Opt :: (LrScheduler sch, OptimizerCst ADAMW_Opt dtype args)
              => sch -> ArgsHMap (OptimizerSym ADAMW_Opt) '(NDArray, dtype) args
              -> IORef (M.HashMap Text (NDArray dtype, NDArray dtype))
              -> ADAMW_Opt dtype

type instance OptimizerSym ADAMW_Opt = "__adamw_update"
type instance OptimizerCst ADAMW_Opt dt args =
    HasArgs (OptimizerSym ADAMW_Opt) '(NDArray, dt) args
            '["beta1", "beta2", "epsilon", "wd", "eta", "clip_gradient", "rescale_grad"]

instance Optimizer ADAMW_Opt where
    data OptimizerTag ADAMW_Opt = ADAMW
    makeOptimizer ADAMW sch args = do
        empty <- newIORef M.empty
        return $ ADAMW_Opt sch args empty

    optimize (ADAMW_Opt sch args emaref) symbol weight gradient = do
        nup <- use $ untag . mod_statistics . stat_num_upd
        let lr = getLR sch nup
        untag . mod_statistics . stat_last_lr .= lr
        liftIO $ do
            ema <- readIORef emaref
            (moving_avg, moving_var) <- case M.lookup symbol ema of
                Nothing    -> do
                    avg <- prim T._zeros_like (#data := weight .& Nil)
                    var <- prim T._ones_like  (#data := weight .& Nil)
                    writeIORef emaref (M.insert symbol (avg, var) ema)
                    return (avg, var)
                Just (a, v) -> return (a, v)
            void $ T.__adamw_update
                (#weight := weight     .&
                 #grad   := gradient   .&
                 #mean   := moving_avg .&
                 #var    := moving_var .&
                 #lr     := lr         .& args)
                (Just [weight])

#endif

