{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE UndecidableInstances #-}

module MXNet.NN.Optimizer (
    Optimizer(..),
    OptimizerTag(..)
) where

import MXNet.Base hiding (Symbol)
import qualified MXNet.Base.Operators.NDArray as A

import Data.IORef
import GHC.TypeLits
import GHC.Exts (Constraint)
import qualified Data.HashMap.Strict as M
import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Monad.State.Class (MonadState)
import Control.Lens (use, (.=))
import MXNet.NN.LrScheduler (LrScheduler(..))
import MXNet.NN.TaggedState (Tagged, untag)
import MXNet.NN.Types (TaggedModuleState, mod_statistics, Statistics, stat_num_upd, stat_last_lr)

-- | Abstract Optimizer type class
class Optimizer (opt :: * -> *) where
    data OptimizerTag opt :: *
    -- | Specific required arguments
    -- data ReqArgs opt :: *
    -- | Specific optional arguments
    -- type OptArgsList opt :: [KV *]
    -- | make the optimizer
    makeOptimizer :: (DType dtype, LrScheduler sch, OptimizerCst opt dtype args)
                  => OptimizerTag opt -> sch -> ArgsHMap (OptimizerSym opt) args -> IO (opt dtype)
    -- | run the optimizer with the input & expected tensor
    optimize :: (DType dtype, MonadState (TaggedModuleState dtype t) m, MonadIO m)
             => opt dtype                            -- optimizer
             -> String                               -- symbol name to optimize
             -> NDArray dytpe                        -- parameter
             -> NDArray dtype                        -- gradient
             -> m ()

type family OptimizerSym (opt :: * -> *) :: Symbol
type family OptimizerCst (opt :: * -> *) dt (args :: [*]) :: Constraint

-- | SGD optimizer
data SGD_Opt dtype where
    SGD_Opt :: (LrScheduler sch, OptimizerCst SGD_Opt dtype args)
            => sch -> ArgsHMap (OptimizerSym SGD_Opt) args -> SGD_Opt dtype

type instance OptimizerSym SGD_Opt = "sgd_update(ndarray)"
-- 1.0.0 type instance OptimizerCst SGD_Opt dt args = HasArgs (OptimizerSym SGD_Opt) args '["wd", "rescale_grad", "clip_gradient"]
type instance OptimizerCst SGD_Opt dt args = HasArgs (OptimizerSym SGD_Opt) args '["wd", "rescale_grad", "clip_gradient", "lazy_update"]

instance Optimizer SGD_Opt where
    data OptimizerTag SGD_Opt = SGD
    makeOptimizer SGD sch args = return $ SGD_Opt sch args
    optimize (SGD_Opt sch args) _ (NDArray weight) (NDArray gradient) = do
        nup <- use $ untag . mod_statistics . stat_num_upd
        let lr = getLR sch nup
        untag . mod_statistics . stat_last_lr .= lr
        liftIO $ A.sgd_update_upd [weight] (
            #weight := weight   .&
            #grad   := gradient .&
            #lr     := lr       .& args)

-- | SGD with momentum optimizer
data SGD_Mom_Opt dtype where
    SGD_Mom_Opt :: (LrScheduler sch, OptimizerCst SGD_Mom_Opt dtype args)
                => sch -> ArgsHMap (OptimizerSym SGD_Mom_Opt) args -> (IORef (M.HashMap String (NDArray dtype))) -> SGD_Mom_Opt dtype

type instance OptimizerSym SGD_Mom_Opt = "sgd_mom_update(ndarray)"
-- 1.0.0 type instance OptimizerCst SGD_Mom_Opt dt args = HasArgs (OptimizerSym SGD_Mom_Opt) args '["momentum", "wd", "rescale_grad", "clip_gradient"]
type instance OptimizerCst SGD_Mom_Opt dt args = HasArgs (OptimizerSym SGD_Mom_Opt) args '["momentum", "wd", "rescale_grad", "clip_gradient", "lazy_update"]

instance Optimizer SGD_Mom_Opt where
    data OptimizerTag SGD_Mom_Opt = SGD'Mom
    makeOptimizer SGD'Mom sch args = do
        empty <- newIORef M.empty
        return $ SGD_Mom_Opt sch args empty

    optimize (SGD_Mom_Opt sch args emaref) symbol (NDArray weight) (NDArray gradient) = do
        nup <- use $ untag . mod_statistics . stat_num_upd
        let lr = getLR sch nup
        untag . mod_statistics . stat_last_lr .= lr
        liftIO $ do
            ema <- readIORef emaref
            momentum <- case M.lookup symbol ema of
                Nothing    -> do
                    [mom] <- A.zeros_like (#data := weight .& Nil)
                    writeIORef emaref (M.insert symbol (NDArray mom) ema)
                    return mom
                Just (NDArray a) -> return a
            A.sgd_mom_update_upd [weight] (
                #weight := weight   .&
                #grad   := gradient .&
                #mom    := momentum .&
                #lr     := lr       .& args)

-- | ADAM optmizer
data ADAM_Opt dtype where
    ADAM_Opt :: (LrScheduler sch, OptimizerCst ADAM_Opt dtype args)
            => sch -> ArgsHMap (OptimizerSym ADAM_Opt) args -> IORef (M.HashMap String (NDArray dtype, NDArray dtype)) -> ADAM_Opt dtype

type instance OptimizerSym ADAM_Opt = "adam_update(ndarray)"
-- 1.0.0 type instance OptimizerCst ADAM_Opt dt args = HasArgs (OptimizerSym ADAM_Opt) args '["beta1", "beta2", "epsilon", "wd", "rescale_grad", "clip_gradient"]
type instance OptimizerCst ADAM_Opt dt args = HasArgs (OptimizerSym ADAM_Opt) args '["beta1", "beta2", "epsilon", "wd", "rescale_grad", "clip_gradient", "lazy_update"]

instance Optimizer ADAM_Opt where
    data OptimizerTag ADAM_Opt = ADAM
    makeOptimizer ADAM sch args = do
        empty <- newIORef M.empty
        return $ ADAM_Opt sch args empty

    optimize (ADAM_Opt sch args emaref) symbol (NDArray weight) (NDArray gradient) = do
        nup <- use $ untag . mod_statistics . stat_num_upd
        let lr = getLR sch nup
        untag . mod_statistics . stat_last_lr .= lr
        liftIO $ do
            ema <- readIORef emaref
            (moving_avg, moving_var) <- case M.lookup symbol ema of
                Nothing    -> do
                    [avg] <- A.zeros_like (#data := weight .& Nil)
                    [var] <- A.zeros_like (#data := weight .& Nil)
                    writeIORef emaref (M.insert symbol (NDArray avg, NDArray var) ema)
                    return (avg, var)
                Just (NDArray a, NDArray v) -> return (a, v)
            A.adam_update_upd [weight] (
                #weight := weight     .&
                #grad   := gradient   .&
                #mean   := moving_avg .&
                #var    := moving_var .&
                #lr     := lr         .& args)
