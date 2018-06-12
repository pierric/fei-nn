{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE ExistentialQuantification #-}

module MXNet.NN.Optimizer (
    Optimizer(..),
    ReqArgs(..),
    OptArgsCst,
    SGD_Opt, SGD_Mom_Opt, ADAM_Opt
) where

import qualified Data.HashMap.Strict as M
import MXNet.Core.Base (DType)
import MXNet.Core.Base.NDArray (NDArray)
import qualified MXNet.Core.Base.NDArray as A
import MXNet.Core.Base.HMap
import MXNet.Core.Base.Internal.TH.NDArray as A
import Data.IORef
import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Monad.State.Class (MonadState)
import Control.Lens (use)
import MXNet.NN.LrScheduler (LrScheduler(..))
import MXNet.NN.Types (Statistics, stat_num_upd)

-- | Constraint of using an optimizer
type OptArgsCst opt args = (ShowKV args, MatchKVList args (OptArgsList opt))

-- | Abstract Optimizer type class
class Optimizer (opt :: * -> [KV *] -> *) where
    -- | Specific required arguments
    data ReqArgs opt :: *
    -- | Specific optional arguments
    type OptArgsList opt :: [KV *]
    -- | make the optimizer
    makeOptimizer :: (OptArgsCst opt oargs, DType dtype) => ReqArgs opt -> HMap oargs -> IO (opt dtype oargs)
    -- | run the optimizer with the input & expected tensor
    optimize :: (OptArgsCst opt oargs, DType dtype, MonadIO m, MonadState Statistics m) 
             => opt dtype oargs                      -- optimizer
             -> String                               -- symbol name to optimize
             -> NDArray dytpe                        -- parameter
             -> NDArray dtype                        -- gradient
             -> m (NDArray dtype)

data Base_Opt args = forall sch. LrScheduler sch => Base_Opt sch (HMap args)

-- | SGD optimizer
data SGD_Opt dtype args = SGD_Opt (Base_Opt args)

instance Optimizer SGD_Opt where
    data ReqArgs SGD_Opt = forall sch. LrScheduler sch => SGD sch
    type OptArgsList SGD_Opt = '["wd"            ':= Float,
                                 "rescale_grad"  ':= Float,
                                 "clip_gradient" ':= Float]
    makeOptimizer (SGD sch) args = return $ SGD_Opt $ Base_Opt sch args
    optimize (SGD_Opt (Base_Opt sch args)) _ weight gradient = do
        nup <- use stat_num_upd
        let lr = getLR sch nup
        liftIO $ A.NDArray <$> A.sgd_update (A.getHandle weight) (A.getHandle gradient) lr args

-- | SGD with momentum optimizer
data SGD_Mom_Opt dtype args = SGD_Mom_Opt (Base_Opt args) (IORef (M.HashMap String (NDArray dtype)))

instance Optimizer SGD_Mom_Opt where
    data ReqArgs SGD_Mom_Opt = forall sch. LrScheduler sch => SGD'Mom sch
    type OptArgsList SGD_Mom_Opt = '["momentum"      ':= Float,
                                     "wd"            ':= Float,
                                     "rescale_grad"  ':= Float,
                                     "clip_gradient" ':= Float]
    makeOptimizer (SGD'Mom sch) args = do
        empty <- newIORef M.empty
        return $ SGD_Mom_Opt (Base_Opt sch args) empty

    optimize (SGD_Mom_Opt (Base_Opt sch args) emaref) symbol weight gradient = do
        nup <- use stat_num_upd
        let lr = getLR sch nup
        liftIO $ do
            ema <- readIORef emaref
            momentum <- case M.lookup symbol ema of
                Nothing    -> do
                    mom <- A.zeros_like (A.getHandle weight) 
                    writeIORef emaref (M.insert symbol (A.NDArray mom) ema)
                    return mom
                Just a -> return (A.getHandle a)
            A.NDArray <$> A.sgd_mom_update (A.getHandle weight) (A.getHandle gradient) momentum lr args

-- | ADAM optmizer
data ADAM_Opt dtype args = ADAM_Opt (Base_Opt args) (IORef (M.HashMap String (NDArray dtype, NDArray dtype)))

instance Optimizer ADAM_Opt where
    data ReqArgs ADAM_Opt = forall sch. LrScheduler sch => ADAM sch
    type OptArgsList ADAM_Opt = '["beta1"         ':= Float,
                                  "beta2"         ':= Float,
                                  "epsilon"       ':= Float,
                                  "wd"            ':= Float,
                                  "rescale_grad"  ':= Float,
                                  "clip_gradient" ':= Float]
    makeOptimizer (ADAM sch) args = do
        empty <- newIORef M.empty
        return $ ADAM_Opt (Base_Opt sch args) empty

    optimize (ADAM_Opt (Base_Opt sch args) emaref) symbol weight gradient = do
        nup <- use stat_num_upd
        let lr = getLR sch nup
        liftIO $ do
            ema <- readIORef emaref
            (moving_avg, moving_var) <- case M.lookup symbol ema of
                Nothing    -> do
                    avg <- A.zeros_like (A.getHandle weight) 
                    var <- A.zeros_like (A.getHandle weight)
                    writeIORef emaref (M.insert symbol (A.NDArray avg, A.NDArray var) ema)
                    return (avg, var)
                Just (a,v) -> return (A.getHandle a, A.getHandle v)
            A.NDArray <$> adam_update (A.getHandle weight) (A.getHandle gradient) moving_avg moving_var lr args
