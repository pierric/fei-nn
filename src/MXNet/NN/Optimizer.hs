{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ConstraintKinds #-}

module MXNet.NN.Optimizer (
    Optimizer(..),
    OptArgsCst,
    SGD, ADAM
) where

import qualified Data.HashMap.Strict as M
import MXNet.Core.Base.NDArray (NDArray)
import qualified MXNet.Core.Base.NDArray as A
import MXNet.Core.Base.HMap
import MXNet.Core.Base.Internal.TH.NDArray as A
import Data.IORef

-- | Constraint of using an optimizer
type OptArgsCst opt args = (ShowKV args, MatchKVList args (OptArgsList opt))

-- | Abstract Optimizer type class
class Optimizer (opt :: * -> [KV *] -> *) where
    -- | Specific constraints of the optimizer
    type OptArgsList opt :: [KV *]
    -- | make the optimizer
    makeOptimizer :: OptArgsCst opt args => Float -> HMap args -> IO (opt dtype args)
    -- | run the optimizer with the input & expected tensor
    optimize :: OptArgsCst opt args => opt dtype args -> String -> NDArray dytpe -> NDArray dtype -> IO (NDArray dtype)

-- | SGD optimizer
data SGD dtype args = SGD Float (HMap args)

instance Optimizer SGD where
    type OptArgsList SGD = '["wd"            ':= Float,
                             "rescale_grad"  ':= Float,
                             "clip_gradient" ':= Float]
    makeOptimizer lr args = return $ SGD lr args
    optimize (SGD lr args) _ weight gradient = A.NDArray <$> A.sgd_update (A.getHandle weight) (A.getHandle gradient) lr args

-- | ADAM optmizer
data ADAM dtype args = ADAM Float (HMap args) (IORef (M.HashMap String (NDArray dtype, NDArray dtype)))

instance Optimizer ADAM where
    type OptArgsList ADAM = '["beta1"         ':= Float,
                              "beta2"         ':= Float,
                              "epsilon"       ':= Float,
                              "wd"            ':= Float,
                              "rescale_grad"  ':= Float,
                              "clip_gradient" ':= Float]
    makeOptimizer lr args = do
        empty <- newIORef M.empty
        return $ ADAM lr args empty

    optimize (ADAM lr args emaref) symbol weight gradient = do
        ema <- readIORef emaref
        (moving_avg, moving_var) <- case M.lookup symbol ema of
            Nothing    -> do
                avg <- A.zeros_like (A.getHandle weight) 
                var <- A.zeros_like (A.getHandle weight)
                writeIORef emaref (M.insert symbol (A.NDArray avg, A.NDArray var) ema)
                return (avg, var)
            Just (a,v) -> return (A.getHandle a, A.getHandle v)
        A.NDArray <$> adam_update (A.getHandle weight) (A.getHandle gradient) moving_avg moving_var lr args
