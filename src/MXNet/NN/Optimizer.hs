{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE ConstrainedClassMethods #-}

module MXNet.NN.Optimizer where

import qualified Data.HashMap.Strict as M
import MXNet.Core.Base.NDArray (NDArray)
import qualified MXNet.Core.Base.NDArray as A
import MXNet.Core.Base.HMap
import MXNet.Core.Base.Internal.TH.NDArray as A
import Data.IORef
import GHC.Exts (Constraint)

class Optimizer opt where
    type OptDType   opt :: *
    type OptArgsLst opt :: [KV *]
    type OptArgsCst opt :: Constraint
    makeOptimizer :: OptArgsCst opt => Float -> HMap (OptArgsLst opt) -> IO opt
    optimize :: OptArgsCst opt => opt -> String -> NDArray dytpe -> NDArray dtype -> IO (NDArray dtype)

-- | SGD optimizer
data SGD dtype args = SGD Float (HMap args)

instance Optimizer (SGD dt args) where
    type OptDType   (SGD dt args) = dt
    type OptArgsLst (SGD dt args) = args
    type OptArgsCst (SGD dt args) = (ShowKV args, MatchKVList args '["wd"            ':= Float,
                                                                  "rescale_grad"  ':= Float,
                                                                  "clip_gradient" ':= Float])
    makeOptimizer lr args = return $ SGD lr args
    optimize (SGD lr args) _ weight gradient = A.NDArray <$> A.sgd_update (A.getHandle weight) (A.getHandle gradient) lr args

-- | ADAM optmizer
data ADAM dtype args = ADAM Float (HMap args) (IORef (M.HashMap String (NDArray dtype, NDArray dtype)))

instance Optimizer (ADAM dt args) where
    type OptDType   (ADAM dt args) = dt
    type OptArgsLst (ADAM dt args) = args
    type OptArgsCst (ADAM dt args) = (ShowKV args, MatchKVList args '["beta1"         ':= Float,
                                                                   "beta2"         ':= Float,
                                                                   "epsilon"       ':= Float,
                                                                   "wd"            ':= Float,
                                                                   "rescale_grad"  ':= Float,
                                                                   "clip_gradient" ':= Float])
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
