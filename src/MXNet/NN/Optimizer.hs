{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ConstraintKinds #-}

module MXNet.NN.Optimizer (
    Optimizer(..),
    ReqArgs(..),
    OptArgsCst,
    SGD, ADAM
) where

import qualified Data.HashMap.Strict as M
import MXNet.Core.Base (Context, DType)
import MXNet.Core.Base.NDArray (NDArray)
import qualified MXNet.Core.Base.NDArray as A
import MXNet.Core.Base.HMap
import MXNet.Core.Base.Internal.TH.NDArray as A
import Data.IORef

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
    optimize :: (OptArgsCst opt oargs, DType dtype) => opt dtype oargs -> String -> NDArray dytpe -> NDArray dtype -> IO (NDArray dtype)

-- | SGD optimizer
data SGD dtype args = SGD Float (HMap args)

instance Optimizer SGD where
    newtype ReqArgs SGD = SGD'Args Float
    type OptArgsList SGD = '["wd"            ':= Float,
                             "rescale_grad"  ':= Float,
                             "clip_gradient" ':= Float]
    makeOptimizer (SGD'Args lr) args = return $ SGD lr args
    optimize (SGD lr args) _ weight gradient = A.NDArray <$> A.sgd_update (A.getHandle weight) (A.getHandle gradient) lr args

-- | SGD with momentum optimizer
data SGD'Mom dtype args = SGD'Mom Float (HMap args) (NDArray dtype)

instance Optimizer SGD'Mom where
    data ReqArgs SGD'Mom = SGD'Mom'Args { _sgd_mom_args_lr :: Float
                                        , _sgd_mom_args_shape :: [Int]
                                        , _sgd_mom_args_context :: Context }
    type OptArgsList SGD'Mom = '["momentum"      ':= Float,
                                 "wd"            ':= Float,
                                 "rescale_grad"  ':= Float,
                                 "clip_gradient" ':= Float]
    makeOptimizer (SGD'Mom'Args lr shp cxt) args = do
        mom <- A.makeEmptyNDArray shp cxt False
        return $ SGD'Mom lr args mom
    optimize (SGD'Mom lr args mom) _ weight gradient = A.NDArray <$> A.sgd_mom_update (A.getHandle weight) (A.getHandle gradient) (A.getHandle mom) lr args

-- | ADAM optmizer
data ADAM dtype args = ADAM Float (HMap args) (IORef (M.HashMap String (NDArray dtype, NDArray dtype)))

instance Optimizer ADAM where
    newtype ReqArgs ADAM = ADAM'Args Float
    type OptArgsList ADAM = '["beta1"         ':= Float,
                              "beta2"         ':= Float,
                              "epsilon"       ':= Float,
                              "wd"            ':= Float,
                              "rescale_grad"  ':= Float,
                              "clip_gradient" ':= Float]
    makeOptimizer (ADAM'Args lr) args = do
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
