{-# LANGUAGE DataKinds #-}
{-# LANGUAGE UndecidableInstances #-}

module MXNet.NN.Layer (
  variable,
  convolution,
  fullyConnected,
  pooling,
  activation,
  softmaxoutput,
  batchnorm,
  cast,
  plus,
  flatten,
  identity,
) where

import MXNet.Base
import qualified MXNet.Base.Operators.Symbol as S

variable :: String -> IO SymbolHandle
variable = mxSymbolCreateVariable

convolution :: (HasOptArg "_Convolution(symbol)" args '["stride", "dilate", "pad", "num_group", "workspace", "layout", "cudnn_tune", "cudnn_off", "no_bias"]
               ,HasReqArg "_Convolution(symbol)" args '["kernel", "num_filter", "data"])
            => String -> ArgsHMap "_Convolution(symbol)" args -> IO SymbolHandle
convolution name args = do
    b <- variable (name ++ "-bias")
    w <- variable (name ++ "-weight")
    S._Convolution name (#weight := w .& #bias := b .& args)

-- fullyConnected :: (MatchKVList kvs '["no_bias" ':= Bool, 
--                                      "flatten" ':= Bool]
--                   ,ShowKV kvs, QueryKV kvs) 
--                => String -> SymbolHandle -> Int -> HMap kvs -> IO SymbolHandle
fullyConnected :: (HasOptArg "_FullyConnected(symbol)" args '["flatten", "no_bias"]
                  ,HasReqArg "_FullyConnected(symbol)" args '["data", "num_hidden"])
              => String -> ArgsHMap "_FullyConnected(symbol)" args -> IO SymbolHandle
fullyConnected name args = do
  b <- variable (name ++ "-bias")
  w <- variable (name ++ "-weight")
  S._FullyConnected name (#weight := w .& #bias := b .& args)

pooling :: (HasOptArg "_Pooling(symbol)" args '["stride", "pad", "pooling_convention", "global_pool", "cudnn_off"]
           ,HasReqArg "_Pooling(symbol)" args '["data", "kernel", "pool_type"])
        => String -> ArgsHMap "_Pooling(symbol)" args -> IO SymbolHandle
pooling = S._Pooling

activation :: (HasReqArg "_Activation(symbol)" args '["data", "act_type"])
        => String -> ArgsHMap "_Activation(symbol)" args -> IO SymbolHandle
activation = S._Activation

softmaxoutput :: (HasOptArg "_SoftmaxOutput(symbol)" args '["out_grad", "smooth_alpha", "normalization", "preserve_shape", "multi_output", "use_ignore", "ignore_label", "grad_scale"]
                 ,HasReqArg "_SoftmaxOutput(symbol)" args '["data", "label"])
        => String -> ArgsHMap "_SoftmaxOutput(symbol)" args -> IO SymbolHandle
softmaxoutput = S._SoftmaxOutput

batchnorm :: (HasOptArg "_BatchNorm(symbol)" args '["eps", "momentum", "fix_gamma", "use_global_stats", "output_mean_var", "axis", "cudnn_off"]
             ,HasReqArg "_BatchNorm(symbol)" args '["data"])
          => String -> ArgsHMap "_BatchNorm(symbol)" args -> IO SymbolHandle
batchnorm name args = do
    gamma    <- variable (name ++ "-gamma")
    beta     <- variable (name ++ "-beta")
    mov_mean <- variable (name ++ "-moving-mean")
    mov_var  <- variable (name ++ "-moving-var")
    S._BatchNorm name (#gamma := gamma .& #beta := beta .& #moving_mean := mov_mean .& #moving_var := mov_var .& args)

cast :: (HasReqArg "_Cast(symbol)" args '["data", "dtype"])
    => String -> ArgsHMap "_Cast(symbol)" args -> IO SymbolHandle
cast name args = S._Cast name args

plus :: (HasReqArg "elemwise_add(symbol)" args '["lhs", "rhs"])
    => String -> ArgsHMap "elemwise_add(symbol)" args -> IO SymbolHandle
plus = S.elemwise_add

flatten :: (HasReqArg "_Flatten(symbol)" args '["data"])
    => String -> ArgsHMap "_Flatten(symbol)" args -> IO SymbolHandle
flatten = S._Flatten

identity :: (HasReqArg "_copy(symbol)" args '["data"])
    => String -> ArgsHMap "_copy(symbol)" args -> IO SymbolHandle
identity = S._copy