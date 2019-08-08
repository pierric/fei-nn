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
  dropout,
  reshape,
) where

import MXNet.Base
import qualified MXNet.Base.Operators.Symbol as S

variable :: String -> IO SymbolHandle
variable = mxSymbolCreateVariable

convolution :: (HasArgs "_Convolution(symbol)" args '["kernel", "num_filter", "data", "stride", "dilate", "pad", "num_group", "workspace", "layout", "cudnn_tune", "cudnn_off", "no_bias"]
               ,WithoutArgs "_Convolution(symbol)" args '["bias", "weight"])
            => String -> ArgsHMap "_Convolution(symbol)" args -> IO SymbolHandle
convolution name args = do
    b <- variable (name ++ "_bias")
    w <- variable (name ++ "_weight")
    if args !? #no_bias == Just True 
      then
        S._Convolution name (#weight := w .& args)
      else
        S._Convolution name (#bias := b .& #weight := w .& args)

fullyConnected :: (HasArgs "_FullyConnected(symbol)" args '["flatten", "no_bias", "data", "num_hidden"]
                  ,WithoutArgs "_FullyConnected(symbol)" args '["bias", "weight"])
              => String -> ArgsHMap "_FullyConnected(symbol)" args -> IO SymbolHandle
fullyConnected name args = do
  b <- variable (name ++ "_bias")
  w <- variable (name ++ "_weight")
  if args !? #no_bias == Just True 
    then
      S._FullyConnected name (#weight := w .& args)
    else
      S._FullyConnected name (#bias := b .& #weight := w .& args)

-- 1.0.0 pooling :: HasArgs "_Pooling(symbol)" args '["data", "kernel", "pool_type", "stride", "pad", "pooling_convention", "global_pool", "cudnn_off"]
pooling :: HasArgs "_Pooling(symbol)" args '["data", "kernel", "pool_type", "stride", "pad", "pooling_convention", "global_pool", "cudnn_off", "p_value", "count_include_pad"]
        => String -> ArgsHMap "_Pooling(symbol)" args -> IO SymbolHandle
pooling = S._Pooling

activation :: HasArgs "_Activation(symbol)" args '["data", "act_type"]
        => String -> ArgsHMap "_Activation(symbol)" args -> IO SymbolHandle
activation = S._Activation

softmaxoutput :: HasArgs "_SoftmaxOutput(symbol)" args '["data", "label", "out_grad", "smooth_alpha", "normalization", "preserve_shape", "multi_output", "use_ignore", "ignore_label", "grad_scale"]
        => String -> ArgsHMap "_SoftmaxOutput(symbol)" args -> IO SymbolHandle
softmaxoutput = S._SoftmaxOutput

batchnorm :: HasArgs "_BatchNorm(symbol)" args '["data", "eps", "momentum", "fix_gamma", "use_global_stats", "output_mean_var", "axis", "cudnn_off"]
          => String -> ArgsHMap "_BatchNorm(symbol)" args -> IO SymbolHandle
batchnorm name args = do
    gamma    <- variable (name ++ "_gamma")
    beta     <- variable (name ++ "_beta")
    mov_mean <- variable (name ++ "_moving_mean")
    mov_var  <- variable (name ++ "_moving_var")
    S._BatchNorm name (#gamma := gamma .& #beta := beta .& #moving_mean := mov_mean .& #moving_var := mov_var .& args)

cast :: HasArgs "_Cast(symbol)" args '["data", "dtype"]
    => String -> ArgsHMap "_Cast(symbol)" args -> IO SymbolHandle
cast name args = S._Cast name args

plus :: HasArgs "elemwise_add(symbol)" args '["lhs", "rhs"]
    => String -> ArgsHMap "elemwise_add(symbol)" args -> IO SymbolHandle
plus = S.elemwise_add

flatten :: HasArgs "_Flatten(symbol)" args '["data"]
    => String -> ArgsHMap "_Flatten(symbol)" args -> IO SymbolHandle
flatten = S._Flatten

identity :: HasArgs "_copy(symbol)" args '["data"]
    => String -> ArgsHMap "_copy(symbol)" args -> IO SymbolHandle
identity = S._copy

dropout :: HasArgs "_Dropout(symbol)" args '["data", "mode", "p", "axes"] 
    => String -> ArgsHMap "_Dropout(symbol)" args -> IO SymbolHandle
dropout = S._Dropout

reshape :: (HasArgs "_Reshape(symbol)" args '["data", "shape", "reverse"]
           ,WithoutArgs "_Reshape(symbol)" args '["target_shape", "keep_highest"])
    => String -> ArgsHMap "_Reshape(symbol)" args -> IO SymbolHandle
reshape = S._Reshape

