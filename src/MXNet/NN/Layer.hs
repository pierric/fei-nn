{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE QuasiQuotes #-}

module MXNet.NN.Layer (
  variable,
  convolution,
  fullyConnected,
  PoolingMethod(..),
  pooling,
  ActivationType(..),
  activation,
  softmaxoutput,
  batchnorm,
  AsDType(..),
  cast,
  plus,
  S.flatten,
  S.identity,
) where

import MXNet.Core.Types.Internal
import MXNet.Core.Base.HMap
import qualified MXNet.Core.Base.Internal.TH.Symbol as S
import qualified MXNet.Core.Base.Internal as I
import MXNet.NN.Utils

variable :: String -> IO SymbolHandle
variable = I.checked . I.mxSymbolCreateVariable

convolution :: (MatchKVList kvs '["stride"     ':= String,
                                  "dilate"     ':= String,
                                  "pad"        ':= String,
                                  "num_group"  ':= Int, 
                                  "workspace"  ':= Int, 
                                  "no_bias"    ':= Bool,
                                  "cudnn_tune" ':= String, 
                                  "cudnn_off"  ':= Bool, 
                                  "layout"     ':= String]
               ,ShowKV kvs, QueryKV kvs)
            => String -> SymbolHandle -> [Int] -> Int -> HMap kvs -> IO SymbolHandle
convolution name dat kernel_shape num_filter args = do
    w <- variable (name ++ "-weight")
    if query @"no_bias" args == Just True
      then 
        S.convolution name dat w Nothing (formatShape kernel_shape) num_filter args
      else do
        b <- variable (name ++ "-bias")
        S.convolution name dat w (Just b) (formatShape kernel_shape) num_filter args

fullyConnected :: (MatchKVList kvs '["no_bias" ':= Bool, 
                                     "flatten" ':= Bool]
                  ,ShowKV kvs, QueryKV kvs) 
               => String -> SymbolHandle -> Int -> HMap kvs -> IO SymbolHandle
fullyConnected name dat num_neuron args = do
    w <- variable (name ++ "-weight")
    if query @"no_bias" args == Just True
      then 
        S.fullyconnected name dat w Nothing num_neuron args
      else do
        b <- variable (name ++ "-bias")
        S.fullyconnected name dat w (Just b) num_neuron args

data PoolingMethod = PoolingMax | PoolingAvg | PoolingSum

poolingMethodToStr :: PoolingMethod -> String
poolingMethodToStr PoolingMax = "max"
poolingMethodToStr PoolingAvg = "avg"
poolingMethodToStr PoolingSum = "sum"

pooling :: (MatchKVList kvs '["global_pool" ':= Bool,
                              "cudnn_off" ':= Bool,
                              "pooling_convention" ':= String,
                              "stride" ':= String,
                              "pad" ':= String]
           ,ShowKV kvs)
        => String -> SymbolHandle -> [Int] -> PoolingMethod -> HMap kvs -> IO SymbolHandle
pooling name input shape method args = S.pooling name input (formatShape shape) (poolingMethodToStr method) args

data ActivationType = Relu | Sigmoid | Tanh | SoftRelu

activationTypeToStr :: ActivationType -> String
activationTypeToStr Relu = "relu"
activationTypeToStr Sigmoid = "sigmoid"
activationTypeToStr Tanh = "tanh"
activationTypeToStr SoftRelu = "softrelu"

activation :: String -> SymbolHandle -> ActivationType -> IO SymbolHandle
activation name input typ = S.activation name input (activationTypeToStr typ)

softmaxoutput :: (MatchKVList kvs '["grad_scale" ':= Float, 
                                    "ignore_label" ':= Float,
                                    "multi_output" ':= Bool, 
                                    "use_ignore" ':= Bool,
                                    "preserve_shape" ':= Bool, 
                                    "normalization" ':= String,
                                    "out_grad" ':= Bool, 
                                    "smooth_alpha" ':= Float],
                  ShowKV kvs)
               => String -> SymbolHandle -> SymbolHandle -> HMap kvs -> IO SymbolHandle
softmaxoutput = S.softmaxoutput

batchnorm :: (MatchKVList kvs '["eps" ':= Double,
                                "momentum" ':= Float,
                                "fix_gamma" ':= Bool,
                                "use_global_stats" ':= Bool,
                                "output_mean_var" ':= Bool,
                                "axis" ':= Int,
                                "cudnn_off" ':= Bool]
             ,ShowKV kvs)
          => String -> SymbolHandle -> HMap kvs -> IO SymbolHandle
batchnorm name dat args = do
    gamma    <- variable (name ++ "-gamma")
    beta     <- variable (name ++ "-beta")
    mov_mean <- variable (name ++ "-moving-mean")
    mov_var  <- variable (name ++ "-moving-var")
    S.batchnorm name dat gamma beta mov_mean mov_var args

data AsDType = AsFloat16 | AsFloat32 | AsFloat64 | AsUInt8 | AsInt32
cast :: String -> SymbolHandle -> AsDType -> IO SymbolHandle
cast name dat dtyp = S.cast name dat typ
  where
    typ = case dtyp of 
            AsFloat16 -> "float16"
            AsFloat32 -> "float32"
            AsFloat64 -> "float64" 
            AsUInt8   -> "uint8"
            AsInt32   -> "int32"

plus :: String -> SymbolHandle -> SymbolHandle -> IO SymbolHandle
plus = S._Plus