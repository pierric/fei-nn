{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}

module MXNet.NN.Layer where

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
               ,ShowKV kvs)
            => String -> SymbolHandle -> [Int] -> Int -> HMap kvs -> IO SymbolHandle
convolution name dat kernel_shape num_filter args = do
    w <- variable (name ++ "-w")
    b <- variable (name ++ "-b")
    S.convolution name dat w b (formatShape kernel_shape) num_filter args

fullyConnected :: (MatchKVList kvs '["no_bias" ':= Bool, 
                                     "flatten" ':= Bool]
                  ,ShowKV kvs) 
               => String -> SymbolHandle -> Int -> HMap kvs -> IO SymbolHandle
fullyConnected name dat num_neuron args = do
    w <- variable (name ++ "-w")
    b <- variable (name ++ "-b")
    S.fullyconnected name dat w b num_neuron args

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

flatten :: String -> SymbolHandle -> IO SymbolHandle
flatten = S.flatten

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
