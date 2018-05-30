{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE QuasiQuotes #-}

module MXNet.NN.Layer where

import Control.Monad (when, void)
import MXNet.Core.Types.Internal
import MXNet.Core.Base.HMap
import qualified MXNet.Core.Base.Internal.TH.Symbol as S
import qualified MXNet.Core.Base.Internal as I
import MXNet.NN.Utils
import MXNet.NN.Utils.HMap

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
    w <- variable (name ++ "-w")
    if query @"no_bias" args == Just True
      then 
        S.convolution name dat w Nothing (formatShape kernel_shape) num_filter args
      else do
        b <- variable (name ++ "-b")
        S.convolution name dat w (Just b) (formatShape kernel_shape) num_filter args

fullyConnected :: (MatchKVList kvs '["no_bias" ':= Bool, 
                                     "flatten" ':= Bool]
                  ,ShowKV kvs, QueryKV kvs) 
               => String -> SymbolHandle -> Int -> HMap kvs -> IO SymbolHandle
fullyConnected name dat num_neuron args = do
    w <- variable (name ++ "-w")
    if query @"no_bias" args == Just True
      then 
        S.fullyconnected name dat w Nothing num_neuron args
      else do
        b <- variable (name ++ "-b")
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

type ResidualOptArgs = '["bottle_neck" ':= Bool, "bn_mom" ':= Float, "workspace" ':= Int, "memonger" ':= Bool]
residual :: (MatchKVList kvs ResidualOptArgs, ShowKV kvs) 
         => String -> SymbolHandle -> Int -> [Int] -> Bool -> HMap kvs -> IO SymbolHandle
residual name dat num_filter stride dim_match oargs = do
    let args = mergeTo oargs (True .+. 0.9 .+. 256 .+. False .+. nil) :: HMap ResidualOptArgs
        workspace = get @"workspace" args :: Int
        bn_mom = get @"bn_mom" args :: Float
        eps = 2e-5 :: Double
    if get @"bottle_neck" args 
      then do
        bn1 <- batchnorm (name ++ "-bn1") dat 
                    [hm| eps       := eps
                       , momentum  := bn_mom
                       , fix_gamma := False |]
        act1 <- activation (name ++ "-relu1") bn1 Relu
        conv1 <- convolution (name ++ "-conv1") act1 [1,1] (num_filter `div` 4) 
                    [hm| stride    := "[1,1]"
                       , pad       := "[0,0]"
                       , workspace := workspace
                       , no_bias   := True |]
        bn2 <- batchnorm (name ++ "-bn2") conv1 
                    [hm| eps       := eps
                       , momentum  := bn_mom
                       , fix_gamma := False |]
        act2 <- activation (name ++ "-relu2") bn2 Relu
        conv2 <- convolution (name ++ "-conv2") act2 [3,3] (num_filter `div` 4) 
                    [hm| stride    := (show stride)
                       , pad       := "[1,1]"
                       , workspace := workspace
                       , no_bias   := True |]
        bn3 <- batchnorm (name ++ "-bn3") conv2
                    [hm| eps       := eps
                       , momentum  := bn_mom
                       , fix_gamma := False |]
        act3 <- activation (name ++ "-relu3") bn3 Relu
        conv3 <- convolution (name ++ "-conv3") act3 [1,1] num_filter 
                    [hm| stride    := "[1,1]"
                       , pad       := "[0,0]"
                       , workspace := workspace
                       , no_bias   := True |]
        shortcut <- if dim_match
                    then return dat
                    else convolution (name ++ "-sc") act1 [1,1] num_filter 
                            [hm| stride    := (show stride)
                               , workspace := workspace
                               , no_bias   := True |]
        when (get @"memonger" args) $ void $ I.mxSymbolSetAttr shortcut "mirror_stage" "true"
        S._Plus name conv3 shortcut
      else do
        bn1 <- batchnorm (name ++ "-bn1") dat 
                    [hm| eps       := eps
                       , momentum  := bn_mom
                       , fix_gamma := False |]
        act1 <- activation (name ++ "-relu1") bn1 Relu
        conv1 <- convolution (name ++ "-conv1") act1 [3,3] num_filter 
                    [hm| stride    := (show stride)
                       , pad       :="[1,1]"
                       , workspace := workspace
                       , no_bias   := True |]
        bn2 <- batchnorm (name ++ "-bn2") conv1 
                    [hm| eps       := eps
                       , momentum  := bn_mom
                       , fix_gamma := False |]
        act2 <- activation (name ++ "-relu2") bn2 Relu
        conv2 <- convolution (name ++ "-conv2") act2 [3,3] num_filter 
                    [hm| stride    := "[1,1]"
                       , pad       := "[1,1]"
                       , workspace := workspace
                       , no_bias   := True |]
        shortcut <- if dim_match
                    then return dat
                    else convolution (name ++ "-sc") act1 [1,1] num_filter 
                            [hm| stride    := (show stride)
                               , workspace := workspace
                               , no_bias   := True |]
        when (get @"memonger" args) $ void $ I.mxSymbolSetAttr shortcut "mirror_stage" "true"
        S._Plus name conv2 shortcut

data DType = AsFloat16 | AsFloat32 | AsFloat64 | AsUInt8 | AsInt32
cast :: String -> SymbolHandle -> DType -> IO SymbolHandle
cast name dat dtyp = S.cast name dat typ
  where
    typ = case dtyp of 
            AsFloat16 -> "float16"
            AsFloat32 -> "float32"
            AsFloat64 -> "float64" 
            AsUInt8   -> "uint8"
            AsInt32   -> "int32"