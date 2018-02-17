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

convolution :: (MatchKVList kvs '["stride" ':= String, "dilate" ':= String, "pad" ':= String,
                                  "num_group" ':= Int, "workspace" ':= Int, "no_bias" ':= Bool,
                                  "cudnn_tune" ':= String, "cudnn_off" ':= Bool, "layout" ':= String],
                ShowKV kvs)
            => String -> SymbolHandle -> [Int] -> Int -> HMap kvs -> IO SymbolHandle
convolution name dat kernel_shape num_filter args = do
    w <- variable (name ++ "-w")
    b <- variable (name ++ "-b")
    S.convolution name dat w b (formatShape kernel_shape) num_filter args

fullyConnected :: (MatchKVList kvs '["no_bias" ':= Bool, "flatten" ':= Bool], ShowKV kvs) 
               => String -> SymbolHandle -> Int -> HMap kvs -> IO SymbolHandle
fullyConnected name dat num_neuron args = do
    w <- variable (name ++ "-w")
    b <- variable (name ++ "-b")
    S.fullyconnected name dat w b num_neuron args