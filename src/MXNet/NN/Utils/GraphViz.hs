{-# LANGUAGE OverloadedStrings #-}
module MXNet.NN.Utils.GraphViz where

import MXNet.Core.Base.Internal (mxSymbolSaveToJSON, checked)
import MXNet.Core.Base.Symbol (Symbol, getHandle)
import Data.Aeson
import Data.Aeson.Types
import Data.ByteString.Lazy.Char8 (pack)
import qualified Data.Map as M

data JSNode = JSNode {
    _node_op :: String,
    _node_name :: String,
    _node_attrs :: Maybe (M.Map String String),
} deriving (Show)

instance FromJSON Node where
    parseJSON (Object v) = JSNode <$> v .:  "op"
                                  <*> v .:  "name"
                                  <*> v .:? "attrs"
    parseJSON invalid    = typeMismatch "JSNode" invalid

data JSGraph = JSGraph {
    _symbol_nodes :: [Node],
} deriving (Show)

instance FromJSON JSGraph where
    parseJSON (Object v) = JSGraph <$> v .: "nodes"
    parseJSON invalid    = typeMismatch "JSGraph" invalid


-- plot_network
-- https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/visualization.py#L196
graph :: Symbol -> IO (Gr String String)
graph sym = do
    js <- checked $ mxSymbolSaveToJSON (getHandle sym)
    case eitherDecode $ pack js of
      Left err -> throwM CannotDecodeJSONofSymbol
      Right (JSGraph nodes) -> digraph (Str "plot") $ do
                                let nodesWithIdx = (zip [1..] nodes)
                                foldM mkNode nodesWithIdx
                                foldM mkEdge nodesWithIdx
  where
    mkNode (id, Node{}) = node id $ case _node_op of 
        "null" -> 
            [textLabel (_node_name node), shape Ellipse, fillColor (colors !! 0)]
        "Convolution" -> 
            let lbl = printf "Convolution\n%s/%s, %s" krnl strd nflt
            in [textLabel lbl, shape BoxShape, fillColor (colors !! 1)]
        "FullyConnected" ->
            let lbl = printf "FullyConnected\n%s" hddn
            in [textLabel lbl, shape BoxShape, fillColor (colors !! 1)]
        "BatchNorm" ->
            [textLabel "batchNorm", shape BoxShape, fillColor (colors !! 3)]
        "Activation" ->
            let lbl = printf "Activation\n%s" actt
            in [textLabel lbl, shape BoxShape, fillColor (colors !! 2)]
        "LeakyReLU" ->
            let lbl = printf "LeakyReLU\n%s" actt
            in [textLabel lbl, shape BoxShape, fillColor (colors !! 2)]
        "Pooling" ->
            let lbl = "Pooling\n%s, %s/%s"
            in [textLabel lbl, shape BoxShape, fillColor (colors !! 4)]
        "Concat"
    colors = ["#8dd3c7", "#fb8072", "#ffffb3", "#bebada", "#80b1d3", "#fdb462", "#b3de69", "#fccde5"]

data Exc = CannotDecodeJSONofSymbol
    deriving (Show, Typeable)
instance Exception Exc