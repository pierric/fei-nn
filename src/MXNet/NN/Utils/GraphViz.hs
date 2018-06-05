{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications, TypeOperators, DataKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
module MXNet.NN.Utils.GraphViz (
    dotPlot,
    dotGraph, 
    GV.GraphvizOutput(..)
) where

import MXNet.Core.Base.Internal (mxSymbolSaveToJSON, checked)
import MXNet.Core.Base.Symbol (Symbol, getHandle, listAuxiliaries)
import MXNet.Core.Base.HMap
import MXNet.Core.Base.DType (DType)
import Data.Aeson
import Data.Aeson.Types
import Data.ByteString.Lazy.Char8 (pack)
import qualified Data.Map as M
import Control.Exception.Base (Exception)
import Control.Monad.Catch(MonadThrow(..))
import Data.Typeable (Typeable)
import Data.Maybe
import Numeric (readHex)
import Text.Printf (printf)
import Control.Monad (forM_, when)
import qualified Data.Text.Lazy as T
import qualified Data.GraphViz as GV
import qualified Data.GraphViz.Attributes.Complete as GV
import qualified Data.GraphViz.Attributes.Colors as GV
import qualified Data.GraphViz.Types.Monadic as GVM
import qualified Data.GraphViz.Types.Generalised as GVM
import MXNet.NN.Utils.HMap (α)

-- The program `dot` must be found in the PATH.

dotPlot :: DType a => Symbol a -> GV.GraphvizOutput -> FilePath -> IO ()
dotPlot sym output filepath = do
    gr <- dotGraph sym
    GV.addExtension (GV.runGraphvizCommand GV.Dot gr) output filepath
    return ()

data JSNode = JSNode {
    _node_op :: String,
    _node_name :: String,
    _node_attrs :: Maybe (M.Map String String),
    _node_inputs :: [[Int]]
} deriving (Show)

instance FromJSON JSNode where
    parseJSON (Object v) = JSNode <$> v .:  "op"
                                  <*> v .:  "name"
                                  <*> v .:? "attrs"
                                  <*> v .:  "inputs"
    parseJSON invalid    = typeMismatch "JSNode" invalid

data JSGraph = JSGraph {
    _symbol_nodes :: [JSNode]
} deriving (Show)

instance FromJSON JSGraph where
    parseJSON (Object v) = JSGraph <$> v .: "nodes"
    parseJSON invalid    = typeMismatch "JSGraph" invalid

-- plot_network
-- https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/visualization.py#L196
dotGraph :: DType a => Symbol a -> IO (GVM.DotGraph Int)
dotGraph sym = do
    js <- checked $ mxSymbolSaveToJSON (getHandle sym)
    auxnodes <- listAuxiliaries sym
    case eitherDecode $ pack js of
      Left err -> throwM CannotDecodeJSONofSymbol
      Right (JSGraph nodes) -> return $ GVM.digraph (GV.Num $ GV.Int 0) $ do
                                let nodesWithIdx = (zip [0..] nodes)
                                    blacklist = map fst $ 
                                                filter (\(id, node) -> elem (_node_name node) auxnodes || 
                                                                       _like "-weight" node || _like "-bias"  node || 
                                                                       _like "-beta"   node || _like "-gamma" node) 
                                                       nodesWithIdx
                                forM_ nodesWithIdx (mkNode blacklist)
                                forM_ nodesWithIdx (mkEdge blacklist)
  where
    mkNode blacklist (id, JSNode{..}) = case _node_op of 
        "null" -> 
            when (not $ elem id blacklist) $ 
              node id [α| label := _node_name, shape := GV.Ellipse, fillcolor := colors !! 0 |]
        "Convolution" -> 
            let attr = fromJust $ _node_attrs
                krnl = fromJust $ M.lookup "kernel" attr
                strd = fromMaybe "1" $ M.lookup "stride" attr
                nflt = fromJust $ M.lookup "num_filter" attr
                lbl = printf "Convolution\n%s/%s, %s" krnl strd nflt
            in node id [α| label := lbl :: String, fillcolor := colors !! 1 |]
        "FullyConnected" ->
            let attr = fromJust $ _node_attrs
                hddn = fromJust $ M.lookup "num_hidden" attr
                lbl = printf "FullyConnected\n%s" hddn
            in node id [α| label := lbl :: String, fillcolor := colors !! 1 |]
        "BatchNorm" ->
            node id [α| label := "batchNorm" :: String, fillcolor := colors !! 3 |]
        "Activation" ->
            let attr = fromJust $ _node_attrs
                actt = fromJust $ M.lookup "act_type" attr
                lbl = printf "Activation\n%s" actt
            in node id [α| label := lbl :: String, fillcolor := colors !! 2 |]
        "LeakyReLU" ->
            let attr = fromJust $ _node_attrs
                actt = fromJust $ M.lookup "act_type" attr
                lbl = printf "LeakyReLU\n%s" actt
            in node id [α| label := lbl :: String, fillcolor := colors !! 2 |]
        "Pooling" ->
            let attr = fromJust $ _node_attrs
                poot = fromJust $ M.lookup "pool_type" attr
                krnl = fromJust $ M.lookup "kernel" attr
                strd = fromMaybe "1" $ M.lookup "stride" attr
                lbl = printf "Pooling\n%s, %s/%s" poot krnl strd
            in node id [α| label := lbl :: String, fillcolor := colors !! 4 |]
        "Concat" -> 
            node id [α| label := "Concat" :: String,  fillcolor := colors !! 5 |]
        "Flatten" ->
            node id [α| label := "Flatten" :: String, fillcolor := colors !! 5 |]
        "Reshape" ->
            node id [α| label := "Reshape" :: String, fillcolor := colors !! 5 |]
        "Softmax" ->
            node id  [α| label := "Softmax" :: String, fillcolor := colors !! 6 |]
        "Custom" ->
            let attr = fromJust $ _node_attrs
                lbl = fromJust $ M.lookup "op_type" attr
            in node id [α| label := lbl :: String, fillcolor := colors !! 7 |]
        _ ->
            node id [α| label := _node_name, fillcolor := colors !! 7 |]

    mkEdge blacklist (tid, tnode) = do
        let op = _node_op tnode
            name = _node_name tnode
        case op of 
            "null" -> return ()
            _ -> forM_ (_node_inputs tnode) $ \(sid:_) -> do
                   when (not $ elem sid blacklist) $ 
                     GVM.edge tid sid [GV.Dir GV.Back, GV.ArrowTail GV.vee]

    colors = catMaybes $ map color ["#8dd3c7", "#fb8072", "#ffffb3", 
                                    "#bebada", "#80b1d3", "#fdb462", 
                                    "#b3de69", "#fccde5"]

    _like sfx node = T.isSuffixOf sfx (T.pack $ _node_name node)

node :: (MatchKVList kvs '["label"     ':= String, 
                           "shape"     ':= GV.Shape, 
                           "fixedsize" ':= Bool, 
                           "fillcolor" ':= GV.Color, 
                           "width"     ':= Float, 
                           "height"    ':= Float, 
                           "style"     ':= GV.Style],
         QueryKV kvs)
      => Int -> HMap kvs -> GVM.DotM Int ()
node id args = GVM.node id attrs
  where
    shp = GV.Shape  $ fromMaybe GV.BoxShape $ query @"shape"     args
    fxs = GV.FixedSize $ if fromMaybe True (query @"fixedsize" args)
                         then GV.SetNodeSize 
                         else GV.GrowAsNeeded
    wdt = GV.Width  $ fromMaybe 1.3         $ query @"width"     args
    hgt = GV.Height $ fromMaybe 0.8034      $ query @"height"    args
    sty = GV.style  $ fromMaybe GV.filled   $ query @"style"     args
    mfc = maybeToList $ GV.FillColor . GV.toColorList . (:[]) <$> query @"fillcolor" args
    lbl = maybeToList $ GV.textLabel . T.pack <$> query @"label" args
    attrs = [shp, fxs, wdt, hgt, sty] ++  lbl ++ mfc

color :: String -> Maybe GV.Color
color ['#',r1,r2,g1,g2,b1,b2] = do
    let dec = listToMaybe . map fst . readHex
    r <- dec [r1,r2]
    g <- dec [g1,g2]
    b <- dec [b1,b2]
    return $ GV.RGB r g b
color _ = Nothing

data Exc = CannotDecodeJSONofSymbol
    deriving (Show, Typeable)
instance Exception Exc