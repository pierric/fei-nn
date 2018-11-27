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
import qualified Data.GraphViz.Types.Monadic as GVM
import qualified Data.GraphViz.Types.Generalised as GVM

import MXNet.Base

-- The program `dot` must be found in the PATH.

dotPlot :: DType a => Symbol a -> GV.GraphvizOutput -> FilePath -> IO ()
dotPlot sym output filepath = do
    gr <- dotGraph sym
    _  <- GV.addExtension (GV.runGraphvizCommand GV.Dot gr) output filepath
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
dotGraph (Symbol sym) = do
    js <- mxSymbolSaveToJSON sym
    auxnodes <- mxSymbolListAuxiliaryStates sym
    case eitherDecode $ pack js of
      Left _ -> throwM CannotDecodeJSONofSymbol
      Right (JSGraph nodes) -> return $ GVM.digraph (GV.Num $ GV.Int 0) $ do
                                let nodesWithIdx = (zip [0..] nodes)
                                    blacklist = map fst $ 
                                                filter (\(_, node) -> elem (_node_name node) auxnodes || 
                                                                      _like "-weight" node || _like "-bias"  node || 
                                                                      _like "-beta"   node || _like "-gamma" node) 
                                                       nodesWithIdx
                                forM_ nodesWithIdx (mkNode_ blacklist)
                                forM_ nodesWithIdx (mkEdge_ blacklist)
  where
    mkNode_ blacklist (nodeid, JSNode{..}) = case _node_op of 
        "null" -> 
            when (not $ elem nodeid blacklist) $ 
            mkNode nodeid (#label := _node_name .& #shape := GV.Ellipse .& #fillcolor := colors !! 0 .& Nil)
        "Convolution" -> do
            let attr = fromJust $ _node_attrs
                krnl = formatTuple (fromJust $ M.lookup "kernel" attr)
                strd = formatTuple (fromMaybe "1" $ M.lookup "stride" attr)
                nflt = fromJust $ M.lookup "num_filter" attr
                lbl = printf "Convolution\n%s/%s, %s" krnl strd nflt
            mkNode nodeid (#label := lbl .& #fillcolor := colors !! 1 .& Nil)
        "FullyConnected" -> do
            let attr = fromJust $ _node_attrs
                hddn = fromJust $ M.lookup "num_hidden" attr
                lbl = printf "FullyConnected\n%s" hddn
            mkNode nodeid (#label := lbl .& #fillcolor := colors !! 1 .& Nil)
        "BatchNorm" ->
            mkNode nodeid (#label := "batchNorm" .& #fillcolor := colors !! 3 .& Nil)
        "Activation" -> do
            let attr = fromJust $ _node_attrs
                actt = fromJust $ M.lookup "act_type" attr
                lbl = printf "Activation\n%s" actt
            mkNode nodeid (#label := lbl .& #fillcolor := colors !! 2 .& Nil)
        "LeakyReLU" -> do
            let attr = fromJust $ _node_attrs
                actt = fromJust $ M.lookup "act_type" attr
                lbl = printf "LeakyReLU\n%s" actt
            mkNode nodeid (#label := lbl .& #fillcolor := colors !! 2 .& Nil)
        "Pooling" -> do
            let attr = fromJust $ _node_attrs
                poot = fromJust $ M.lookup "pool_type" attr
                krnl = formatTuple (fromJust $ M.lookup "kernel" attr)
                strd = formatTuple (fromMaybe "1" $ M.lookup "stride" attr)
                lbl = printf "Pooling\n%s, %s/%s" poot krnl strd
            mkNode nodeid (#label := lbl .& #fillcolor := colors !! 4 .& Nil)
        "Concat" -> 
            mkNode nodeid (#label := "Concat" .& #fillcolor := colors !! 5 .& Nil)
        "Flatten" ->
            mkNode nodeid (#label := "Flatten" .& #fillcolor := colors !! 5 .& Nil)
        "Reshape" ->
            mkNode nodeid (#label := "Reshape" .& #fillcolor := colors !! 5 .& Nil)
        "Softmax" ->
            mkNode nodeid (#label := "Softmax" .& #fillcolor := colors !! 6 .& Nil)
        "Custom" -> do
            let attr = fromJust $ _node_attrs
                lbl = fromJust $ M.lookup "op_type" attr
            mkNode nodeid (#label := lbl .& #fillcolor := colors !! 7 .& Nil)
        _ ->
            mkNode nodeid (#label := _node_name .& #fillcolor := colors !! 7 .& Nil)

    mkEdge_ blacklist (tid, tnode) = do
        let op = _node_op tnode
            -- name = _node_name tnode
        case op of 
            "null" -> return ()
            _ -> forM_ (_node_inputs tnode) $ \(sid:_) -> do
                   when (not $ elem sid blacklist) $ 
                     GVM.edge tid sid [GV.Dir GV.Back, GV.ArrowTail GV.vee]

    colors = catMaybes $ map color ["#8dd3c7", "#fb8072", "#ffffb3", 
                                    "#bebada", "#80b1d3", "#fdb462", 
                                    "#b3de69", "#fccde5"]

    _like sfx node = T.isSuffixOf sfx (T.pack $ _node_name node)

type instance ParameterList "graphviz_node" = 
    '[ '("label",     'AttrOpt String),
       '("shape",     'AttrOpt GV.Shape),
       '("fixedsize", 'AttrOpt Bool),
       '("fillcolor", 'AttrOpt GV.Color), 
       '("width",     'AttrOpt Double), 
       '("height",    'AttrOpt Double), 
       '("style",     'AttrOpt GV.Style) ]

mkNode :: (Fullfilled "graphviz_node" args)
      => Int -> ArgsHMap "graphviz_node" args -> GVM.DotM Int ()
mkNode nodeid args = GVM.node nodeid attrs
  where
    shp = GV.Shape  $ fromMaybe GV.BoxShape $ args !? #shape
    fxs = GV.FixedSize $ if fromMaybe True (args !? #fixedsize)
                         then GV.SetNodeSize 
                         else GV.GrowAsNeeded
    wdt = GV.Width  $ fromMaybe 1.3         $ args !? #width
    hgt = GV.Height $ fromMaybe 0.8034      $ args !? #height
    sty = GV.style  $ fromMaybe GV.filled   $ args !? #style
    mfc = maybeToList $ GV.FillColor . GV.toColorList . (:[]) <$> (args !? #fillcolor)
    lbl = maybeToList $ GV.textLabel . T.pack <$> (args !? #label)
    attrs = [shp, fxs, wdt, hgt, sty] ++  lbl ++ mfc

color :: String -> Maybe GV.Color
color ['#',r1,r2,g1,g2,b1,b2] = do
    let dec = listToMaybe . map fst . readHex
    r <- dec [r1,r2]
    g <- dec [g1,g2]
    b <- dec [b1,b2]
    return $ GV.RGB r g b
color _ = Nothing

formatTuple :: String -> String
formatTuple str 
    | [((a,b),"")] <- (reads :: ReadS (Int,Int)) str = printf "%dx%d" a b
    | [([a,b],"")] <- (reads :: ReadS [Int]) str     = printf "%dx%d" a b
    | otherwise = str

data Exc = CannotDecodeJSONofSymbol
    deriving (Show, Typeable)
instance Exception Exc