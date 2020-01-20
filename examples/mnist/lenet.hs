{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE OverloadedLabels #-}
{-# LANGUAGE TypeApplications #-}
module Main where

import Control.Monad (forM_, void)
import qualified Data.Vector.Storable as SV
import Control.Monad.IO.Class
import qualified Data.HashMap.Strict as M

import MXNet.Base (
    Symbol(..), NDArray(..), ArgOf(..), HMap(..), (.&), 
    mxListAllOpNames, contextCPU, contextGPU0, toVector)
import qualified MXNet.Base.Operators.NDArray as A
import MXNet.NN

import DatasetVector

-- # first conv
-- conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
-- tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
-- pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
-- # second conv
-- conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
-- tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
-- pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
-- # first fullc
-- flatten = mx.symbol.Flatten(data=pool2)
-- fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
-- tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
-- # second fullc
-- fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=num_classes)
-- # loss
-- lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

model :: IO SymbolF
model = do
    x  <- variable "x"
    y  <- variable "y"

    v1 <- convolution "conv1"  (#data := x  .& #kernel := [5,5] .& #num_filter := 20 .& Nil)
    a1 <- activation "conv1-a" (#data := v1 .& #act_type := #tanh .& Nil)
    p1 <- pooling "conv1-p"    (#data := a1 .& #kernel := [2,2] .& #pool_type := #max .& Nil)

    v2 <- convolution "conv2"  (#data := p1 .& #kernel := [5,5] .& #num_filter := 50 .& Nil)
    a2 <- activation "conv2-a" (#data := v2 .& #act_type := #tanh .& Nil)
    p2 <- pooling "conv2-p"    (#data := a2 .& #kernel := [2,2] .& #pool_type := #max .& Nil)

    fl <- flatten "flatten"    (#data := p2 .& Nil)

    v3 <- fullyConnected "fc1" (#data := fl .& #num_hidden := 500 .& Nil)
    a3 <- activation "fc1-a"   (#data := v3 .& #act_type := #tanh .& Nil)

    v4 <- fullyConnected "fc2" (#data := a3 .& #num_hidden := 10  .& Nil)
    a4 <- softmaxoutput "softmax" (#data := v4 .& #label := y .& Nil)
    return $ Symbol a4

range :: Int -> [Int]
range = enumFromTo 1

default_initializer :: Initializer Float
default_initializer name shp@[_]   = zeros name shp
default_initializer name shp@[_,_] = xavier 2.0 XavierGaussian XavierIn name shp
default_initializer name shp = normal 0.1 name shp

main :: IO ()
main = do
    -- call mxListAllOpNames can ensure the MXNet itself is properly initialized
    -- i.e. MXNet operators are registered in the NNVM
    _      <- mxListAllOpNames
    symbol <- model

    trainingData <- loadTrainingData
    testingData  <- loadTestingData

    optimizer <- makeOptimizer SGD'Mom (Const 0.0002) (#momentum := 0.9 .& #wd := 0.0001 .& Nil)

    mod <- initialize @"lenet" symbol $ Config {
               _cfg_data = M.singleton "x" [1,28,28],
               _cfg_label = ["y"],
               _cfg_initializers = M.empty,
               _cfg_default_initializer = default_initializer,
               _cfg_context = contextGPU0
           }

    train mod $ do
        liftIO $ putStrLn $ "[Train] "
        forM_ (range 1) $ \ind -> do
            liftIO $ putStrLn $ "iteration " ++ show ind
            -- metric <- newMetric "train" (CrossEntropy "y")
            void $ forEachD_ni (liftD trainingData) $ \((t,i), (x, y)) -> do
                -- eval <- format metric
                liftIO $ printInLine $ show i ++ "/" ++ show t -- ++ " " ++ eval
                fit (M.fromList [("x", x), ("y", y)])
                update optimizer M.empty
            liftIO $ putStrLn ""

        liftIO $ putStrLn $ "[Test] "
        result <- forEachD_ni (liftD testingData) $ \((t,i), (x, y)) -> do
            liftIO $ printInLine $ show i ++ "/" ++ show t
            [y'] <-  forwardOnly (M.fromList [("x", x), ("y", y)])
            liftIO $ do
                y_true <- toVector y
                y_pred <- argmax y' >>= toVector
                return (y_true, y_pred)
        liftIO $ putStrLn ""

        let (ls,ps) = unzip result
            ls_unbatched = mconcat ls
            ps_unbatched = mconcat ps
            total_test_items = SV.length ls_unbatched
            correct = SV.length $ SV.filter id $ SV.zipWith (==) ls_unbatched ps_unbatched
        liftIO $ putStrLn $ "Accuracy: " ++ show correct ++ "/" ++ show total_test_items

  where
    argmax :: ArrayF -> IO ArrayF
    argmax (NDArray ys) = NDArray . head <$> A.argmax (#data := ys .& #axis := Just 1 .& Nil)
