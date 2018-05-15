{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
module Main where

import MXNet.Core.Base (DType, contextCPU, contextGPU, mxListAllOpNames)
import MXNet.Core.Base.HMap
import qualified MXNet.Core.Base.NDArray as A
import qualified MXNet.Core.Base.Internal.TH.NDArray as A
import qualified MXNet.Core.Base.Symbol as S
import qualified Data.HashMap.Strict as M
import Control.Monad (forM_, void)
import qualified Data.Vector.Storable as SV
import Control.Monad.IO.Class
import Control.Monad.Trans.Resource
import System.IO (hFlush, stdout)
import MXNet.NN
import MXNet.NN.Layer
import MXNet.NN.EvalMetric
import MXNet.NN.Initializer
import MXNet.NN.DataIter.Class

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

neural :: IO SymbolF
neural = do
    x  <- variable "x"
    y  <- variable "y"

    v1 <- convolution "conv1" x [5,5] 20 nil
    a1 <- activation "conv1-a" v1 Tanh
    p1 <- pooling "conv1-p" a1 [2,2] PoolingMax nil

    v2 <- convolution "conv2" p1 [5,5] 50 nil
    a2 <- activation "conv2-a" v2 Tanh
    p2 <- pooling "conv2-p" a2 [2,2] PoolingMax nil

    fl <- flatten "flatten" p2

    v3 <- fullyConnected "fc1" fl 500 nil
    a3 <- activation "fc1-a" v3 Tanh

    v4 <- fullyConnected "fc2" a3 10 nil
    a4 <- softmaxoutput "softmax" v4 y nil
    return $ S.Symbol a4

range :: Int -> [Int]
range = enumFromTo 1

default_initializer :: Initializer Float
default_initializer shp@[_]   = zeros shp
default_initializer shp@[_,_] = xavier 2.0 XavierGaussian XavierIn shp
default_initializer shp = normal 0.1 shp

main :: IO ()
main = do
    -- call mxListAllOpNames can ensure the MXNet itself is properly initialized
    -- i.e. MXNet operators are registered in the NNVM
    _    <- mxListAllOpNames
    net  <- neural
    sess <- initialize net $ Config { 
                _cfg_placeholders = M.singleton "x" [1,1,28,28],
                _cfg_initializers = M.empty,
                _cfg_default_initializer = default_initializer,
                _cfg_context = contextCPU
            }
    optimizer <- makeOptimizer (SGD'Mom 0.0002) (add @"momentum" 0.9 $ add @"wd" 0.0001 nil)

    runResourceT $ train sess $ do 

        trainingData <- loadTrainingData
        testingData  <- loadTestingData

        liftIO $ putStrLn $ "[Train] "
        forM_ (range 5) $ \ind -> do
            liftIO $ putStrLn $ "iteration " ++ show ind
            metric <- newMetric CrossEntropy "CrossEntropy" ["y"]
            void $ forEachD_ni trainingData $ \((t,i), (x, y)) -> do
                liftIO $ do
                   eval <- formatMetric metric
                   putStr $ "\r\ESC[K" ++ show i ++ "/" ++ show t ++ " " ++ eval
                   hFlush stdout
                fitAndEval optimizer net (M.fromList [("x", x), ("y", y)]) metric
            liftIO $ putStrLn ""
        
        liftIO $ putStrLn $ "[Test] "
        result <- forEachD_ni testingData $ \((t,i), (x, y)) -> do 
            liftIO $ do 
                putStr $ "\r\ESC[K" ++ show i ++ "/" ++ show t
                hFlush stdout
            [y'] <- forwardOnly net (M.fromList [("x", Just x), ("y", Nothing)])
            ind1 <- liftIO $ A.items y
            ind2 <- liftIO $ argmax y' >>= A.items
            return (ind1, ind2)
        liftIO $ putStr "\r\ESC[K"

        let (ls,ps) = unzip result
            ls_unbatched = mconcat ls
            ps_unbatched = mconcat ps
            total_test_items = SV.length ls_unbatched
            correct = SV.length $ SV.filter id $ SV.zipWith (==) ls_unbatched ps_unbatched
        liftIO $ putStrLn $ "Accuracy: " ++ show correct ++ "/" ++ show total_test_items
  
  where
    argmax :: ArrayF -> IO ArrayF
    argmax ys = A.NDArray <$> A.argmax (A.getHandle ys) (add @"axis" 1 nil)
