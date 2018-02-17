{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
module Main where

import MXNet.Core.Base hiding (variable, convolution, fullyConnected)
import qualified MXNet.Core.Base.NDArray as A
import qualified MXNet.Core.Base.Internal.TH.NDArray as A
import qualified MXNet.Core.Base.Symbol as S
import qualified MXNet.Core.Base.Internal.TH.Symbol as S
import qualified Data.HashMap.Strict as M
import Control.Monad (forM_, void)
import qualified Streaming.Prelude as SR
import qualified Data.Vector.Storable as SV
import Control.Monad.IO.Class
import Control.Monad.Trans.Resource
import System.IO (hFlush, stdout)
import MXNet.NN
import MXNet.NN.Utils
import MXNet.NN.Layer
import Dataset

-- # first conv
-- conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
-- tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
-- pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
--                           kernel=(2,2), stride=(2,2))
-- # second conv
-- conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
-- tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
-- pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
--                           kernel=(2,2), stride=(2,2))
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

    v1 <- convolution "conv1" x [5,5] 30 nil
    a1 <- S.activation "conv1-a" v1 "relu"
    p1 <- S.pooling "conv1-p" a1 "(2,2)" "max" nil

    v2 <- convolution "conv2" p1 [5,5] 50 nil
    a2 <- S.activation "conv2-a" v2 "tanh"
    p2 <- S.pooling "conv2-p" a2 "(2,2)" "max" nil

    fl <- S.flatten "flatten" p2

    v3 <- fullyConnected "fc1" fl 500 nil
    a3 <- S.activation "fc1-a" v3 "tanh"

    v4 <- fullyConnected "fc2" a3 10 nil
    a4 <- S.softmaxoutput "softmax" v4 y nil
    return $ S.Symbol a4

range :: Int -> [Int]
range = enumFromTo 1

default_initializer :: DType a => Initializer a
default_initializer cxt shape = A.NDArray <$> A.random_normal 
                                        (add @"loc" 0 $ 
                                         add @"scale" 1 $ 
                                         add @"shape" (formatShape shape) $ 
                                         add @"ctx" (formatContext cxt) nil)
    
optimizer :: DType a => Optimizer a
optimizer _ v g = A.NDArray <$> A.sgd_update (A.getHandle v) (A.getHandle g) 0.01 nil

main :: IO ()
main = do
    -- call mxListAllOpNames can ensure the MXNet itself is properly initialized
    -- i.e. MXNet operators are registered in the NNVM
    _  <- mxListAllOpNames
    net <- neural
    params <- initialize net $ Config { 
                _cfg_placeholders = M.singleton "x" [1,1,28,28],
                _cfg_initializers = M.empty,
                _cfg_default_initializer = default_initializer,
                _cfg_context = contextGPU
              }

    result <- runResourceT $ train params $ do 
        liftIO $ putStrLn $ "[Train] "
        trdat <- getContext >>= return . trainingData
        ttdat <- getContext >>= return . testingData
        let index = SR.enumFrom (1 :: Int)
        forM_ (range 10) $ \ind -> do
            liftIO $ putStrLn $ "iteration " ++ show ind
            total <- SR.effects trdat
            _ <- flip SR.mapM_ (SR.zip index trdat) $ \(i, (x, y)) -> do
                liftIO $ do
                    putStr $ "\r\ESC[K" ++ show i ++ "/" ++ show total
                    hFlush stdout
                x <- liftIO $ reshape x [32,1,28,28]
                fit optimizer net $ M.fromList [("x", x), ("y", y)]
            liftIO $ putStr "\r\ESC[K"
        liftIO $ putStrLn $ "[Test] "
        total <- SR.effects ttdat
        SR.toList_ $ void $ flip SR.mapM (SR.zip index ttdat) $ \(i, (x, y)) -> do 
            liftIO $ do 
                putStr $ "\r\ESC[K" ++ show i ++ "/" ++ show total
                hFlush stdout
            x <- liftIO $ reshape x [1,1,28,28]
            [y'] <- forwardOnly net (M.fromList [("x", Just x), ("y", Nothing)])
            ind1 <- liftIO $ argmax y  >>= items
            ind2 <- liftIO $ argmax y' >>= items
            return (ind1, ind2)
    let (ls,ps) = unzip result
        ls_unbatched = mconcat ls
        ps_unbatched = mconcat ps
        total   = SV.length ls_unbatched
        correct = SV.length $ SV.filter id $ SV.zipWith (==) ls_unbatched ps_unbatched
    putStrLn $ "Accuracy: " ++ show correct ++ "/" ++ show total
  
  where
    argmax :: ArrayF -> IO ArrayF
    argmax ys = A.NDArray <$> A.argmax (A.getHandle ys) (add @"axis" 1 nil)
