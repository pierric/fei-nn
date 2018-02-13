{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
module Main where

import MXNet.Core.Base
import qualified MXNet.Core.Base.NDArray as A
import qualified MXNet.Core.Base.Internal.TH.NDArray as A
import qualified Data.HashMap.Strict as M
import Control.Monad (forM_)
import qualified Streaming.Prelude as SR
import qualified Data.Vector.Storable as SV
import Data.List (intersperse)
import Control.Monad.IO.Class
import Control.Monad.Trans.Resource
import MXNet.NN
import Dataset

neural :: IO SymbolF
neural = do
    x  <- variable "x"  :: IO SymbolF 
    y  <- variable "y"  :: IO SymbolF
    w1 <- variable "w1" :: IO SymbolF
    b1 <- variable "b1" :: IO SymbolF
    v1 <- convolution x w1 b1 "(5,5)" 20
    a1 <- activation v1 "tanh"
    a2 <- pooling a1 "(2,2)" "max"
    w2 <- variable "w2" :: IO SymbolF
    b2 <- variable "b2" :: IO SymbolF
    v2 <- fullyConnected a2 w2 b2 128
    a3 <- activation v2 "relu"
    w3 <- variable "w3" :: IO SymbolF
    b3 <- variable "b3" :: IO SymbolF
    v3 <- fullyConnected a3 w3 b3 10
    a4 <- softmaxOutput v3 y 
    return a4

range :: Int -> [Int]
range = enumFromTo 1

default_initializer :: DType a => [Int] -> IO (NDArray a)
default_initializer shape = A.NDArray <$> A.random_normal (add @"loc" 0 $ add @"scale" 1 $ add @"shape" formatedShape nil)
  where
    formatedShape = concat $ ["("] ++ intersperse "," (map show shape) ++ [")"]
    
optimizer :: DType a => NDArray a -> NDArray a -> IO (NDArray a)
optimizer v g = A.NDArray <$> (A.sgd_update (A.getHandle v) (A.getHandle g) 0.01 nil)

main :: IO ()
main = do
    -- call mxListAllOpNames can ensure the MXNet itself is properly initialized
    -- i.e. MXNet operators are registered in the NNVM
    _  <- mxListAllOpNames
    net <- neural
    params <- initialize net $ Config { 
                _cfg_placeholders = M.singleton "x" [32,1,28,28],
                _cfg_initializers = M.empty,
                _cfg_default_initializer = default_initializer
              }
    result <- runResourceT $ train params contextCPU $ do 
        liftIO $ putStrLn $ "[Train] "
        forM_ (range 5) $ \ind -> do
            liftIO $ putStrLn $ "iteration " ++ show ind
            SR.mapM_ (\(x, y) -> do
                x <- liftIO $ reshape x [32,1,28,28]
                fit optimizer net $ M.fromList [("x", x), ("y", y)]) trainingData
        liftIO $ putStrLn $ "[Test] "
        SR.toList_ $ flip SR.mapM testingData $ \(x, y) -> do 
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