{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
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
import MXNet.NN
import MXNet.NN.Utils
import MXNet.NN.Layer
import Dataset

neural :: IO SymbolF
neural = do
    x  <- variable "x"
    y  <- variable "y"
    v1 <- fullyConnected "fc1" x 128 nil
    a1 <- S.activation "fc1-a" v1 "relu"
    v2 <- fullyConnected "fc2" a1 10 nil
    a2 <- S.softmaxoutput "softmax" v2 y nil
    return $ S.Symbol a2

range :: Int -> [Int]
range = enumFromTo 1

default_initializer :: DType a => Initializer a
default_initializer cxt shape = A.NDArray <$> A.random_normal 
                                        (add @"loc" 0 $ 
                                         add @"scale" 1 $ 
                                         add @"shape" (formatShape shape) $ 
                                         add @"ctx" (formatContext cxt) nil)    
main :: IO ()
main = do
    -- call mxListAllOpNames can ensure the MXNet itself is properly initialized
    -- i.e. MXNet operators are registered in the NNVM
    _    <- mxListAllOpNames
    net  <- neural
    sess <- initialize net $ Config { 
                _cfg_placeholders = M.singleton "x" [32,28,28],
                _cfg_initializers = M.empty,
                _cfg_default_initializer = default_initializer,
                _cfg_context = contextGPU
            }
    optimizer <- makeOptimizer 0.002 nil :: IO (SGD Float '[])

    result <- runResourceT $ train sess $ do 
        liftIO $ putStrLn $ "[Train] "
        forM_ (range 5) $ \ind -> do
            liftIO $ putStrLn $ "iteration " ++ show ind
            SR.mapM_ (\(x, y) -> fit optimizer net $ M.fromList [("x", x), ("y", y)]) trainingData

        liftIO $ putStrLn $ "[Test] "
        SR.toList_ $ void $ flip SR.mapM testingData $ \(x, y) -> do 
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
