{-# LANGUAGE DataKinds         #-}
{-# LANGUAGE OverloadedLabels  #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}
module TestModule where

import           Control.Lens                 (ix, use, (^?!), (^?))
import           RIO                          hiding (Const (..))
import qualified RIO.HashMap                  as M
import qualified RIO.HashSet                  as S
import qualified RIO.Vector.Storable          as VS
import           Test.Tasty
import           Test.Tasty.HUnit

import           MXNet.Base
import           MXNet.Base.Operators.Tensor  as T
import           MXNet.Base.Tensor.Functional as F
import           MXNet.NN
import           MXNet.NN.Layer
import           MXNet.NN.Module

mkSymbol :: Layer (Symbol Float)
mkSymbol = do
    x <- variable "x"
    unique "features" $ do
        m <- named "m" $ parameter "m" ReqWrite (Just [4, 4])
        o <- prim T.__npi_matmul (#a := x .& #b := m .& Nil)
        l <- F.sum_ o Nothing False >>= F.reshape [1]
        l <- makeLoss l 1.0
        o <- blockGrad o
        group [o, l]

mkSymbolNoLoss :: Layer (Symbol Float)
mkSymbolNoLoss = do
    x <- variable "x"
    unique "features" $ do
        m <- named "m" $ parameter "m" ReqWrite (Just [4, 4])
        prim T.__npi_matmul (#a := x .& #b := m .& Nil)

mkConfig  = Config {
    _cfg_data = M.fromList [("x", [1, 4])],
    _cfg_label = [],
    _cfg_initializers = M.empty,
    _cfg_default_initializer = SomeInitializer InitOnes,
    _cfg_fixed_params = S.fromList [],
    _cfg_context = contextCPU }

test_train_step :: TestTree
test_train_step = testCase "Test fitAndEval" step
    where
    -- instead of 'runFei . Simple', we have to use the 'runFeiMX'
    -- interface because we cannot send NotifyShutdown to mxnet.
    step = runFeiMX () False . unSimpleFeiM $ do
        x <- liftIO $ ndOnes [1, 4]
        sym <- runLayerBuilder mkSymbol
        initSession @"test" sym mkConfig
        optm <- makeOptimizer SGD'Mom (Const 0.1) Nil
        askSession $ do
            let mapping = M.fromList [("x", x)]
            fitAndEval optm mapping MNilData
            params <- use $ untag . mod_params
            case params ^? ix "features.m" of
                Just (ParameterG p g) -> liftIO $ do
                    p <- toVector p
                    g <- toVector g
                    assertBool "param == 0.9" $ VS.all (== 0.9) p
                    assertBool "gradient == 1.0" $ VS.all (== 1.0) g
                Just _  -> liftIO $ assertFailure "param 'm' has no gradient"
                Nothing -> liftIO $ assertFailure "param 'm' exists"

test_forward :: TestTree
test_forward = testCase "Test forwardOnly" step
    where
    step = runFeiMX () False . unSimpleFeiM $ do
        x <- liftIO $ ndOnes [1, 4]
        sym <- runLayerBuilder mkSymbol
        initSession @"test" sym mkConfig
        askSession $ do
            let mapping = M.fromList [("x", x)]
            out <- forwardOnly mapping
            liftIO $ do
                assertBool "two outputs" $ length out == 2
                out <- toVector $ out ^?! ix 0
                assertBool "output == 4.0" $ VS.all (== 4.0) out


test_shared_forward :: TestTree
test_shared_forward = testCase "Test forwardOnly with shared parameters" step
    where
    step = runFeiMX () False . unSimpleFeiM $ do
        sym <- runLayerBuilder mkSymbol
        sym_shared <- runLayerBuilder mkSymbolNoLoss
        initSession @"test" sym mkConfig
        askSession $ do
            let mapping = M.fromList [("x", [2, 4])]
            withSharedParameters sym_shared mapping $ \forward -> do
                x <- ndOnes [2, 4]
                out <- forward (M.fromList [("x", x)])
                assertBool "two outputs" $ length out == 1
                out <- toVector $ out ^?! ix 0
                assertBool "output == 4.0" $ VS.all (== 4.0) out

                x <- fromVector [2, 4] (VS.replicate 8 0.5)
                out <- forward (M.fromList [("x", x)])
                assertBool "two outputs" $ length out == 1
                out <- toVector $ out ^?! ix 0
                assertBool "output == 2.0" $ VS.all (== 2.0) out
