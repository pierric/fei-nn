{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables   #-}
module MXNet.NN.Initializer where

import           GHC.Read
import           RIO
import qualified RIO.Text                    as T
import qualified RIO.Vector.Storable         as SV

import           MXNet.Base
import qualified MXNet.Base.Operators.Tensor as T
import           MXNet.NN.Types
import           MXNet.NN.Utils

upd :: forall a. IO [NDArray a] -> IO ()
upd = void

data SimpleInit a = InitEmpty | InitZeros | InitOnes | InitWithVal Float | InitWithVec (SV.Vector a)
    deriving (Show, Read)

instance DType a => Initializer SimpleInit a where
    initNDArray InitEmpty _ _              = return ()
    initNDArray InitZeros name arr         = constant 0 name arr
    initNDArray InitOnes  name arr         = constant 1 name arr
    initNDArray (InitWithVal val) name arr = constant val name arr
    initNDArray (InitWithVec val) _ arr    = copyFromVector arr val

constant :: forall a. DType a => Float -> Text -> NDArray a -> IO ()
constant val _ arr = upd @a $ T.__set_value (#src := val .& Nil) (Just [arr])

data RandomInit a = InitUniform Float
                  | InitNormal  Float
                  | InitXavier  Float XavierRandom XavierFactor
    deriving (Show, Read)

data XavierFactor = XavierAvg
    | XavierIn
    | XavierOut
    deriving (Show, Read)
data XavierRandom = XavierUniform
    | XavierGaussian
    deriving (Show, Read)

instance (DType a, InEnum (DTypeName a) BasicFloatDTypes)
  => Initializer RandomInit a where
    initNDArray (InitUniform sca) _ arr =
        upd @a $ T.__random_uniform (#low    := (-sca)
                                  .& #high   := sca
                                  .& Nil) (Just [arr])
    initNDArray (InitNormal sigma) _ arr =
        upd @a $ T.__random_normal (#loc    := (0 :: Float)
                                 .& #scale  := sigma
                                 .& Nil) (Just [arr])
    initNDArray (InitXavier magnitude distr factor) name arr = do
        shp <- ndshape arr
        if length shp < 2
        then throwM $ InvalidArgument $
                T.concat ["invalid shape ", formatShape shp, " for xavier initializer"]
        else do
            let ofan : dims = shp
                ifan = product dims
                scale = case factor of
                          XavierIn  -> sqrt (magnitude / fromIntegral ifan)
                          XavierOut -> sqrt (magnitude / fromIntegral ofan)
                          XavierAvg -> sqrt (magnitude * 2.0 / fromIntegral (ifan + ofan))
            case distr of
              XavierUniform  -> initNDArray (InitUniform scale) name arr
              XavierGaussian -> initNDArray (InitNormal  scale) name arr


newtype CustomInit a = CustomInit (Text -> NDArray a -> IO ())

instance DType a => Initializer CustomInit a where
    initNDArray (CustomInit func) name arr = func name arr
