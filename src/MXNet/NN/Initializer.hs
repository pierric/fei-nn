{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE QuasiQuotes #-}
module MXNet.NN.Initializer where

import Control.Monad.Trans.Resource (MonadThrow(..))

import MXNet.Base
import qualified MXNet.Base.Operators.NDArray as A
import qualified Data.Vector.Storable as SV

import MXNet.NN.Types
import MXNet.NN.Utils

zeros :: DType a => Initializer a
zeros = constant 0

ones :: DType a => Initializer a
ones  = constant 1

constant :: DType a => a -> Initializer a
constant val _ shp cxt = makeNDArray shp cxt $ SV.replicate (product shp) val

uniform :: forall a. (DType a, HasEnum (DTypeName a) '["None", "float16" ,"float32", "float64"]) 
    => Float -> Initializer a
uniform sca _ shp cxt = NDArray . head <$> (A._random_uniform 
                               (  #low    := (-sca) 
                               .& #high   := sca
                               .& #shape  := shp
                               .& #ctx    := formatContext cxt
                               .& #dtype  := EnumType (typename (undefined :: a))
                               .& Nil))

normal :: forall a. (DType a, HasEnum (DTypeName a) '["None", "float16" ,"float32", "float64"]) 
    => Float -> Initializer a
normal sigma _ shp cxt = NDArray . head <$> (A._random_normal
                               (  #loc    := (0 :: Float)
                               .& #scale  := sigma
                               .& #shape  := shp
                               .& #ctx    := formatContext cxt
                               .& #dtype  := EnumType (typename (undefined :: a))
                               .& Nil))

data XavierFactor = XavierAvg | XavierIn | XavierOut
data XavierRandom = XavierUniform | XavierGaussian

xavier :: (DType a, HasEnum (DTypeName a) '["None", "float16" ,"float32", "float64"])
    => Float -> XavierRandom -> XavierFactor -> Initializer a
xavier magnitude distr factor name (shp@[ofan,ifan]) cxt =
    let scale = case factor of 
                  XavierIn  -> sqrt (magnitude / fromIntegral ifan)
                  XavierOut -> sqrt (magnitude / fromIntegral ofan)
                  XavierAvg -> sqrt (magnitude * 2.0 / fromIntegral (ifan + ofan))
    in case distr of
         XavierUniform -> uniform scale name shp cxt
         XavierGaussian-> normal  scale name shp cxt
xavier _ _ _ _ shp _ = throwM $ InvalidArgument $ "invalid shape " ++ show  shp ++ " for xavier initializer"