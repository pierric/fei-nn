{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE QuasiQuotes #-}
module MXNet.NN.Initializer where

import MXNet.Core.Base
import qualified MXNet.Core.Base.NDArray as A
import qualified MXNet.Core.Base.Internal.TH.NDArray as A
import qualified Data.Vector.Storable as SV
import Control.Monad.Trans.Resource (MonadThrow(..))

import MXNet.NN.Types
import MXNet.NN.Utils
import MXNet.NN.Utils.HMap

zeros :: DType a => Initializer a
zeros = constant 0

ones :: DType a => Initializer a
ones  = constant 1

constant :: DType a => a -> Initializer a
constant val shp cxt = makeNDArray shp cxt $ SV.replicate (product shp) val

uniform :: forall a. (DType a, Floating a) => Float -> Initializer a
uniform sca shp cxt = A.NDArray <$> (A.random_uniform 
                            [hm| low    := (-sca) 
                               , high   := sca
                               , shape  := (formatShape shp)
                               , ctx    := (formatContext cxt)
                               , dtype  := (typename (undefined :: a)) |])

normal :: forall a. (DType a, Floating a) => Float -> Initializer a
normal sigma shp cxt = A.NDArray <$> (A.random_normal 
                            [hm| loc    := (0 :: Float)
                               , scale  := sigma
                               , shape  := (formatShape shp)
                               , ctx    := (formatContext cxt) 
                               , dtype  := (typename (undefined :: a)) |])

data XavierFactor = XavierAvg | XavierIn | XavierOut
data XavierRandom = XavierUniform | XavierGaussian

xavier :: (DType a, Floating a) => Float -> XavierRandom -> XavierFactor -> Initializer a
xavier magnitude distr factor (shp@[ofan,ifan]) cxt =
    let scale = case factor of 
                  XavierIn  -> sqrt (magnitude / fromIntegral ifan)
                  XavierOut -> sqrt (magnitude / fromIntegral ofan)
                  XavierAvg -> sqrt (magnitude * 2.0 / fromIntegral (ifan + ofan))
    in case distr of
         XavierUniform -> uniform scale shp cxt
         XavierGaussian-> normal  scale shp cxt
xavier _ _ _ shp _ = throwM $ InvalidArgument $ "invalid shape " ++ show  shp ++ " for xavier initializer"