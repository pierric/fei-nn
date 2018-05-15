{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.NN.Initializer where

import MXNet.Core.Base
import qualified MXNet.Core.Base.NDArray as A
import qualified MXNet.Core.Base.Internal.TH.NDArray as A
import qualified Data.Vector.Storable as SV
import Control.Monad.Trans.Resource (MonadThrow(..))

import MXNet.NN.Types
import MXNet.NN.Utils

zeros :: DType a => Initializer a
zeros = constant 0

ones :: DType a => Initializer a
ones  = constant 1

constant :: DType a => a -> Initializer a
constant val shp cxt = makeNDArray shp cxt $ SV.replicate (product shp) val

uniform :: forall a. (DType a, Floating a) => Float -> Initializer a
uniform sca shp cxt = A.NDArray <$> (A.random_uniform 
                        $ add @"low"    (-sca) 
                        $ add @"high"   sca
                        $ add @"shape"  (formatShape shp)
                        $ add @"ctx"    (formatContext cxt)
                        $ add @"dtype"  (typename (undefined :: a)) nil)

normal :: forall a. (DType a, Floating a) => Float -> Initializer a
normal sigma shp cxt = A.NDArray <$> (A.random_normal 
                        $ add @"loc"    0
                        $ add @"scale"  (fromRational $ toRational sigma)
                        $ add @"shape"  (formatShape shp)
                        $ add @"ctx"    (formatContext cxt) 
                        $ add @"dtype"  (typename (undefined :: a)) nil)

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
         XavierGaussian-> normal scale shp cxt
xavier _ _ _ shp _ = throwM $ InvalidArgument $ "invalid shape " ++ show  shp ++ " for xavier initializer"