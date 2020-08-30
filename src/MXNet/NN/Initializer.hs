{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.NN.Initializer where

import           RIO
import qualified RIO.NonEmpty                as RNE
import qualified RIO.Text                    as T
import qualified RIO.Vector.Storable         as SV

import           MXNet.Base
import qualified MXNet.Base.Operators.Tensor as T
import           MXNet.Base.Tensor           (prim)
import           MXNet.NN.Types
import           MXNet.NN.Utils

empty :: DType a => Initializer a
empty _ shp cxt = makeEmptyNDArray shp cxt

zeros :: DType a => Initializer a
zeros = constant 0

ones :: DType a => Initializer a
ones  = constant 1

constant :: DType a => a -> Initializer a
constant val _ shp cxt = makeNDArray shp cxt $ SV.replicate (product shp) val

uniform :: forall a. (DType a, HasEnum (DTypeName a) '["None", "float16" ,"float32", "float64"])
    => Float -> Initializer a
uniform sca _ shp cxt = prim T.__random_uniform
                               (  #low    := (-sca)
                               .& #high   := sca
                               .& #shape  := RNE.toList shp
                               .& #ctx    := formatContext cxt
                               .& #dtype  := EnumType (typename (undefined :: a))
                               .& Nil)

normal :: forall a. (DType a, HasEnum (DTypeName a) '["None", "float16" ,"float32", "float64"])
    => Float -> Initializer a
normal sigma _ shp cxt = prim T.__random_normal
                               (  #loc    := (0 :: Float)
                               .& #scale  := sigma
                               .& #shape  := RNE.toList shp
                               .& #ctx    := formatContext cxt
                               .& #dtype  := EnumType (typename (undefined :: a))
                               .& Nil)

data XavierFactor = XavierAvg
    | XavierIn
    | XavierOut
data XavierRandom = XavierUniform
    | XavierGaussian

xavier :: (DType a, HasEnum (DTypeName a) '["None", "float16" ,"float32", "float64"])
    => Float -> XavierRandom -> XavierFactor -> Initializer a
xavier magnitude distr factor name shp cxt
    | RNE.length shp < 2 = throwM $ InvalidArgument $
        T.concat ["invalid shape ", formatShape shp, " for xavier initializer"]
    | otherwise =
        let ofan :| dims = shp
            ifan = product dims
            scale = case factor of
                      XavierIn  -> sqrt (magnitude / fromIntegral ifan)
                      XavierOut -> sqrt (magnitude / fromIntegral ofan)
                      XavierAvg -> sqrt (magnitude * 2.0 / fromIntegral (ifan + ofan))
        in case distr of
             XavierUniform  -> uniform scale name shp cxt
             XavierGaussian-> normal  scale name shp cxt
