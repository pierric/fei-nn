{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.NN.Initializer where

import           RIO
import qualified RIO.Text                    as T
import qualified RIO.Vector.Storable         as SV
import           Type.Set                    (Insert)

import           MXNet.Base
import qualified MXNet.Base.Operators.Tensor as T
import           MXNet.NN.Types
import           MXNet.NN.Utils

upd :: forall a. IO [NDArray a] -> IO ()
upd = void

empty :: DType a => Initializer a
empty _ arr = return ()

zeros :: DType a => Initializer a
zeros = constant 0

ones :: DType a => Initializer a
ones  = constant 1

constant :: forall a. DType a => Float -> Initializer a
constant val _ arr = upd @a $ T.__set_value (#src := val .& Nil) (Just [arr])

vector :: DType a => SV.Vector a -> Initializer a
vector val _ arr = copyFromVector arr val

type BasicFloatDTypesWithNone = Insert "None" BasicFloatDTypes

uniform :: forall a. (DType a, InEnum (DTypeName a) BasicFloatDTypesWithNone)
    => Float -> Initializer a
uniform sca _ arr = upd @a $ T.__random_uniform
                            (#low    := (-sca)
                          .& #high   := sca
                          .& Nil) (Just [arr])

normal :: forall a. (DType a, InEnum (DTypeName a) BasicFloatDTypesWithNone)
    => Float -> Initializer a
normal sigma _ arr = upd @a $ T.__random_normal
                             (#loc    := (0 :: Float)
                           .& #scale  := sigma
                           .& Nil) (Just [arr])

data XavierFactor = XavierAvg
    | XavierIn
    | XavierOut
data XavierRandom = XavierUniform
    | XavierGaussian

xavier :: (DType a, InEnum (DTypeName a) BasicFloatDTypesWithNone)
    => Float -> XavierRandom -> XavierFactor -> Initializer a
xavier magnitude distr factor name arr = do
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
          XavierUniform  -> uniform scale name arr
          XavierGaussian -> normal  scale name arr
