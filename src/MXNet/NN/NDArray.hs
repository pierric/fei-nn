module MXNet.NN.NDArray where

import           MXNet.Base
import qualified MXNet.Base.Operators.NDArray as I
import           RIO
import qualified RIO.NonEmpty                 as RNE

reshape :: DType a => NDArray a -> NonEmpty Int -> IO (NDArray a)
reshape arr shp = do
    hdl <- sing I._Reshape (#data := unNDArray arr .& #shape := RNE.toList shp .& Nil)
    return $ NDArray hdl

transpose :: DType a => NDArray a -> [Int] -> IO (NDArray a)
transpose arr axes = do
    hdl <- sing I.transpose (#data := unNDArray arr .& #axes := axes .& Nil)
    return $ NDArray hdl

copy :: DType a => NDArray a -> NDArray a -> IO (NDArray a)
copy src dst = do
    I._copyto_upd [unNDArray dst] (#data := unNDArray src .& Nil)
    return dst

expandDims :: DType a => NDArray a -> Int -> IO (NDArray a)
expandDims arr axis = do
    hdl <- sing I.expand_dims (#data := unNDArray arr .& #axis := axis .& Nil)
    return $ NDArray hdl

