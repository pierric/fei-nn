module MXNet.NN.NDArray where

import MXNet.Base
import qualified MXNet.Base.Operators.NDArray as I

reshape :: DType a => NDArray a -> [Int] -> IO (NDArray a)
reshape arr shp = do
    [hdl] <- I._Reshape (#data := unNDArray arr .& #shape := shp .& Nil)
    return $ NDArray hdl

transpose :: DType a => NDArray a -> [Int] -> IO (NDArray a)
transpose arr axes = do
    [hdl] <- I.transpose (#data := unNDArray arr .& #axes := axes .& Nil)
    return $ NDArray hdl
