{-# LANGUAGE RecordWildCards #-}
module MXNet.NN.Utils where

import MXNet.Core.Base.DType
import Data.List (intersperse)

-- | format a shape
formatShape :: [Int] -> String
formatShape shape = concat $ ["("] ++ intersperse "," (map show shape) ++ [")"]

-- | format a context
formatContext :: Context -> String
formatContext Context{..} = getDeviceName deviceType ++ "(" ++ show deviceId ++ ")"
  where 
    getDeviceName 1 = "cpu"
    getDeviceName 2 = "gpu"
    getDeviceName 3 = "cpu_pinned"
    getDeviceName _ = error "formatContext: unknown device type"