{-# LANGUAGE RecordWildCards #-}
module MXNet.NN.Utils where

import Data.List (intersperse)
import qualified Data.Text as T
import qualified Data.HashMap.Strict as M
import Control.Lens (use)
import Control.Monad.IO.Class (MonadIO, liftIO)

import MXNet.Base (Context(..), DType, NDArray(..), Symbol(..), mxSymbolSaveToFile, mxNDArraySave)
import MXNet.NN.Types 

-- | format a shape
formatShape :: [Int] -> String
formatShape shape = concat $ ["("] ++ intersperse "," (map show shape) ++ [")"]

-- | format a context
formatContext :: Context -> String
formatContext Context{..} = getDeviceName _device_type ++ "(" ++ show _device_id ++ ")"
  where 
    getDeviceName 1 = "cpu"
    getDeviceName 2 = "gpu"
    getDeviceName 3 = "cpu_pinned"
    getDeviceName _ = error "formatContext: unknown device type"

endsWith :: String -> String -> Bool
endsWith s1 s2 = T.isSuffixOf (T.pack s1) (T.pack s2)

saveSession :: (MonadIO m, DType a) => String -> TrainM a m ()
saveSession filename = do
    (dat, _) <- use sess_data
    (lbl, _) <- use sess_label
    params <- use sess_param
    net <- use sess_symbol
    let modelParams = map getModelParam $ M.toList $ M.filterWithKey (\k _ -> k /= dat && k /= lbl) params
    liftIO $ do
        mxSymbolSaveToFile (filename ++ ".json") (unSymbol net)
        mxNDArraySave (filename ++ ".params") modelParams
  where
    getModelParam (key, ParameterI a _) = ("arg:" ++ key, unNDArray a)
    getModelParam (key, ParameterA a) = ("aux:" ++ key, unNDArray a)