{-# LANGUAGE RecordWildCards #-}
module MXNet.NN.Utils where

import Data.List (intersperse)
import qualified Data.Text as T
import qualified Data.HashMap.Strict as M
import Control.Lens (use)
import Control.Monad (forM_)
import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Monad.Trans.Resource (MonadThrow(..))
import Text.Printf

import MXNet.Base (
    Context(..), DType, NDArray(..), Symbol(..), 
    HMap(..), (.&), ArgOf(..),
    mxSymbolSaveToFile, mxNDArraySave, mxNDArrayLoad)
import MXNet.NN.Types
import qualified MXNet.Base.Operators.NDArray as A

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
    dat_vars <- M.keys <$> use sess_data
    lbl_vars <- M.keys <$> use sess_label
    params <- use sess_param
    net <- use sess_symbol
    let all_vars = dat_vars ++ lbl_vars
        modelParams = map getModelParam $ M.toList $ M.filterWithKey (\k _ -> not (k `elem` all_vars)) params
    liftIO $ do
        mxSymbolSaveToFile (filename ++ ".json") (unSymbol net)
        mxNDArraySave (filename ++ ".params") modelParams
  where
    getModelParam (key, ParameterI a _) = ("arg:" ++ key, unNDArray a)
    getModelParam (key, ParameterA a) = ("aux:" ++ key, unNDArray a)

loadSession :: (MonadThrow m, MonadIO m, DType a) => String -> TrainM a m ()
loadSession filename = do
    arrays <- liftIO $ mxNDArrayLoad (filename ++ ".params")
    params <- use sess_param
    forM_ arrays $ \(name, hdl) -> 
        case break (==':') name of
            (_, "") -> throwM (LoadSessionInvalidTensorName name)
            ("", _) -> throwM (LoadSessionInvalidTensorName name)
            (typ, ':':key) -> 
                case (typ, M.lookup key params) of
                    (_, Nothing) -> liftIO $ putStrLn $ printf "Tensor %s is missing." name
                    ("arg", Just (ParameterI target grad)) -> liftIO $ A._copyto_upd [unNDArray target] (#data := hdl .& Nil)
                    ("aux", Just (ParameterA target))      -> liftIO $ A._copyto_upd [unNDArray target] (#data := hdl .& Nil)
                    _ -> throwM (LoadSessionMismatchedTensorKind name)
