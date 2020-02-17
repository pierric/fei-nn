{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
module MXNet.NN.Utils where

import System.IO (hFlush, stdout)
import Data.List (intersperse, sortOn)
import Data.Maybe (mapMaybe)
import qualified Data.Text as T
import qualified Data.HashMap.Strict as M
import Control.Lens (use)
import Control.Monad (when, forM_)
import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Monad.Trans.Resource (MonadThrow(..))
import System.Directory (listDirectory, getModificationTime)
import System.FilePath ((</>), dropExtension)
import Text.Printf

import MXNet.Base (Context(..), NDArray(..), Symbol(..), (.&), ArgOf(..), HMap(..))
import MXNet.Base.Raw (mxSymbolSaveToFile, mxNDArraySave, mxNDArrayLoad)
import qualified MXNet.Base.Operators.NDArray as A
import MXNet.NN.Types (Module, Parameter(..), Exc(..), mod_params, mod_symbol)
import MXNet.NN.TaggedState (untag)

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


-- class Session s where
--     saveSession :: (String -> String) -> Bool -> ST.StateT s IO ()
--
-- instance L.KnownSymbol tag => Session (TaggedModuleState a tag) where
--     saveSession make_filename save_symbol = do
--         st <- ST.get
--         let name = L.symbolVal (Proxy :: Proxy tag)
--         liftIO $ saveState save_symbol (make_filename name) (st ^. untag)
--
-- instance DT.Every L.KnownSymbol tags => Session (DT.Prod (TaggedModuleState a) tags) where
--     saveSession make_filename save_symbol = do
--         tagged_states <- ST.get
--         let names  = toNames (DT.map1 (const Proxy) tagged_states)
--             states = DT.toList (^. untag) tagged_states
--         liftIO $ zipWithM_ (saveState save_symbol) names states
--       where
--         toNames :: forall (t :: [L.Symbol]). DT.Every L.KnownSymbol t => DT.Prod Proxy t -> [String]
--         toNames DT.Ø = []
--         toNames (v DT.:< rem) = L.symbolVal v : toNames rem


saveState :: MonadIO m => Bool -> String -> Module t a m ()
saveState save_symbol name = do
    params <- use (untag . mod_params)
    symbol <- use (untag . mod_symbol)
    let modelParams = mapMaybe getModelParam $ M.toList params
    liftIO $ do
        when save_symbol $ mxSymbolSaveToFile (name ++ ".json") (unSymbol symbol)
        mxNDArraySave (name ++ ".params") modelParams
  where
    getModelParam (_, ParameterV _)     = Nothing
    getModelParam (key, ParameterG a _) = Just ("arg:" ++ key, unNDArray a)
    getModelParam (key, ParameterA a)   = Just ("aux:" ++ key, unNDArray a)

loadState :: MonadIO m => String -> [String] -> Module t a m ()
loadState weights_filename ignores = do
    arrays <- liftIO $ mxNDArrayLoad (weights_filename ++ ".params")
    params <- use (untag . mod_params)
    liftIO $ forM_ arrays $ \(name, hdl) -> do
        case break (==':') name of
            (typ, ':':key) | typ /= ""->
                case (key `elem` ignores, typ, M.lookup key params) of
                    (True, _, _) -> return ()
                    (_, _, Nothing) -> putStrLn $ printf "Tensor %s is missing." name
                    (_, "arg", Just (ParameterG target _)) -> A._copyto_upd [unNDArray target] (#data := hdl .& Nil)
                    (_, "arg", Just (ParameterV target))   -> A._copyto_upd [unNDArray target] (#data := hdl .& Nil)
                    (_, "aux", Just (ParameterA target))   -> A._copyto_upd [unNDArray target] (#data := hdl .& Nil)
                    _ -> throwM (LoadSessionMismatchedTensorKind name)
            _ -> throwM (LoadSessionInvalidTensorName name)

lastSavedState :: MonadIO m => String -> Module t a m (Maybe FilePath)
lastSavedState dir = liftIO $ do
    files <- listDirectory dir
    let param_files = filter (endsWith ".params") files
    if null param_files
        then return Nothing
        else do
            mod_time <- mapM (getModificationTime . (dir </>)) param_files
            let latest = fst $ last $ sortOn snd (zip param_files mod_time)
            return $ Just $ dir </> dropExtension latest

printInLine :: String -> IO ()
printInLine str = do
    putStr $ "\r\ESC[K" ++ str
    hFlush stdout

