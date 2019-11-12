{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
module MXNet.NN.Utils where

import Data.List (intersperse)
import qualified Data.Text as T
import System.IO (hFlush, stdout)


import MXNet.Base (Context(..))

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


-- saveState :: Bool -> String -> ModuleState a -> IO ()
-- saveState save_symbol name state = do
--     let params = state ^. mod_params
--         symbol = state ^. mod_symbol
--         modelParams = map getModelParam $ M.toList params
--     when save_symbol $ mxSymbolSaveToFile (name ++ ".json") (unSymbol symbol)
--     mxNDArraySave (name ++ ".params") modelParams
--   where
--     getModelParam (key, ParameterI a _) = ("arg:" ++ key, unNDArray a)
--     getModelParam (key, ParameterA a) = ("aux:" ++ key, unNDArray a)

-- loadSession :: DType a => String -> [String] -> TrainM a t ()
-- loadSession filename ignores = do
--     arrays <- liftIO $ mxNDArrayLoad (filename ++ ".params")
--     params <- use sess_param
--     forM_ arrays $ \(name, hdl) ->
--         case break (==':') name of
--             (_, "") -> throwM (LoadSessionInvalidTensorName name)
--             ("", _) -> throwM (LoadSessionInvalidTensorName name)
--             (typ, ':':key) ->
--                 case (key `elem` ignores, typ, M.lookup key params) of
--                     (True, _, _) -> return ()
--                     (_, _, Nothing) -> liftIO $ putStrLn $ printf "Tensor %s is missing." name
--                     (_, "arg", Just (ParameterI target grad)) -> liftIO $ A._copyto_upd [unNDArray target] (#data := hdl .& Nil)
--                     (_, "aux", Just (ParameterA target))      -> liftIO $ A._copyto_upd [unNDArray target] (#data := hdl .& Nil)
--                     _ -> throwM (LoadSessionMismatchedTensorKind name)

printInLine :: String -> IO ()
printInLine str = do
    putStr $ "\r\ESC[K" ++ str
    hFlush stdout

