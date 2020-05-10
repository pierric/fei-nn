{-# LANGUAGE RecordWildCards #-}
module MXNet.NN.Utils where

import RIO
import qualified RIO.Text as T
import qualified RIO.NonEmpty as RNE
import RIO.List (sortOn, lastMaybe)
import Data.Maybe (mapMaybe)
import RIO.Directory (listDirectory, getModificationTime)
import RIO.FilePath ((</>), dropExtension)
import qualified RIO.HashMap as M
import System.IO (stdout, putStr)
import Control.Lens (use)
import Formatting

import MXNet.Base (Context(..), NDArray(..), Symbol(..), (.&), ArgOf(..), HMap(..))
import MXNet.Base.Raw (mxSymbolSaveToFile, mxNDArraySave, mxNDArrayLoad)
import qualified MXNet.Base.Operators.NDArray as A
import MXNet.NN.Types (Module, Parameter(..), mod_params, mod_symbol)
import MXNet.NN.TaggedState (untag)

-- | format a shape
formatShape :: NonEmpty Int -> Text
formatShape shape = let sshape = RNE.intersperse "," (RNE.map tshow shape)
                    in T.concat $ ["("] ++ RNE.toList sshape ++ [")"]

-- | format a context
formatContext :: Context -> Text
formatContext Context{..} = sformat (stext % "(" % int % ")") (getDeviceName _device_type) _device_id
  where
    getDeviceName :: Int -> Text
    getDeviceName 1 = "cpu"
    getDeviceName 2 = "gpu"
    getDeviceName 3 = "cpu_pinned"
    getDeviceName _ = error "formatContext: unknown device type"

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
        when save_symbol $ mxSymbolSaveToFile (T.pack $ name ++ ".json") (unSymbol symbol)
        mxNDArraySave (T.pack $ name ++ ".params") modelParams
  where
    getModelParam (_,   ParameterV _)   = Nothing
    getModelParam (key, ParameterF a)   = Just (key, unNDArray a)
    getModelParam (key, ParameterG a _) = Just (key, unNDArray a)
    getModelParam (key, ParameterA a)   = Just (key, unNDArray a)

loadState :: (MonadIO m, MonadReader env m, HasLogFunc env, HasCallStack)
    => String -> [Text] -> Module t a m ()
loadState weights_filename ignores = do
    arrays <- liftIO $ mxNDArrayLoad (T.pack $ weights_filename ++ ".params")
    params <- use (untag . mod_params)
    forM_ arrays $ \(name, hdl) -> do
        case (name `elem` ignores, M.lookup name params) of
            (True, _) ->
                return ()
            (_, Nothing) ->
                lift $ logInfo $ display $ sformat ("Tensor " % stext % " is missing.") name
            (_, Just (ParameterG target _)) ->
                liftIO $ A._copyto_upd [unNDArray target] (#data := hdl .& Nil)
            (_, Just (ParameterF target)) ->
                liftIO $ A._copyto_upd [unNDArray target] (#data := hdl .& Nil)
            (_, Just (ParameterA target)) ->
                liftIO $ A._copyto_upd [unNDArray target] (#data := hdl .& Nil)
            (_, Just (ParameterV _)) ->
                logWarn . display $ sformat
                    ("a variable (" % stext % ") found in the state file.") name

lastSavedState :: MonadIO m => Text -> Module t a m (Maybe FilePath)
lastSavedState dir = liftIO $ do
    let sdir = T.unpack dir
    files <- listDirectory sdir
    let param_files = filter (T.isSuffixOf ".params" . T.pack) files
    if null param_files
        then return Nothing
        else do
            mod_time <- mapM (getModificationTime . (sdir </>)) param_files
            case lastMaybe $ sortOn snd (zip param_files mod_time) of
                Nothing -> return Nothing
                Just (latest, _) -> return $ Just $ sdir </> dropExtension latest

printInLine :: Text -> IO ()
printInLine str = do
    putStr $ "\r\ESC[K" ++ T.unpack str
    hFlush stdout

