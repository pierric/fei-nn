{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards       #-}
module MXNet.NN.Utils where

import           Control.Lens                 (use)
import           Formatting
import           RIO
import           RIO.Directory                (getModificationTime,
                                               listDirectory)
import           RIO.FilePath                 (dropExtension, (</>))
import qualified RIO.HashMap                  as M
import           RIO.List                     (intersperse, lastMaybe, sortOn)
import qualified RIO.Text                     as T

import           MXNet.Base                   (Context (..), DType,
                                               NDArray (..), Symbol (..),
                                               ndshape)
import           MXNet.Base.Raw               (mxNDArrayLoad, mxNDArraySave,
                                               mxSymbolSaveToFile)
import           MXNet.Base.Tensor.Functional (copy)
import           MXNet.NN.TaggedState         (untag)
import           MXNet.NN.Types               (Module, Parameter (..),
                                               mod_params, mod_symbol)

-- | format a shape
formatShape :: [Int] -> Text
formatShape shape = let sshape = intersperse "," (map tshow shape)
                    in T.concat $ ["("] ++ sshape ++ [")"]

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
    let modelParams = concatMap getModelParam $ M.toList params
    liftIO $ do
        when save_symbol $
            mxSymbolSaveToFile (T.pack $ name ++ ".json") $ unSymbol symbol
        mxNDArraySave (T.pack $ name ++ ".params") modelParams
  where
    getModelParam (key, ParameterV a)   = [(key, unNDArray a)]
    getModelParam (key, ParameterA a)   = [(key, unNDArray a)]
    getModelParam (key, ParameterG a g) =
        [(key, unNDArray a), (key `T.append` "__grad", unNDArray g)]

loadState :: (DType a, MonadIO m, MonadReader env m, HasLogFunc env, HasCallStack)
    => String -> [Text] -> Module t a m ()
loadState weights_filename ignores = do
    arrays <- liftIO $ mxNDArrayLoad (T.pack $ weights_filename ++ ".params")
    params <- use (untag . mod_params)
    forM_ arrays $ \(name, hdl) -> do
        let nameIfGrad = T.stripSuffix "__grad" name
            nameIngore = name `elem` ignores
            param1 = M.lookup name params
            param2 = nameIfGrad >>= flip M.lookup params
        case (nameIngore, param1, nameIfGrad, param2) of
            (True, _, _, _) ->
                return ()
            (_, Nothing, _, Nothing) ->
                lift $ logInfo $ display $ sformat
                    ("Tensor " % stext % " is ignored as missing in the model.") name
            (_, Nothing, Just name', Just (ParameterG _ target)) -> do
                checkShape name' (NDArray hdl) target
                liftIO $ void $ copy (NDArray hdl) target
            (_, Nothing, Just _, Just _) -> do
                -- we silently ignore any missing grad,
                -- for it is too common if we load the model for inference
                return ()
            (_, Just (ParameterV target), _, _) -> do
                -- we load weights for variables (tensor w/o grad) too
                -- because fixed tensor is treated as variable.
                checkShape name (NDArray hdl) target
                liftIO $ void $ copy (NDArray hdl) target
            (_, Just (ParameterG target _), _, _) -> do
                checkShape name (NDArray hdl) target
                liftIO $ void $ copy (NDArray hdl) target
            (_, Just (ParameterA target), _, _)   -> do
                checkShape name (NDArray hdl) target
                liftIO $ void $ copy (NDArray hdl) target
            (_, _, Nothing, Just _) -> do
                error "This won't happen"
    where
        checkShape :: (MonadReader env m, HasLogFunc env, MonadIO m, DType a)
                   => Text -> NDArray a -> NDArray a -> m ()
        checkShape name arr1 arr2 = do
            shp1 <- liftIO $ ndshape $ arr1
            shp2 <- liftIO $ ndshape $ arr2
            when (shp1 /= shp2) $
                logWarn . display $ sformat
                    ("variable (" % stext %
                     ") has shape " % stext %
                     ", different from that in saved state " % stext %
                     ".") name (tshow shp2) (tshow shp1)

lastSavedState :: MonadIO m => Text -> Text -> m (Maybe FilePath)
lastSavedState dir prefix = liftIO $ do
    let sdir = T.unpack dir
    files <- listDirectory sdir
    let match name = T.isSuffixOf ".params" name && T.isPrefixOf prefix name
        param_files = filter (match . T.pack) files
    if null param_files
        then return Nothing
        else do
            mod_time <- mapM (getModificationTime . (sdir </>)) param_files
            case lastMaybe $ sortOn snd (zip param_files mod_time) of
                Nothing -> return Nothing
                Just (latest, _) -> return $ Just $ sdir </> dropExtension latest

