{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
module MXNet.NN.Session where

import qualified Control.Monad.State.Strict as ST
import qualified GHC.TypeLits as L
import qualified Data.Type.Product as DT
import qualified Data.Type.Index as DT
import qualified Data.HashMap.Strict as M
import Data.Kind (Constraint)
import Control.Lens ((^.))
import Control.Monad (when)
import Control.Monad.IO.Class (MonadIO)

import MXNet.Base
import MXNet.NN.Types (Module, ModuleSet, ModuleState, TaggedModuleState, Parameter(..), mod_params, mod_symbol)
import MXNet.NN.TaggedState (liftSub, toPair)

class MonadIO sess => Session sess where
    type SessionDType sess
    type SessionState sess
    type SessionHasModule sess (t :: L.Symbol) a :: Constraint

    train :: SessionState sess -> sess r -> IO r
    runModule :: SessionHasModule sess t a => Module t a r -> sess r
    getStates :: sess [(String, ModuleState (SessionDType sess))]

instance DT.Every L.KnownSymbol t => Session (ModuleSet t a) where
    type SessionDType (ModuleSet t a) = a
    type SessionState (ModuleSet t a) = DT.Prod (TaggedModuleState a) t
    type SessionHasModule (ModuleSet t a) t' a' = (DT.Elem t t', a ~ a')
    train = flip ST.evalStateT
    runModule = liftSub
    getStates = walk <$> ST.get
      where
        walk :: DT.Every L.KnownSymbol t => DT.Prod (TaggedModuleState a) t -> [(String, ModuleState a)]
        walk DT.Ø = []
        walk (v DT.:< rem) = toPair v : walk rem

instance L.KnownSymbol t => Session (Module t a) where
    type SessionDType (Module t a) = a
    type SessionState (Module t a) = TaggedModuleState a t
    type SessionHasModule (Module t a) t' a' = (t ~ t', a ~ a')
    train = flip ST.evalStateT
    runModule = id
    getStates = (:[]) . toPair <$> ST.get

class CallbackClass c where
    begOfBatch :: (L.KnownSymbol t, DType a) => Int -> Int -> c -> Module t a ()
    begOfBatch _ _ _ = return ()
    endOfBatch :: (L.KnownSymbol t, DType a) => Int -> Int -> c -> Module t a ()
    endOfBatch _ _ _ = return ()
    begOfEpoch :: (L.KnownSymbol t, DType a) => Int -> Int -> c -> Module t a ()
    begOfEpoch _ _ _ = return ()
    endOfEpoch :: (L.KnownSymbol t, DType a) => Int -> Int -> c -> Module t a ()
    endOfEpoch _ _ _ = return ()
    endOfVal   :: (L.KnownSymbol t, DType a) => Int -> Int -> c -> Module t a ()
    endOfVal   _ _ _ = return ()
data Callback where
    Callback :: CallbackClass a => a -> Callback

instance CallbackClass Callback where
    begOfBatch i n (Callback a) = begOfBatch i n a
    endOfBatch i n (Callback a) = endOfBatch i n a
    begOfEpoch i n (Callback a) = begOfEpoch i n a
    endOfEpoch i n (Callback a) = endOfEpoch i n a
    endOfVal   i n (Callback a) = endOfVal   i n a


saveState :: Bool -> String -> ModuleState a -> IO ()
saveState save_symbol name state = do
    let params = state ^. mod_params
        symbol = state ^. mod_symbol
        modelParams = map getModelParam $ M.toList params
    when save_symbol $ mxSymbolSaveToFile (name ++ ".json") (unSymbol symbol)
    mxNDArraySave (name ++ ".params") modelParams
  where
    getModelParam (key, ParameterI a _) = ("arg:" ++ key, unNDArray a)
    getModelParam (key, ParameterA a) = ("aux:" ++ key, unNDArray a)
