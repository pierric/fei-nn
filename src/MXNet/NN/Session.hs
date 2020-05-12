{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
module MXNet.NN.Session where

import RIO
import qualified RIO.State as ST
import qualified GHC.TypeLits as L
import qualified Data.Type.Product as DT
import qualified Data.Type.Index as DT
import Data.Kind (Constraint)

import MXNet.Base
import MXNet.NN.Types (Module, ModuleSet, ModuleState, TaggedModuleState)
import MXNet.NN.TaggedState (liftSub, toPair)

class MonadIO sess => Session sess where
    type SessionDType sess
    type SessionState sess
    type SessionMonad sess :: * -> *
    type SessionHasModule sess (t :: L.Symbol) a :: Constraint

    train :: SessionState sess -> sess r -> SessionMonad sess r
    runModule :: SessionHasModule sess t a => Module t a (SessionMonad sess) r -> sess r
    getStates :: sess [(String, ModuleState (SessionDType sess))]

instance (DT.Every L.KnownSymbol t, MonadIO m) => Session (ModuleSet t a m) where
    type SessionDType (ModuleSet t a m) = a
    type SessionState (ModuleSet t a m) = DT.Prod (TaggedModuleState a) t
    type SessionMonad (ModuleSet t a m) = m
    type SessionHasModule (ModuleSet t a m) t' a' = (DT.Elem t t', a ~ a')
    train = flip ST.evalStateT
    runModule = liftSub
    getStates = walk <$> ST.get
      where
        walk :: DT.Every L.KnownSymbol t => DT.Prod (TaggedModuleState a) t -> [(String, ModuleState a)]
        walk DT.Ø = []
        walk (v DT.:< rest) = toPair v : walk rest

instance (L.KnownSymbol t, MonadIO m) => Session (Module t a m) where
    type SessionDType (Module t a m) = a
    type SessionState (Module t a m) = TaggedModuleState a t
    type SessionMonad (Module t a m) = m
    type SessionHasModule (Module t a m) t' a' = (t ~ t', a ~ a')
    train = flip ST.evalStateT
    runModule = id
    getStates = (:[]) . toPair <$> ST.get

class CallbackClass c where
    begOfBatch :: ( L.KnownSymbol t
                  , DType a
                  , MonadIO m
                  , MonadReader env m
                  , HasLogFunc env
                  , HasCallStack)
        => Int -> Int -> c -> Module t a m ()
    begOfBatch _ _ _ = return ()
    endOfBatch :: ( L.KnownSymbol t
                  , DType a
                  , MonadIO m
                  , MonadReader env m
                  , HasLogFunc env
                  , HasCallStack)
        => Int -> Int -> c -> Module t a m ()
    endOfBatch _ _ _ = return ()
    begOfEpoch :: ( L.KnownSymbol t
                  , DType a
                  , MonadIO m
                  , MonadReader env m
                  , HasLogFunc env
                  , HasCallStack)
        => Int -> Int -> c -> Module t a m ()
    begOfEpoch _ _ _ = return ()
    endOfEpoch :: ( L.KnownSymbol t
                  , DType a
                  , MonadIO m
                  , MonadReader env m
                  , HasLogFunc env
                  , HasCallStack)
        => Int -> Int -> c -> Module t a m ()
    endOfEpoch _ _ _ = return ()
    endOfVal   :: ( L.KnownSymbol t
                  , DType a
                  , MonadIO m
                  , MonadReader env m
                  , HasLogFunc env
                  , HasCallStack)
        => Int -> Int -> c -> Module t a m ()
    endOfVal   _ _ _ = return ()
data Callback where
    Callback :: CallbackClass a => a -> Callback

instance CallbackClass Callback where
    begOfBatch i n (Callback a) = begOfBatch i n a
    endOfBatch i n (Callback a) = endOfBatch i n a
    begOfEpoch i n (Callback a) = begOfEpoch i n a
    endOfEpoch i n (Callback a) = endOfEpoch i n a
    endOfVal   i n (Callback a) = endOfVal   i n a

