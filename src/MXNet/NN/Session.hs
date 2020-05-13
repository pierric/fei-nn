{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
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

class Session (sess :: (* -> *) -> * -> *) (sst :: *) | sess -> sst, sst -> sess where
    type SessionDType sst
    type SessionHasModule sst (t :: L.Symbol) :: Constraint
    withSession   :: MonadIO m => MVar sst -> sess m r -> m r
    subSession    :: MonadIO m => SessionHasModule sst t => Module t (SessionDType sst) m r -> sess m r
    sessionStates :: MonadIO m => sess m [(String, ModuleState (SessionDType sst))]

instance (DT.Every L.KnownSymbol t) => Session (ModuleSet t a) (DT.Prod (TaggedModuleState a) t) where
    type SessionDType (DT.Prod (TaggedModuleState a) t) = a
    type SessionHasModule (DT.Prod (TaggedModuleState a) t) t' = DT.Elem t t'
    withSession var proc = do
        st <- liftIO $ takeMVar var
        (rt, st) <- ST.runStateT proc st
        liftIO $ putMVar var st
        return rt
    subSession = liftSub
    sessionStates = walk <$> ST.get
      where
        walk :: DT.Every L.KnownSymbol t => DT.Prod (TaggedModuleState a) t -> [(String, ModuleState a)]
        walk DT.Ø = []
        walk (v DT.:< rest) = toPair v : walk rest

instance (L.KnownSymbol t) => Session (Module t a) (TaggedModuleState a t) where
    type SessionDType (TaggedModuleState a t) = a
    type SessionHasModule (TaggedModuleState a t) t' = t ~ t'
    withSession var proc = do
        st <- liftIO $ takeMVar var
        (rt, st) <- ST.runStateT proc st
        liftIO $ putMVar var st
        return rt
    subSession = id
    sessionStates = (:[]) . toPair <$> ST.get

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

