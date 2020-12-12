{-# LANGUAGE FlexibleInstances      #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses  #-}
{-# LANGUAGE TemplateHaskell        #-}
module MXNet.NN.Session where

import           Data.Kind            (Constraint)
import qualified Data.Type.Index      as DT
import qualified Data.Type.Product    as DT
import qualified GHC.TypeLits         as L
import           RIO
import qualified RIO.State            as ST

import           MXNet.Base
import           MXNet.NN.TaggedState (liftSub, toPair)
import           MXNet.NN.Types       (Module, ModuleSet, ModuleState,
                                       TaggedModuleState)

class Session (sess :: (* -> *) -> * -> *) (sst :: *) | sess -> sst, sst -> sess where
    type SessionDType sst
    type SessionHasModule sst (t :: L.Symbol) :: Constraint
    runSession    :: sess m r -> sst -> m (r, sst)
    subSession    :: MonadIO m => SessionHasModule sst t => Module t (SessionDType sst) m r -> sess m r
    sessionStates :: MonadIO m => sess m [(String, ModuleState (SessionDType sst))]

class HasSessionRef e s | e -> s where
    sessionRefL :: Lens' e (MVar s)

askSession :: (MonadIO m, MonadReader e m, HasSessionRef e s, Session sess s)
           => sess m r -> m r
askSession proc = do
    env <- ask
    let var = env ^. sessionRefL
    st <- liftIO $ takeMVar var
    (rt, st) <- runSession proc st
    liftIO $ putMVar var st
    return rt

withSession :: (MonadIO m, Session sess s)
            => MVar s -> sess m r -> m r
withSession var proc = do
    st <- liftIO $ takeMVar var
    (rt, st) <- runSession proc st
    liftIO $ putMVar var st
    return rt

instance (DT.Every L.KnownSymbol t) => Session (ModuleSet t a) (DT.Prod (TaggedModuleState a) t) where
    type SessionDType (DT.Prod (TaggedModuleState a) t) = a
    type SessionHasModule (DT.Prod (TaggedModuleState a) t) t' = DT.Elem t t'
    runSession = ST.runStateT
    subSession = liftSub
    sessionStates = walk <$> ST.get
      where
        walk :: DT.Every L.KnownSymbol t => DT.Prod (TaggedModuleState a) t -> [(String, ModuleState a)]
        walk DT.Ø           = []
        walk (v DT.:< rest) = toPair v : walk rest

instance (L.KnownSymbol t) => Session (Module t a) (TaggedModuleState a t) where
    type SessionDType (TaggedModuleState a t) = a
    type SessionHasModule (TaggedModuleState a t) t' = t ~ t'
    runSession = ST.runStateT
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

