{-# LANGUAGE CPP                        #-}
{-# LANGUAGE GADTs                      #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses      #-}
{-# LANGUAGE ScopedTypeVariables        #-}
{-# LANGUAGE StandaloneDeriving         #-}
{-# LANGUAGE TemplateHaskell            #-}
module MXNet.NN (
    module MXNet.NN.Module,
    module MXNet.NN.Optimizer,
    module MXNet.NN.LrScheduler,
    module MXNet.NN.EvalMetric,
    module MXNet.NN.Layer,
    module MXNet.NN.Types,
    module MXNet.NN.Utils,
#if USE_REPA
    module MXNet.NN.Utils.Repa,
#endif
    module MXNet.NN.Utils.GraphViz,
    module MXNet.NN.TaggedState,
    module MXNet.NN.Session,
    module MXNet.NN.Callback,
    module MXNet.NN.DataIter.Class,
    FeiApp,
    Feiable(..),
    FeiMType(..),
    fa_log_func,
    fa_process_context,
    fa_session,
    fa_extra,
#ifdef NEPTUNE
    NeptExtra,
    neptLog,
#endif
    initSession,
) where

import           Control.Lens
import           Control.Monad.Trans.Resource
import           RIO                          hiding (view)
import           RIO.Process

import           MXNet.Base
import           MXNet.NN.Callback
import           MXNet.NN.DataIter.Class
import           MXNet.NN.EvalMetric
import           MXNet.NN.Layer
import           MXNet.NN.LrScheduler
import           MXNet.NN.Module
import           MXNet.NN.Optimizer
import           MXNet.NN.Session
import           MXNet.NN.TaggedState
import           MXNet.NN.Types
import           MXNet.NN.Utils
import           MXNet.NN.Utils.GraphViz

#ifdef CPP
import           MXNet.NN.Utils.Repa
#endif

#ifdef NEPTUNE
import           Neptune.Client
import           Neptune.Session              (Experiment, NeptuneSession)
#endif

data FeiApp t n x = FeiApp
    { _fa_log_func        :: !LogFunc
    , _fa_process_context :: !ProcessContext
    , _fa_session         :: MVar (TaggedModuleState t n)
    , _fa_extra           :: x
    }
makeLenses ''FeiApp

instance HasLogFunc (FeiApp t n x) where
    logFuncL = fa_log_func

instance HasSessionRef (FeiApp t n x) (TaggedModuleState t n) where
    sessionRefL = fa_session

newtype SimpleFeiM t n a = SimpleFeiM (ReaderT (FeiApp t n ()) (ResourceT IO) a)
    deriving (Functor, Applicative, Monad, MonadIO, MonadFail)

deriving instance MonadReader (FeiApp t n ()) (SimpleFeiM t n)


runFeiMX :: x -> ReaderT (FeiApp t n x) (ResourceT IO) a -> IO a
runFeiMX x body = do
    -- call mxListAllOpNames can ensure the MXNet itself is properly initialized
    -- i.e. MXNet operators are registered in the NNVM
    void mxListAllOpNames
    logopt  <- logOptionsHandle stdout False
    pcontx  <- mkDefaultProcessContext
    session <- newEmptyMVar
    runResourceT $ do
        _ <- register mxNotifyShutdown
        withLogFunc logopt $ \logfunc ->
            flip runReaderT (FeiApp logfunc pcontx session x) body

class Feiable (m :: * -> *) where
    data FeiMType m a :: *
    runFeiM :: FeiMType m a -> IO a

data SessionAlreadyExist = SessionAlreadyExist
    deriving (Typeable, Show)
instance Exception SessionAlreadyExist

instance Feiable (SimpleFeiM t n) where
    data FeiMType (SimpleFeiM t n) a = Simple (SimpleFeiM t n a)
    runFeiM (Simple (SimpleFeiM body)) = runFeiMX () body

#ifdef NEPTUNE

type NeptExtra = (NeptuneSession, Experiment, Text -> Double -> IO ())

newtype NeptFeiM t n a = NeptFeiM (ReaderT (FeiApp t n NeptExtra) (ResourceT IO) a)
    deriving (Functor, Applicative, Monad, MonadIO, MonadFail)

deriving instance MonadReader (FeiApp t n NeptExtra) (NeptFeiM t n)

instance Feiable (NeptFeiM t n) where
    data FeiMType (NeptFeiM t n) a = WithNept Text (NeptFeiM t n a)
    runFeiM (WithNept project (NeptFeiM body)) = do
        withNept project $ \ nsess nexpt ->
            let logger k v = nlog nexpt k (fromRational (toRational v) :: Double)
             in runFeiMX (nsess, nexpt, logger) body

neptLog :: Text -> Double -> NeptFeiM t n ()
neptLog key value = do
    logger <- view $ fa_extra . _3
    liftIO $ logger key value

#endif

initSession :: forall n t m x. (FloatDType t, Feiable m, MonadIO m, MonadReader (FeiApp t n x) m)
            => SymbolHandle -> Config t -> m ()
initSession sym cfg = do
    sess_ref <- view $ fa_session
    liftIO $ do
        sess <- initialize sym cfg
        succ <- tryPutMVar sess_ref sess
        when (not succ) $ throwM SessionAlreadyExist
