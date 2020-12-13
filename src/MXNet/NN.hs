{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TemplateHaskell       #-}
module MXNet.NN (
    module MXNet.NN.Module,
    module MXNet.NN.Optimizer,
    module MXNet.NN.LrScheduler,
    module MXNet.NN.EvalMetric,
    module MXNet.NN.Layer,
    module MXNet.NN.Types,
    module MXNet.NN.Utils,
    module MXNet.NN.Utils.Repa,
    module MXNet.NN.Utils.GraphViz,
    module MXNet.NN.TaggedState,
    module MXNet.NN.Session,
    module MXNet.NN.Callback,
    module MXNet.NN.DataIter.Class,
    FeiApp,
    FeiM,
    fa_log_func,
    fa_process_context,
    fa_session,
    fa_extra,
    runFeiM,
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
import           MXNet.NN.Utils.Repa

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

type FeiM t n x a = ReaderT (FeiApp t n x) (ResourceT IO) a


data SessionAlreadyExist = SessionAlreadyExist
    deriving (Typeable, Show)
instance Exception SessionAlreadyExist


runFeiM :: x -> FeiM n t x a -> IO a
runFeiM x body = do
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


initSession :: forall n t x. FloatDType t => SymbolHandle -> Config t -> FeiM t n x ()
initSession sym cfg = do
    sess_ref <- view $ fa_session
    liftIO $ do
        sess <- initialize sym cfg
        succ <- tryPutMVar sess_ref sess
        when (not succ) $ throwM SessionAlreadyExist
