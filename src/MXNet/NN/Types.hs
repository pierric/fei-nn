{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE ExplicitForAll #-}
module MXNet.NN.Types where

import Control.Lens (makeLenses)
import qualified Data.HashMap.Strict as M
import qualified Control.Monad.State.Strict as ST
import Control.Exception.Base (Exception)
import Data.Typeable (Typeable)
import Control.Monad.IO.Class (MonadIO)

import MXNet.Base

-- | A parameter is two 'NDArray' to back a 'Symbol'
data Parameter a = ParameterI { _param_in :: NDArray a, _param_grad :: Maybe (NDArray a) }
                 | ParameterA { _param_aux :: NDArray a }
    -- deriving Show

data Statistics = Statistics {
    _stat_num_upd :: !Int,
    _stat_last_lr :: !Float
}

class CallbackClass a where
    begOfBatch :: MonadIO m => Int -> Int -> a -> TrainM e m ()
    begOfBatch _ _ _ = return ()
    endOfBatch :: MonadIO m => Int -> Int -> a -> TrainM e m ()
    endOfBatch _ _ _ = return ()
    begOfEpoch :: MonadIO m => Int -> a -> TrainM e m ()
    begOfEpoch _ _ = return ()
    endOfEpoch :: MonadIO m => Int -> a -> TrainM e m ()
    endOfEpoch _ _ = return ()

data Callback where
    Callback :: CallbackClass a => a -> Callback

instance CallbackClass Callback where
    begOfBatch i n (Callback a) = begOfBatch i n a
    endOfBatch i n (Callback a) = endOfBatch i n a
    begOfEpoch n (Callback a)   = begOfEpoch n a
    endOfEpoch n (Callback a)   = endOfEpoch n a

-- | Session is all the 'Parameters' and a 'Device'
-- type Session a = (M.HashMap String (Parameter a), Context)
data Session a = Session {
      _sess_symbol :: Symbol a
    , _sess_placeholders :: M.HashMap String [Int]
    , _sess_param   :: !(M.HashMap String (Parameter a))
    , _sess_context :: !Context
    , _sess_callbacks :: [Callback]
    -- , _sess_prof :: (NominalDiffTime, NominalDiffTime, NominalDiffTime, NominalDiffTime, NominalDiffTime, NominalDiffTime)
}
-- | TrainM is a 'StateT' monad
type TrainM a m = ST.StateT (Session a) (ST.StateT Statistics m)

-- | For every symbol in the neural network, it can be placeholder or a variable.
-- therefore, a Config is to specify the shape of the placeholder and the 
-- method to initialize the variables.
-- 
-- Note that it is not right to specify a symbol as both placeholder and 
-- initializer, although it is tolerated and such a symbol is considered
-- as a variable.
-- 
-- Note that any symbol not specified will be initialized with the 
-- _cfg_default_initializer.
data Config a = Config {
    _cfg_placeholders :: M.HashMap String [Int],
    _cfg_initializers :: M.HashMap String (Initializer a),
    _cfg_default_initializer :: Initializer a,
    _cfg_context :: Context
}

-- | Initializer is about how to create a NDArray from the symbol name and the given shape. 
-- 
-- Usually, it can be a wrapper of MXNet operators, such as @random_uniform@, @random_normal@, 
-- @random_gamma@, etc..
type Initializer a = String -> [Int] -> Context -> IO (NDArray a)

-- | Possible exception in 'TrainM'
data Exc = MismatchedShapeOfSym String [Int] [Int]
         | MismatchedShapeInEval [Int] [Int]
         | NotAParameter String
         | InvalidArgument String
         | InferredShapeInComplete
    deriving (Show, Typeable)
instance Exception Exc

makeLenses ''Statistics
makeLenses ''Session