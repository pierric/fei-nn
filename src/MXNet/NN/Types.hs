{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
module MXNet.NN.Types where

import Control.Lens (makeLenses)
import qualified Data.HashMap.Strict as M
import Control.Exception.Base (Exception)
import Data.Typeable (Typeable)
import qualified Control.Monad.State.Strict as ST
import qualified Data.Type.Product as DT

import MXNet.Base
import MXNet.NN.TaggedState (Tagged)

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
    _cfg_data                :: M.HashMap String [Int],
    _cfg_label               :: [String],
    _cfg_initializers        :: M.HashMap String (Initializer a),
    _cfg_default_initializer :: Initializer a,
    _cfg_context             :: Context
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
         | DatasetOfUnknownBatchSize
         | LoadSessionInvalidTensorName String
         | LoadSessionMismatchedTensorKind String
    deriving (Show, Typeable)
instance Exception Exc

type TaggedModuleState a = Tagged (ModuleState a)
type Module tag a = ST.StateT (TaggedModuleState a tag) IO
type ModuleSet tags a = ST.StateT (DT.Prod (TaggedModuleState a) tags) IO

data ModuleState a = ModuleState {
      _mod_symbol       :: Symbol a
    , _mod_input_shapes :: M.HashMap String [Int]
    , _mod_params       :: M.HashMap String (Parameter a)
    , _mod_context      :: Context
    , _mod_executor     :: Executor a
    , _mod_statistics   :: Statistics
    , _mod_scores       :: M.HashMap String Double
}

-- | A parameter is two 'NDArray' to back a 'Symbol'
data Parameter a = ParameterI { _param_in :: NDArray a, _param_grad :: Maybe (NDArray a) }
                 | ParameterA { _param_aux :: NDArray a }
  deriving Show

data Statistics = Statistics {
    _stat_num_upd :: !Int,
    _stat_last_lr :: !Float
}

makeLenses ''Statistics
makeLenses ''ModuleState
makeLenses ''Config

