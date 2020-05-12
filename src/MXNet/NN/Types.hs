{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
module MXNet.NN.Types where

import RIO
import RIO.HashMap (HashMap)
import RIO.HashSet (HashSet)
import qualified RIO.State as ST
import Control.Lens (makeLenses)
import Data.Typeable (Typeable)
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
    _cfg_data                :: HashMap Text FShape,
    _cfg_label               :: [Text],
    _cfg_initializers        :: HashMap Text (Initializer a),
    _cfg_default_initializer :: Initializer a,
    _cfg_fixed_params        :: HashSet Text,
    _cfg_context             :: Context
}

-- | Initializer is about how to create a NDArray from the symbol name and the given shape.
--
-- Usually, it can be a wrapper of MXNet operators, such as @random_uniform@, @random_normal@,
-- @random_gamma@, etc..
type Initializer a = Text -> NonEmpty Int -> Context -> IO (NDArray a)

-- | Possible exception in 'TrainM'
data Exc = MismatchedShapeOfSym Text (NonEmpty Int) (NonEmpty Int)
         | MismatchedShapeInEval (NonEmpty Int) (NonEmpty Int)
         | NotAParameter Text
         | InvalidArgument Text
         | InferredShapeInComplete
         | DatasetOfUnknownBatchSize
         | LoadSessionInvalidTensorName Text
         | LoadSessionMismatchedTensorKind Text
    deriving (Show, Typeable)
instance Exception Exc

type TaggedModuleState a = Tagged (ModuleState a)
type Module tag a m = ST.StateT (TaggedModuleState a tag) m
type ModuleSet tags a m = ST.StateT (DT.Prod (TaggedModuleState a) tags) m

data ModuleState a = ModuleState {
      _mod_symbol       :: Symbol a
    , _mod_input_shapes :: HashMap Text (NonEmpty Int)
    , _mod_params       :: HashMap Text (Parameter a)
    , _mod_context      :: Context
    , _mod_executor     :: Executor a
    , _mod_statistics   :: Statistics
    , _mod_scores       :: HashMap Text Double
    , _mod_fixed_args   :: HashSet Text
}

-- | A parameter is two 'NDArray' to back a 'Symbol'
data Parameter a = ParameterV { _param_var :: NDArray a }
                 | ParameterF { _param_arg :: NDArray a }
                 | ParameterG { _param_arg :: NDArray a, _param_grad :: NDArray a }
                 | ParameterA { _param_aux :: NDArray a }
  deriving Show

data Statistics = Statistics {
    _stat_num_upd :: !Int,
    _stat_last_lr :: !Float
}

makeLenses ''Statistics
makeLenses ''ModuleState
makeLenses ''Config

