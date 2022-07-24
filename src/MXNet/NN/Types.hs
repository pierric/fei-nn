{-# LANGUAGE AllowAmbiguousTypes   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ExplicitForAll        #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TemplateHaskell       #-}
module MXNet.NN.Types where

import           Control.Lens         (makeLenses)
import qualified Data.Type.Product    as DT
import           RIO
import           RIO.State            (StateT)

import           MXNet.Base
import           MXNet.NN.TaggedState (Tagged)

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
data Config a = Config
    { _cfg_data                :: HashMap Text Shape
    , _cfg_label               :: [Text]
    , _cfg_initializers        :: HashMap Text (SomeInitializer a)
    , _cfg_default_initializer :: SomeInitializer a
    , _cfg_fixed_params        :: HashSet Text
    , _cfg_context             :: Context
    }

-- | Initializer is to assign a NDArray some value.
--
-- Usually, it can be a wrapper of MXNet operators, such as @random_uniform@, @random_normal@,
-- @random_gamma@, etc..
class DType a => Initializer n a where
    initNDArray :: n a -> Text -> NDArray a -> IO ()

data SomeInitializer a = forall n . Initializer n a => SomeInitializer (n a)

-- | Possible exception in 'TrainM'
data Exc = MismatchedShapeOfSym Text [Int] [Int]
    | MismatchedShapeInEval [Int] [Int]
    | NotAParameter Text
    | InvalidArgument Text
    | InferredShapeInComplete
    | DatasetOfUnknownBatchSize
    | LoadSessionInvalidTensorName Text
    | LoadSessionMismatchedTensorKind Text
    deriving (Show, Typeable)
instance Exception Exc

type TaggedModuleState dty = Tagged (ModuleState dty)
type Module tag dty = StateT (TaggedModuleState dty tag)
type ModuleSet tags dty = StateT (DT.Prod (TaggedModuleState dty) tags)

data ModuleState a = ModuleState
    { _mod_symbol       :: Symbol a
    , _mod_input_shapes :: HashMap Text [Int]
    , _mod_params       :: HashMap Text (Parameter a)
    , _mod_context      :: Context
    , _mod_executor     :: Executor a
    , _mod_statistics   :: Statistics
    , _mod_scores       :: HashMap Text Double
    , _mod_fixed_args   :: HashSet Text
    }

-- | A parameter is two 'NDArray' to back a 'Symbol'
data Parameter a = ParameterV
    { _param_var :: NDArray a
    }
    | ParameterG
    { _param_arg  :: NDArray a
    , _param_grad :: NDArray a
    }
    | ParameterA
    { _param_aux :: NDArray a
    }
    deriving Show

data Statistics = Statistics
    { _stat_num_upd :: !Int
    , _stat_last_lr :: !Float
    }

hasGradient :: Parameter a -> Bool
hasGradient (ParameterG _ _) = True
hasGradient _                = False

isAuxiliary :: Parameter a -> Bool
isAuxiliary (ParameterA _) = True
isAuxiliary _              = False

makeLenses ''Statistics
makeLenses ''ModuleState
makeLenses ''Config

