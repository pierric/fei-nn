{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
module MXNet.NN.Types where

import Control.Lens (makeLenses)
import qualified Data.HashMap.Strict as M
import qualified Control.Monad.State.Strict as ST
import Control.Exception.Base (Exception)
import Data.Typeable (Typeable)
import Data.Dynamic (Dynamic)
import Control.Monad.IO.Class (MonadIO)

import MXNet.Base

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

makeLenses ''Config

