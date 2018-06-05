{-# LANGUAGE TemplateHaskell #-}
module MXNet.NN.Types where

import Control.Lens (makeLenses)
import qualified Data.HashMap.Strict as M
import qualified Control.Monad.State.Strict as ST
import Control.Exception.Base (Exception)
import Data.Typeable (Typeable)
import MXNet.Core.Base hiding (bind, context, (^.))

-- | A parameter is two 'NDArray' to back a 'Symbol'
data Parameter a = ParameterI { _param_in :: NDArray a, _param_grad :: NDArray a }
                 | ParameterA { _param_aux :: NDArray a }
    deriving Show

-- | Session is all the 'Parameters' and a 'Context'
-- type Session a = (M.HashMap String (Parameter a), Context)
data Session a = Session { _sess_param :: !(M.HashMap String (Parameter a)), _sess_context :: !Context }
makeLenses ''Session
-- | TrainM is a 'StateT' monad
type TrainM a m = ST.StateT (Session a) m

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
         | InvalidArgument String
    deriving (Show, Typeable)
instance Exception Exc
