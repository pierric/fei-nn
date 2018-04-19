{-# LANGUAGE TemplateHaskell #-}
module MXNet.NN.Types where

import Control.Lens (makeLenses)
import qualified Data.HashMap.Strict as M
import qualified Control.Monad.State.Strict as ST
import MXNet.Core.Base hiding (bind, context, (^.))

-- | A parameter is two 'NDArray' to back a 'Symbol'
data Parameter a = Parameter { _param_in :: NDArray a, _param_grad :: NDArray a }
    deriving Show

-- | Session is all the 'Parameters' and a 'Context'
-- type Session a = (M.HashMap String (Parameter a), Context)
data Session a = Session { _sess_param :: !(M.HashMap String (Parameter a)), _sess_context :: !Context }
makeLenses ''Session
-- | TrainM is a 'StateT' monad
type TrainM a m = ST.StateT (Session a) m

-- | Initializer is about how to create a NDArray from a given shape. 
-- 
-- Usually, it can be a wrapper of MXNet operators, such as @random_uniform@, @random_normal@, 
-- @random_gamma@, etc..
type Initializer a = Context -> [Int] -> IO (NDArray a)