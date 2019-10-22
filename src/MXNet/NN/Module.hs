module MXNet.NN.Module where

import qualified Data.HashMap.Strict as M
import Control.Monad.State (StateT)
import Control.Arrow (second, Kleisli(..))
import Control.Lens.Setter ((.=))

type Module tag dty = StateT (Tagged tag (ModuleState dty)) IO

data ModuleState dty = ModuleState {
      _mod_symbol     :: Symbol dty
    , _mod_shapes     :: M.HashMap String [Int] 
    , _mod_params     :: M.HashMap String (Parameter dty)
    , _mod_context    :: Context
    , _mod_executor   :: Executor dty
    , _mod_statistics :: Statistics
}

adapt inputs = do
    symbol  <- use mod_symbol
    exec    <- use mod_executor
    context <- use mod_context
    shapes  <- use mod_shapes

    shapes' <- lift $ mapM (runKleisli $ second $ Kleisli ndshape) $ M.toList inputs
    when (shapes /= shapes') $ do
        (args, grads, auxs, exec) <- lift $ execReshapeEx exec True True context shapes
        arg_names <- listArguments
        let arg_ndarrs = M.fromList $ zip arg_names $ map (uncurry . ParameterI) (zip args grads)
            aux_ndarrs = M.fromList $ zip aux_names $ map ParameterA auxs
        aux_names <- listAuxiliaryStates
        mod_executor .= exec
        mod_shapes   .= shapes'
        mod_params   .= M.union arg_ndarrs aux_ndarrs

forward :: M.HashMap String (NDArray dty) -> Module tag dty [NDArray dty]
forward inputs = do
    adapt inputs
    exec <- use mod_executor
    lift $ do
        execForward exec False
        execGetOutputs exec

