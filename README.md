# fei-nn
This library builds a general neural network solver on top of the MXNet raw c-apis and operators.

# Background
The `Symbol` API of MXNet synthesizes a symbolic graph of the neural network. To solve such a graph, it is necessary to back every `Symbol` with two `NDArray`, one for forwarding propagation and one for backward propagation. By calling the API `mxExecutorBind`, the symbolic graph and backing NDArrays are bound together, producing an `Executor`. And with this executor, `mxExecutorForward` and `mxExecutorBackward` can run. By optimization, the backing NDArrays of the neural network is updated in each iteration.

# DataIter
MXNet provides data iterators. And it can be wrapped in a [Stream](https://hackage.haskell.org/package/streaming) or [Conduit](https://hackage.haskell.org/package/conduit). The [fei-dataiter](https://github.com/pierric/fei-dataiter) provides an implementation.

# `Module` Monad
`Module` is a tagged `StateT` monad, where the internal state `TaggedModuleState` has a hashmap of `NDArray`s to back the symbolic graph, together with a few other pieces of information.

## `initialize`
```
initialize :: forall tag dty. (HasCallStack, FloatDType dty)
            => SymbolHandle
            -> Config dty
            -> IO (TaggedModuleState dty tag)` 
```

It takes the symbolic graph, and a configuration of 
1) shapes of the placeholder in the training phase.
2) how to initialize the NDArrays

It returns an initial state for `Module`.

## `fit`
```
fit :: (FloatDType dty, MonadIO m)
    => M.HashMap Text (NDArray dty)
    -> Module tag dty m ()
```

Given bindings of variables, `fit` carries out a complete forward/backward.

## `forwardOnly`
```
forwardOnly :: (FloatDType dty, MonadIO m)
            => M.HashMap Text (NDArray dty)
            -> Module tag dty m [NDArray dty]
```

Given bindings of variables, `forwardOnly` carries out a forward phase only, returning the output of the neural network.

## `fitAndEval`
```
fitAndEval :: (FloatDType dty, Optimizer opt, EvalMetricMethod mtr, MonadIO m)
           => opt dty
           -> M.HashMap Text (NDArray dty)
           -> MetricData mtr dty
           -> Module tag dty m ()
```

Fit the neural network, update gradients, and then evaluate and record metrics.

# Writing a full training/inference application
It is possible to write a training loop with the above `Module`. But it is still a bit difficult to connect with other code pieces such as data loading, logging/debugging. The major obstacle is that `Module` has a `StateT` monad under the hood, which rules out the chance to be work in places requiring a `MonadUnliftIO`.

Therefore, we embed the `Module`'s state in a top-level enviornment `FeiApp`. An appplication will be written in a `ReaderT` monad, and call `askSession` when needed to "enter" the `Module` monad.

```haskell
data FeiApp t n x = FeiApp
    { _fa_log_func        :: !LogFunc
    , _fa_process_context :: !ProcessContext
    , _fa_session         :: MVar (TaggedModuleState t n)
    , _fa_extra           :: x
    }

initSession :: forall n t m x. (FloatDType t, Feiable m, MonadIO m, MonadReader (FeiApp t n x) m)
            => SymbolHandle -> Config t -> m ()

askSession :: (MonadIO m, MonadReader e m, HasSessionRef e s, Session sess s)
           => sess m r -> m r
```

There are two pre-made top-level `ReaderT` monads. The `SimpleFeiM` uses `FeiApp` without extra infomation, while `NeptFeiM` holds an extra `NeptExtra` data structure. As name suggests, `NeptFeiM` is augmented with the capability to record logs to netpune.

```haskell
newtype SimpleFeiM t n a = SimpleFeiM (ReaderT (FeiApp t n ()) (ResourceT IO) a)
    deriving (Functor, Applicative, Monad, MonadIO, MonadFail)

newtype NeptFeiM t n a = NeptFeiM (ReaderT (FeiApp t n NeptExtra) (ResourceT IO) a)
    deriving (Functor, Applicative, Monad, MonadIO, MonadFail)
```

The type class `Feiable` is then invented to unify the common interface of `SimpleFeiM` and `NeptFeiM`, and possibility support future extension as well. `runFeiM` is supposed to properly initialize mxnet before running action and cleanup before termination.

```haskell
class Feiable (m :: * -> *) where
    data FeiMType m a :: *
    runFeiM :: FeiMType m a -> IO a
```

# Usage
See the examples of [fei-examples](https://github.com/pierric/fei-examples) repository.
