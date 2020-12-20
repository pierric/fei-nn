# fei-nn
This library builds a general neural network solver on top the MXNet raw c-apis and operators.

# Background
The `Symbol` API of MXNet synthesize a symbolic graph of the neural network. To solve such a graph, it is necessary to back every `Symbol` with two `NDArray`, one for forward propagation and one for backward propagation. By calling the API `mxExecutorBind`, the symbolic graph and backing NDArrays are bind together, producing an `Executor`. And with this executor, `mxExecutorForward` and `mxExecutorBackward` can run. By optimization, the the backing NDArrays of the neural network is updated in each iteration.

# DataIter
MXNet provides data iterators. And it can be wrapped in a [Stream](https://hackage.haskell.org/package/streaming) or [Conduit](https://hackage.haskell.org/package/conduit). The [fei-dataiter](https://github.com/pierric/fei-dataiter) provides a implementation.

# `Module` Monad
`Module` is a tagged `StateT` monad, where the internal state `TaggedModuleState` has a hashmap of `NDArray`s to back the symbolic graph, together with a few informations.

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

It returns a initial state for `Module`.

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

# `FeiM` Monad
`FeiM` help to build a training/inference application. It integrates RIO's logging framework, and keeps track of a
`TaggedModuleState`.

## `initSession`
```
initSession :: forall n t x. FloatDType t => SymbolHandle -> Config t -> FeiM t n x ()
```
a wrap-up of `initialize`.

## `runFeiM`
```
runFeiM :: x -> FeiM n t x a -> IO a
```
properly initialize mxnet before running an action and some cleanups before termination.

## `askSession`
```
askSession :: Module n t (FeiM n t x) r -> FeiM n t x r
```
helps to embed a `Module` (training/inference procedure) into `FeiM`.

# Usage
See the examples of [fei-examples](https://github.com/pierric/fei-examples) repository.
