# mxnet-nn
High level APIs for training a neural network with MXNet in Haskell

# Motivation
The honorable [mxnet-haskell](https://github.com/sighingnow/mxnet-haskell) wraps the C APIs of MXNet. However there is still quite a few routine work to write a neural work solver. This library tries to tackle this complexity, by providing a Monad named `TrainM` and APIs `fit` and `forwardOnly`.

# Background
The `Symbol` API of MXNet synthesize a symbolic graph of the neural network. To solve such a graph, it is necessary to back every `Symbol` with two `NDArray`, one for forward propagation and one for backward propagation. By calling the API `mxExecutorBind`, the symbolic graph and backing NDArrays are bind together, producing an `Executor`. And with this executor, `mxExecutorForward` and `mxExecutorBackward` can run. Also notice that to optimize the neural network is exactly updating the backing NDArrays.

# DataIter
MXNet provides data iterators. And it can be wrapped in a [Stream](https://hackage.haskell.org/package/streaming) or [Conduit](https://hackage.haskell.org/package/conduit). The [mxnet-dataiter](https://github.com/pierric/mxnet-dataiter) provides a implementation.

# Some explanation
## `TrainM` Monad
`TrainM` is simply a `StateT` monad, where the internal state is a store of all the backing `NDArray`s together with a `Context` (CPU/GPU). Both `fit` and `forwardOnly` must be run inside this monad.

## `initialize`
`initialize :: DType a => Symbol a -> Config a -> IO (Session a)` 

It takes the symbolic graph, and a configuration of 
1) shapes of the placeholder in the training phase.
2) how to initialize the NDArrays

It returns a initial session, with it to run the training in `TrainM` Monad.

## `fit`
`fit :: (DType a, MonadIO m, MonadThrow m, Optimizer opt, OptArgsCst opt g)  => opt a g -> Symbol a -> M.HashMap String (NDArray a) -> TrainM a m ()`

Given a optimizer, the symbolic graph, and feeding the placeholders, `fit` carries out a complete forward/backward phase, updating the NDArrays.

## `fitAndEval`
`fitAndEval :: (DType a, MonadIO m, MonadThrow m, Optimizer opt, OptArgsCst opt g, EvalMetricMethod mth) => opt a g -> Symbol a -> M.HashMap String (NDArray a) -> Metric a mth -> TrainM a m ()`

Fit the neural network, and also record the evaluation.

## `forwardOnly`
`forwardOnly ::  (DType a, MonadIO m, MonadThrow m) => Symbol a -> M.HashMap String (Maybe (NDArray a)) -> TrainM a m [NDArray a]`

Given a the symbolic graph, and feeding the placeholders (data with `Just xx`, and label with `Nothing`). `forwardOnly` carries out a forward phase only, returning the output of the neural network.

# Usage
- Please see the example in the `examples` directory.
- Also see the examples of [mxnet-dataiter](https://github.com/pierric/mxnet-dataiter) project.
