{-# LANGUAGE UndecidableInstances #-}
module MXNet.NN.Layer (
  Layer,
  runLayerBuilder,
  dumpCurrentScope,
  sequential, sequential',
  unique, unique',
  named,
  subscope, subscope_named, subscope_next_name,
  prim,
  variable,
  convolution,
  fullyConnected,
  pooling,
  activation,
  softmaxoutput,
  softmax,
  batchnorm,
  cast,
  plus,
  flatten,
  identity,
  dropout,
  reshape,
  add_, sub_, mul_, div_,
) where

import qualified Data.UUID                   as UUID
import qualified Data.UUID.V4                as UUID
import           Formatting                  (formatToString, int, shown, stext,
                                              (%))
import           RIO
import qualified RIO.State                   as ST
import qualified RIO.Text                    as RT
import           System.IO.Unsafe            (unsafePerformIO)

import           MXNet.Base
import qualified MXNet.Base.Operators.Symbol as S

class Show nb => NameBuilder nb where
    nextName :: MonadIO m => nb -> m Text

type Layer = ST.StateT [(Maybe Text, SomeNameBuilder)] IO

runLayerBuilder :: Layer a -> IO a
runLayerBuilder = flip ST.evalStateT []

data SomeNameBuilder = forall nb . (NameBuilder nb) => SomeNameBuilder nb

instance Show SomeNameBuilder where
    show (SomeNameBuilder nb) = show nb

instance NameBuilder SomeNameBuilder where
    nextName (SomeNameBuilder nb) = nextName nb

data UUIDNameBuilder = UUIDNameBuilder

instance Show UUIDNameBuilder where
    show _ = "UUID"

instance NameBuilder UUIDNameBuilder where
    nextName _ = do
        uuid <- liftIO $ UUID.nextRandom
        return $ UUID.toText uuid

newtype SequNameBuilder = SequNameBuilder (IORef Int)

instance Show SequNameBuilder where
    show (SequNameBuilder ref) =
        let idx = unsafePerformIO (readIORef ref)
        in formatToString ("Seq:" % int) idx

instance NameBuilder SequNameBuilder where
    nextName (SequNameBuilder ref) = do
        n <- liftIO $ readIORef ref
        liftIO $ writeIORef ref (n+1)
        return (tshow n)

data OnceNameBuilder = OnceNameBuilder Text (IORef Bool)

instance Show OnceNameBuilder where
    show (OnceNameBuilder n ref) =
        let flag = unsafePerformIO (readIORef ref)
        in formatToString ("Once:" % stext % "[" % shown % "]") n flag

instance NameBuilder OnceNameBuilder where
    nextName (OnceNameBuilder name flag) = do
        fresh <- readIORef flag
        if fresh
        then do
            writeIORef flag False
            return name
        else throwString (formatToString ("name \"" % stext % "\" has been used.") name)

dumpCurrentScope :: Layer Text
dumpCurrentScope = do
    scopes <- ST.get
    return $ tshow scopes

sequential :: Text -> Layer a -> Layer a
sequential name mk = do
    nb <- liftIO $ SequNameBuilder <$> newIORef 0
    subscope (Just name, SomeNameBuilder nb) mk

sequential' :: Layer a -> Layer a
sequential' mk = do
    nb <- liftIO $ SequNameBuilder <$> newIORef 0
    subscope (Nothing, SomeNameBuilder nb) mk

unique :: Text -> Layer a -> Layer a
unique name = subscope (Just name, SomeNameBuilder UUIDNameBuilder)

unique' :: Layer a -> Layer a
unique' = subscope (Nothing, SomeNameBuilder UUIDNameBuilder)

named :: Text -> Layer a -> Layer a
named name mk = do
    scopes <- ST.get
    fresh <- newIORef True
    ST.put ((Nothing, SomeNameBuilder (OnceNameBuilder name fresh)) : scopes)
    a <- mk
    ST.put scopes
    return a

getNextName :: Layer Text
getNextName = do
    ((_, nb) : _) <- ST.get
    nextName nb

getNextNamePrefixed :: Layer Text
getNextNamePrefixed = do
    name <- getNextName
    getNamePrefixed (Just name)

getNamePrefixed :: Maybe Text -> Layer Text
getNamePrefixed name = do
    scopes <- ST.get
    let comps = catMaybes $ reverse (map fst scopes) ++ [name]
    return $ RT.intercalate "." comps

subscope :: (Maybe Text, SomeNameBuilder) -> Layer a -> Layer a
subscope scope mk = do
    old_scopes <- ST.get
    ST.put (scope : old_scopes)
    a <- mk
    ST.put old_scopes
    return a

subscope_named :: Text -> Layer a -> Layer a
subscope_named name = subscope (Just name, SomeNameBuilder UUIDNameBuilder)

subscope_next_name :: Layer a -> Layer a
subscope_next_name mk = do
    name <- getNextName
    subscope_named name mk

variable :: Text -> Layer SymbolHandle
variable name = getNamePrefixed (Just name) >>= liftIO . mxSymbolCreateVariable

prim :: (Text -> args -> IO SymbolHandle) -> args -> Layer SymbolHandle
prim op args = getNextNamePrefixed >>= liftIO . flip op args

convolution :: (HasArgs "_Convolution(symbol)" args '["kernel", "num_filter", "data", "stride", "dilate", "pad", "num_group", "workspace", "layout", "cudnn_tune", "cudnn_off", "no_bias"]
               ,WithoutArgs "_Convolution(symbol)" args '["bias", "weight"])
            => ArgsHMap "_Convolution(symbol)" args -> Layer SymbolHandle
convolution args = subscope_next_name $ do
    b <- variable "bias"
    w <- variable "weight"

    name <- getNamePrefixed Nothing
    if args !? #no_bias == Just True
      then
        liftIO $ S._Convolution name (#weight := w .& args)
      else
        liftIO $ S._Convolution name (#bias := b .& #weight := w .& args)

fullyConnected :: (HasArgs "_FullyConnected(symbol)" args '["flatten", "no_bias", "data", "num_hidden"]
                  ,WithoutArgs "_FullyConnected(symbol)" args '["bias", "weight"])
              => ArgsHMap "_FullyConnected(symbol)" args -> Layer SymbolHandle
fullyConnected args = subscope_next_name $ do
    b <- variable "bias"
    w <- variable "weight"

    name <- getNamePrefixed Nothing
    if args !? #no_bias == Just True
    then
        liftIO $ S._FullyConnected name (#weight := w .& args)
    else
        liftIO $ S._FullyConnected name (#bias := b .& #weight := w .& args)

-- 1.0.0 pooling :: HasArgs "_Pooling(symbol)" args '["data", "kernel", "pool_type", "stride", "pad", "pooling_convention", "global_pool", "cudnn_off"]
-- 1.4.0 pooling :: HasArgs "_Pooling(symbol)" args '["data", "kernel", "pool_type", "stride", "pad", "pooling_convention", "global_pool", "cudnn_off", "p_value", "count_include_pad"]
-- 1.5.0
pooling :: HasArgs "_Pooling(symbol)" args '["data", "kernel", "pool_type", "stride", "pad", "pooling_convention", "global_pool", "cudnn_off", "p_value", "count_include_pad", "layout"]
        => ArgsHMap "_Pooling(symbol)" args -> Layer SymbolHandle
pooling = prim S._Pooling

activation :: HasArgs "_Activation(symbol)" args '["data", "act_type"]
        => ArgsHMap "_Activation(symbol)" args -> Layer SymbolHandle
activation = prim S._Activation

softmax :: Fullfilled "softmax(symbol)" args
        => ArgsHMap "softmax(symbol)" args -> Layer SymbolHandle
softmax = prim S.softmax

softmaxoutput :: Fullfilled "_SoftmaxOutput(symbol)" args
        => ArgsHMap "_SoftmaxOutput(symbol)" args -> Layer SymbolHandle
softmaxoutput = prim S._SoftmaxOutput

batchnorm :: HasArgs "_BatchNorm(symbol)" args '["data", "eps", "momentum", "fix_gamma", "use_global_stats", "output_mean_var", "axis", "cudnn_off", "min_calib_range", "max_calib_range"]
          => ArgsHMap "_BatchNorm(symbol)" args -> Layer SymbolHandle
batchnorm args = subscope_next_name $ do
    gamma    <- variable "gamma"
    beta     <- variable "beta"
    mov_mean <- variable "running_mean"
    mov_var  <- variable "running_var"

    name <- getNamePrefixed Nothing
    liftIO $ S._BatchNorm name (#gamma := gamma
                             .& #beta := beta
                             .& #moving_mean := mov_mean
                             .& #moving_var := mov_var
                             .& args)

cast :: HasArgs "_Cast(symbol)" args '["data", "dtype"]
    => ArgsHMap "_Cast(symbol)" args -> Layer SymbolHandle
cast = prim S._Cast

plus :: HasArgs "elemwise_add(symbol)" args '["lhs", "rhs"]
    => ArgsHMap "elemwise_add(symbol)" args -> Layer SymbolHandle
plus = prim S.elemwise_add

flatten :: HasArgs "_Flatten(symbol)" args '["data"]
    => ArgsHMap "_Flatten(symbol)" args -> Layer SymbolHandle
flatten = prim S._Flatten

identity :: HasArgs "_copy(symbol)" args '["data"]
    => ArgsHMap "_copy(symbol)" args -> Layer SymbolHandle
identity = prim S._copy

-- 1.4.0 dropout :: HasArgs "_Dropout(symbol)" args '["data", "mode", "p", "axes"]
-- 1.5.0
dropout :: HasArgs "_Dropout(symbol)" args '["data", "mode", "p", "axes", "cudnn_off"]
    => ArgsHMap "_Dropout(symbol)" args -> Layer SymbolHandle
dropout = prim S._Dropout

reshape :: (HasArgs "_Reshape(symbol)" args '["data", "shape", "reverse"]
           ,WithoutArgs "_Reshape(symbol)" args '["target_shape", "keep_highest"])
    => ArgsHMap "_Reshape(symbol)" args -> Layer SymbolHandle
reshape = prim S._Reshape

add_ :: SymbolHandle -> SymbolHandle -> Layer SymbolHandle
add_ a b = prim S.elemwise_add (#lhs := a .& #rhs := b .& Nil)

sub_ :: SymbolHandle -> SymbolHandle -> Layer SymbolHandle
sub_ a b = prim S.elemwise_sub (#lhs := a .& #rhs := b .& Nil)

mul_ :: SymbolHandle -> SymbolHandle -> Layer SymbolHandle
mul_ a b = prim S.elemwise_mul (#lhs := a .& #rhs := b .& Nil)

div_ :: SymbolHandle -> SymbolHandle -> Layer SymbolHandle
div_ a b = prim S.elemwise_div (#lhs := a .& #rhs := b .& Nil)

