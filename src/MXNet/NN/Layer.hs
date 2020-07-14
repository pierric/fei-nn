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
  convolutionShared,
  fullyConnected,
  fullyConnectedShared,
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
  stack,
  concat_,
  splitBySections,
  where_,
  takeI,
  blockGrad,
  add_, sub_, mul_, div_,
  eq_, neq_, gt_, geq_, lt_, leq_,
  addScalar, subScalar, mulScalar, divScalar,
  eqScalar, neqScalar, gtScalar, geqScalar, ltScalar, leqScalar,
  addBroadcast, subBroadcast, mulBroadcast, divBroadcast,
  eqBroadcast, neqBroadcast, gtBroadcast, geqBroadcast, ltBroadcast, leqBroadcast,
  ceil_,
  floor_,
  sqrt_,
  log2_,
  square_,
  zerosLike, onesLike,
  squeeze, expandDims,
  broadcastAxis,
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

convolutionShared :: (HasArgs "_Convolution(symbol)" args
                        '["kernel", "num_filter", "stride",
                          "dilate", "pad", "num_group", "workspace",
                          "layout", "cudnn_tune", "cudnn_off", "no_bias"]
                     ,WithoutArgs "_Convolution(symbol)" args '["data", "bias", "weight"])
                  => ArgsHMap "_Convolution(symbol)" args -> Layer (SymbolHandle -> Layer SymbolHandle)
convolutionShared args = subscope_next_name $ do
    b <- variable "bias"
    w <- variable "weight"

    return $ \data_ -> do
        name <- getNextNamePrefixed
        if args !? #no_bias == Just True
          then
            liftIO $ S._Convolution name (#data := data_ .& #weight := w .& args)
          else
            liftIO $ S._Convolution name (#data := data_ .& #bias := b .& #weight := w .& args)

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

fullyConnectedShared :: (HasArgs "_FullyConnected(symbol)" args
                            '["flatten", "no_bias", "num_hidden"]
                        ,WithoutArgs "_FullyConnected(symbol)" args '["bias", "weight"])
                     => ArgsHMap "_FullyConnected(symbol)" args -> Layer (SymbolHandle -> Layer SymbolHandle)
fullyConnectedShared args = subscope_next_name $ do
    b <- variable "bias"
    w <- variable "weight"

    return $ \data_ -> do
        name <- getNextNamePrefixed
        if args !? #no_bias == Just True
        then
            liftIO $ S._FullyConnected name (#data := data_ .& #weight := w .& args)
        else
            liftIO $ S._FullyConnected name (#data := data_ .& #bias := b .& #weight := w .& args)

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

stack :: (HasArgs "stack(symbol)" args '["data", "axis"]
         ,WithoutArgs "stack(symbol)" args '["num_args"])
      => ArgsHMap "stack(symbol)" args -> Layer SymbolHandle
stack args = prim S.stack (#num_args := len .& args)
    where
        data_ = args !? #data
        len = fromMaybe 0 (length <$> data_)

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

reshape :: [Int] -> SymbolHandle -> Layer SymbolHandle
reshape shape a = prim S._Reshape (#data := a .& #shape := shape .& Nil)

add_ :: SymbolHandle -> SymbolHandle -> Layer SymbolHandle
sub_ :: SymbolHandle -> SymbolHandle -> Layer SymbolHandle
mul_ :: SymbolHandle -> SymbolHandle -> Layer SymbolHandle
div_ :: SymbolHandle -> SymbolHandle -> Layer SymbolHandle

add_ a b = prim S.elemwise_add (#lhs := a .& #rhs := b .& Nil)
sub_ a b = prim S.elemwise_sub (#lhs := a .& #rhs := b .& Nil)
mul_ a b = prim S.elemwise_mul (#lhs := a .& #rhs := b .& Nil)
div_ a b = prim S.elemwise_div (#lhs := a .& #rhs := b .& Nil)

eq_  a b = prim S._equal (#lhs := a .& #rhs := b .& Nil)
neq_ a b = prim S._not_equal (#lhs := a .& #rhs := b .& Nil)
lt_   a b = prim S._lesser (#lhs := a .& #rhs := b .& Nil)
leq_ a b = prim S._lesser_equal (#lhs := a .& #rhs := b .& Nil)
gt_   a b = prim S._greater (#lhs := a .& #rhs := b .& Nil)
geq_ a b = prim S._greater_equal (#lhs := a .& #rhs := b .& Nil)

addScalar b a = prim S._plus_scalar (#data := a .& #scalar := b .& Nil)
subScalar b a = prim S._minus_scalar (#data := a .& #scalar := b .& Nil)
mulScalar b a = prim S._mul_scalar (#data := a .& #scalar := b .& Nil)
divScalar b a = prim S._div_scalar (#data := a .& #scalar := b .& Nil)

eqScalar  b a = prim S._equal_scalar (#data := a .& #scalar := b .& Nil)
neqScalar b a = prim S._not_equal_scalar (#data := a .& #scalar := b .& Nil)
ltScalar  b a = prim S._lesser_scalar (#data := a .& #scalar := b .& Nil)
leqScalar b a = prim S._lesser_equal_scalar (#data := a .& #scalar := b .& Nil)
gtScalar  b a = prim S._greater_scalar (#data := a .& #scalar := b .& Nil)
geqScalar b a = prim S._greater_equal_scalar (#data := a .& #scalar := b .& Nil)

addBroadcast a b = prim S.broadcast_add (#lhs := a .& #rhs := b .& Nil)
subBroadcast a b = prim S.broadcast_sub (#lhs := a .& #rhs := b .& Nil)
mulBroadcast a b = prim S.broadcast_mul (#lhs := a .& #rhs := b .& Nil)
divBroadcast a b = prim S.broadcast_div (#lhs := a .& #rhs := b .& Nil)

eqBroadcast a b = prim S.broadcast_equal (#lhs := a .& #rhs := b .& Nil)
neqBroadcast a b = prim S.broadcast_not_equal (#lhs := a .& #rhs := b .& Nil)
ltBroadcast a b = prim S.broadcast_lesser (#lhs := a .& #rhs := b .& Nil)
leqBroadcast a b = prim S.broadcast_lesser_equal (#lhs := a .& #rhs := b .& Nil)
gtBroadcast a b = prim S.broadcast_greater (#lhs := a .& #rhs := b .& Nil)
geqBroadcast a b = prim S.broadcast_greater_equal (#lhs := a .& #rhs := b .& Nil)

ceil_   a = prim S.ceil   (#data := a .& Nil)
floor_  a = prim S.floor  (#data := a .& Nil)
sqrt_   a = prim S.sqrt   (#data := a .& Nil)
log2_   a = prim S.log2   (#data := a .& Nil)
square_ a = prim S.square (#data := a .& Nil)

blockGrad :: SymbolHandle -> Layer SymbolHandle
blockGrad s = prim S._BlockGrad (#data := s .& Nil)

concat_ :: Int -> [SymbolHandle] -> Layer SymbolHandle
concat_ d s = prim S._Concat (#data := s .& #num_args := length s .& #dim := d .& Nil)

splitBySections :: Int -> Int -> Bool -> SymbolHandle -> Layer [SymbolHandle]
splitBySections num_sections axis squeeze s = do
    r <- prim S._split_v2 (#data := s
                        .& #axis := axis
                        .& #indices := []
                        .& #sections := num_sections
                        .& #squeeze_axis := squeeze .& Nil)
    mapM (at r) ([0..num_sections] :: [Int])

takeI :: SymbolHandle -> SymbolHandle -> Layer SymbolHandle
takeI i a = prim S.take (#a := a .& #indices := i .& Nil)

where_ c a b = prim S._where (#condition := c .& #x := a .& #y := b .& Nil)

zerosLike a = prim S.zeros_like (#data := a .& Nil)
onesLike  a = prim S.ones_like  (#data := a .& Nil)

squeeze axis a = prim S.squeeze (#data := a .& #axis := axis .& Nil)
expandDims axis a = prim S.expand_dims (#data := a .& #axis := axis .& Nil)

broadcastAxis axis size a = prim S.broadcast_axis (#data := a .& #axis := axis .& #size := size .& Nil)
