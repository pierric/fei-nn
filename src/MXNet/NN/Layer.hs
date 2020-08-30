{-# LANGUAGE PartialTypeSignatures  #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances   #-}
module MXNet.NN.Layer where

import qualified Data.UUID                   as UUID
import qualified Data.UUID.V4                as UUID
import           Formatting                  (formatToString, int, shown, stext,
                                              (%))
import           RIO
import qualified RIO.State                   as ST
import qualified RIO.Text                    as RT
import           System.IO.Unsafe            (unsafePerformIO)

import           MXNet.Base
import qualified MXNet.Base.Operators.Tensor as S
import qualified MXNet.NN.Types              as S

runLayerBuilder :: Layer a -> IO a
runLayerBuilder = flip ST.evalStateT []

type instance TensorM SymbolHandle  = Layer SymbolHandle

instance PrimTensorOp SymbolHandle where
    prim op args = getNextNamePrefixed >>= liftIO . op args

type Layer = ST.StateT [(Maybe Text, SomeNameBuilder)] IO

class Show nb => NameBuilder nb where
    nextName :: MonadIO m => nb -> m Text

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

convolution :: (HasArgs "_Convolution" SymbolHandle args
                    '["kernel", "num_filter", "data", "stride", "dilate", "pad",
                      "num_group", "workspace", "layout",
                      "cudnn_tune", "cudnn_off", "no_bias"]
               ,WithoutArgs "_Convolution" SymbolHandle args
                    '["bias", "weight"])
            => ArgsHMap "_Convolution" _ args -> Layer SymbolHandle
convolution args = subscope_next_name $ do
    b <- variable "bias"
    w <- variable "weight"

    name <- getNamePrefixed Nothing
    if args !? #no_bias == Just True
      then
        liftIO $ S._Convolution (#weight := w .& args) name
      else
        liftIO $ S._Convolution (#bias := b .& #weight := w .& args) name

convolutionShared :: (HasArgs "_Convolution" SymbolHandle args
                        '["kernel", "num_filter", "stride",
                          "dilate", "pad", "num_group", "workspace",
                          "layout", "cudnn_tune", "cudnn_off", "no_bias"]
                     ,WithoutArgs "_Convolution" SymbolHandle args
                        '["data", "bias", "weight"])
                  => ArgsHMap "_Convolution" _ args -> Layer (SymbolHandle -> Layer SymbolHandle)
convolutionShared args = subscope_next_name $ do
    b <- variable "bias"
    w <- variable "weight"

    return $ \data_ -> do
        name <- getNextNamePrefixed
        if args !? #no_bias == Just True
          then
            liftIO $ S._Convolution (#data := data_ .& #weight := w .& args) name
          else
            liftIO $ S._Convolution (#data := data_ .& #bias := b .& #weight := w .& args) name

fullyConnected :: (HasArgs "_FullyConnected" SymbolHandle args
                    '["flatten", "no_bias", "data", "num_hidden"]
                  ,WithoutArgs "_FullyConnected" SymbolHandle args
                    '["bias", "weight"])
              => ArgsHMap "_FullyConnected" _ args -> Layer SymbolHandle
fullyConnected args = subscope_next_name $ do
    b <- variable "bias"
    w <- variable "weight"

    name <- getNamePrefixed Nothing
    if args !? #no_bias == Just True
    then
        liftIO $ S._FullyConnected (#weight := w .& args) name
    else
        liftIO $ S._FullyConnected (#bias := b .& #weight := w .& args) name

fullyConnectedShared :: (HasArgs "_FullyConnected" SymbolHandle args
                            '["flatten", "no_bias", "num_hidden"]
                        ,WithoutArgs "_FullyConnected" SymbolHandle args
                            '["bias", "weight"])
                     => ArgsHMap "_FullyConnected" _ args -> Layer (SymbolHandle -> Layer SymbolHandle)
fullyConnectedShared args = subscope_next_name $ do
    b <- variable "bias"
    w <- variable "weight"

    return $ \data_ -> do
        name <- getNextNamePrefixed
        if args !? #no_bias == Just True
        then
            liftIO $ S._FullyConnected (#data := data_ .& #weight := w .& args) name
        else
            liftIO $ S._FullyConnected (#data := data_ .& #bias := b .& #weight := w .& args) name

batchnorm :: HasArgs "_BatchNorm" SymbolHandle  args
                '["data", "eps", "momentum", "fix_gamma",
                  "use_global_stats", "output_mean_var", "axis",
                  "cudnn_off", "min_calib_range", "max_calib_range"]
          => ArgsHMap "_BatchNorm" _ args -> Layer SymbolHandle
batchnorm args = subscope_next_name $ do
    gamma    <- variable "gamma"
    beta     <- variable "beta"
    mov_mean <- variable "running_mean"
    mov_var  <- variable "running_var"

    name <- getNamePrefixed Nothing
    liftIO $ S._BatchNorm (#gamma := gamma
                        .& #beta := beta
                        .& #moving_mean := mov_mean
                        .& #moving_var := mov_var
                        .& args) name

blockGrad :: SymbolHandle -> Layer SymbolHandle
blockGrad s = prim S._BlockGrad (#data := s .& Nil)

splitBySections :: HasCallStack => Int -> Int -> Bool -> SymbolHandle -> Layer [SymbolHandle]
splitBySections num_sections axis squeeze s = do
    r <- prim S.__split_v2 (#data := s
                        .& #axis := axis
                        .& #indices := []
                        .& #sections := num_sections
                        .& #squeeze_axis := squeeze .& Nil)
    mapM (at r) ([0..num_sections-1] :: [Int])

-----------------------------------------------------------------------------
-- For both Symbol and NDArray
-----------------------------------------------------------------------------

pooling :: (PrimTensorOp t, Fullfilled "_Pooling" t args)
        => ArgsHMap "_Pooling" t args -> TensorM t
pooling = prim S._Pooling

activation :: (PrimTensorOp t, Fullfilled "_Activation" t args)
           => ArgsHMap "_Activation" t args -> TensorM t
activation = prim S._Activation

softmax :: (PrimTensorOp t, Fullfilled "_softmax" t args)
        => ArgsHMap "_softmax" t args -> TensorM t
softmax = prim S._softmax

softmaxoutput :: (PrimTensorOp t, Fullfilled "_SoftmaxOutput" t args)
              => ArgsHMap "_SoftmaxOutput" t args -> TensorM t
softmaxoutput = prim S._SoftmaxOutput

pick :: (PrimTensorOp t, Fullfilled "_pick" t args)
     => ArgsHMap "_pick" t args -> TensorM t
pick = prim S._pick

cast dt t = prim S._Cast (#dtype := dt .& #data := t .& Nil)
stack axis ts = prim S._stack (#num_args := length ts .& #data := ts .& Nil)
flatten t = prim S._Flatten (#data := t .& Nil)
identity s = prim S.__copy (#data := s .& Nil)
dropout t p = prim S._Dropout (#data := t .& #p := p .& Nil)
reshape shape a = prim S._Reshape (#data := a .& #shape := shape .& Nil)

add_, sub_, mul_, div_, eq_, neq_, lt_, leq_, gt_, geq_ ::
    PrimTensorOp t => t -> t -> TensorM t
add_ a b = prim S._elemwise_add (#lhs := a .& #rhs := b .& Nil)
sub_ a b = prim S._elemwise_sub (#lhs := a .& #rhs := b .& Nil)
mul_ a b = prim S._elemwise_mul (#lhs := a .& #rhs := b .& Nil)
div_ a b = prim S._elemwise_div (#lhs := a .& #rhs := b .& Nil)

eq_   a b = prim S.__equal (#lhs := a .& #rhs := b .& Nil)
neq_  a b = prim S.__not_equal (#lhs := a .& #rhs := b .& Nil)
lt_   a b = prim S.__lesser (#lhs := a .& #rhs := b .& Nil)
leq_  a b = prim S.__lesser_equal (#lhs := a .& #rhs := b .& Nil)
gt_   a b = prim S.__greater (#lhs := a .& #rhs := b .& Nil)
geq_  a b = prim S.__greater_equal (#lhs := a .& #rhs := b .& Nil)

addScalar  b a = prim S.__plus_scalar (#data := a .& #scalar := b .& Nil)
subScalar  b a = prim S.__minus_scalar (#data := a .& #scalar := b .& Nil)
rsubScalar b a = prim S.__rminus_scalar (#data := a .& #scalar := b .& Nil)
mulScalar  b a = prim S.__mul_scalar (#data := a .& #scalar := b .& Nil)
divScalar  b a = prim S.__div_scalar (#data := a .& #scalar := b .& Nil)
rdivScalar b a = prim S.__rdiv_scalar (#data := a .& #scalar := b .& Nil)

eqScalar  b a = prim S.__equal_scalar (#data := a .& #scalar := b .& Nil)
neqScalar b a = prim S.__not_equal_scalar (#data := a .& #scalar := b .& Nil)
ltScalar  b a = prim S.__lesser_scalar (#data := a .& #scalar := b .& Nil)
leqScalar b a = prim S.__lesser_equal_scalar (#data := a .& #scalar := b .& Nil)
gtScalar  b a = prim S.__greater_scalar (#data := a .& #scalar := b .& Nil)
geqScalar b a = prim S.__greater_equal_scalar (#data := a .& #scalar := b .& Nil)

addBroadcast a b = prim S._broadcast_add (#lhs := a .& #rhs := b .& Nil)
subBroadcast a b = prim S._broadcast_sub (#lhs := a .& #rhs := b .& Nil)
mulBroadcast a b = prim S._broadcast_mul (#lhs := a .& #rhs := b .& Nil)
divBroadcast a b = prim S._broadcast_div (#lhs := a .& #rhs := b .& Nil)

eqBroadcast  a b = prim S._broadcast_equal (#lhs := a .& #rhs := b .& Nil)
neqBroadcast a b = prim S._broadcast_not_equal (#lhs := a .& #rhs := b .& Nil)
ltBroadcast  a b = prim S._broadcast_lesser (#lhs := a .& #rhs := b .& Nil)
leqBroadcast a b = prim S._broadcast_lesser_equal (#lhs := a .& #rhs := b .& Nil)
gtBroadcast  a b = prim S._broadcast_greater (#lhs := a .& #rhs := b .& Nil)
geqBroadcast a b = prim S._broadcast_greater_equal (#lhs := a .& #rhs := b .& Nil)

ceil_   a = prim S._ceil   (#data := a .& Nil)
floor_  a = prim S._floor  (#data := a .& Nil)
sqrt_   a = prim S._sqrt   (#data := a .& Nil)
log2_   a = prim S._log2   (#data := a .& Nil)
square_ a = prim S._square (#data := a .& Nil)

concat_ :: PrimTensorOp t => Int -> [t] -> TensorM t
concat_ d s = prim S._Concat (#data := s .& #num_args := length s .& #dim := d .& Nil)

takeI :: (HasCallStack, PrimTensorOp t)
      => t -> t -> TensorM t
takeI i a = prim S._take (#a := a .& #indices := i .& Nil)

where_ c a b = prim S._where (#condition := c .& #x := a .& #y := b .& Nil)

zerosLike a = prim S._zeros_like (#data := a .& Nil)
onesLike  a = prim S._ones_like  (#data := a .& Nil)

squeeze axis a = prim S._squeeze (#data := a .& #axis := axis .& Nil)
expandDims axis a = prim S._expand_dims (#data := a .& #axis := axis .& Nil)

broadcastAxis axis size a = prim S._broadcast_axis (#data := a .& #axis := axis .& #size := size .& Nil)

sum_ s axis keepdims = prim S._sum (#data := s .& #axis:= axis .& #keepdims := keepdims .& Nil)

transpose a axes = prim S._transpose (#data := a .& #axes := axes .& Nil)

argmax a axis keepdims = prim S._argmax (#data := a .& #axis := axis .& #keepdims := keepdims .& Nil)

slice_axis a axis beg end = prim S._slice_axis (#data := a .& #axis := axis .& #begin := beg .& #end := end .& Nil)
-----------------------------------------------------------------------------
-- For NDArray Only
-----------------------------------------------------------------------------
copy src dst = do
    [ret] <- S.__copyto (#data := src .& Nil) (Just [dst])
    return ret


