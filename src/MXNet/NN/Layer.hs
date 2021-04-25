{-# LANGUAGE MultiParamTypeClasses  #-}
{-# LANGUAGE PartialTypeSignatures  #-}
{-# LANGUAGE ScopedTypeVariables    #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances   #-}
module MXNet.NN.Layer where

import qualified Data.UUID                   as UUID
import qualified Data.UUID.V4                as UUID
import           Formatting                  (formatToString, int, shown, stext,
                                              (%))
import           GHC.TypeLits                (KnownSymbol, symbolVal)
import           RIO
import qualified RIO.NonEmpty                as NE
import qualified RIO.State                   as ST
import qualified RIO.Text                    as RT
import           System.IO.Unsafe            (unsafePerformIO)

import           MXNet.Base
import qualified MXNet.Base.Operators.Tensor as S

runLayerBuilder :: MonadIO m => Layer a -> m a
runLayerBuilder = liftIO . flip ST.evalStateT []

type instance TensorMonad Symbol = Layer

instance PrimTensorOp Symbol where
    prim      op args = getNextNamePrefixed >>= liftIO . op args
    primMulti op args = do
        out <- prim op args
        num <- numOutputs out
        mapM (at out) ([0..num-1] :: [Int])

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

newtype SequNameBuilder
  = SequNameBuilder (IORef Int)

instance Show SequNameBuilder where
    show (SequNameBuilder ref) =
        let idx = unsafePerformIO (readIORef ref)
        in formatToString ("Seq:" % int) idx

instance NameBuilder SequNameBuilder where
    nextName (SequNameBuilder ref) = do
        n <- liftIO $ readIORef ref
        liftIO $ writeIORef ref (n+1)
        return (tshow n)

data OnceNameBuilder
  = OnceNameBuilder Text (IORef Bool)

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

sequential :: HasCallStack => Text -> Layer a -> Layer a
sequential name mk = do
    nb <- liftIO $ SequNameBuilder <$> newIORef 0
    subscope (Just name, SomeNameBuilder nb) mk

sequential' :: HasCallStack => Layer a -> Layer a
sequential' mk = do
    nb <- liftIO $ SequNameBuilder <$> newIORef 0
    subscope (Nothing, SomeNameBuilder nb) mk

unique :: HasCallStack => Text -> Layer a -> Layer a
unique name = subscope (Just name, SomeNameBuilder UUIDNameBuilder)

unique' :: HasCallStack => Layer a -> Layer a
unique' = subscope (Nothing, SomeNameBuilder UUIDNameBuilder)

named :: HasCallStack => Text -> Layer a -> Layer a
named name mk = do
    scopes <- ST.get
    fresh <- newIORef True
    ST.put ((Nothing, SomeNameBuilder (OnceNameBuilder name fresh)) : scopes)
    a <- mk
    ST.put scopes
    return a

getNextName :: HasCallStack => Layer Text
getNextName = do
    scopes <- ST.get
    case scopes of
        ((_, nb) : _) -> nextName nb
        _ -> throwString ("No next name avaiable. The current scopes: " ++ show scopes)

getNextNamePrefixed :: HasCallStack => Layer Text
getNextNamePrefixed = do
    name <- getNextName
    getNamePrefixed (Just name)

getNamePrefixed :: HasCallStack => Maybe Text -> Layer Text
getNamePrefixed name = do
    scopes <- ST.get
    let comps = catMaybes $ reverse (map fst scopes) ++ [name]
    return $ RT.intercalate "." comps

subscope :: HasCallStack => (Maybe Text, SomeNameBuilder) -> Layer a -> Layer a
subscope scope mk = do
    old_scopes <- ST.get
    ST.put (scope : old_scopes)
    a <- mk
    ST.put old_scopes
    return a

subscope_named :: HasCallStack => Text -> Layer a -> Layer a
subscope_named name = subscope (Just name, SomeNameBuilder UUIDNameBuilder)

subscope_next_name :: HasCallStack => Layer a -> Layer a
subscope_next_name mk = do
    name <- getNextName
    subscope_named name mk


parameter :: forall a. (DType a, KnownSymbol (DTypeName a))
          => Text -> ReqType -> Layer (Symbol a)
parameter name grad_req = do
    pn <- getNamePrefixed (Just name)
    liftIO $ do
        sym <- mxSymbolCreateVariable pn
        let dtype = RT.pack $ symbolVal $ (Proxy :: Proxy (DTypeName a))
        setAttr sym "__dtype__" dtype
        setAttr sym "__storage_type__" "default"
        setAttr sym "__grad_req__" rtype
        return $ Symbol sym
    where
        rtype = case grad_req of
                  ReqNull    -> "null"
                  ReqWrite   -> "write"
                  ReqAdd     -> "add"
                  ReqInplace -> "inplace"

variable :: (DType a, KnownSymbol (DTypeName a))
          => Text -> Layer (Symbol a)
variable = flip parameter ReqNull


constant :: forall a. (DType a, KnownSymbol (DTypeName a))
         => NonEmpty Int -> [Float] -> Layer (Symbol a)
constant shape value = do
    name <- getNextNamePrefixed
    sym  <- liftIO $ do
                var <- mxSymbolCreateVariable name
                let dtype = RT.pack $ symbolVal $ (Proxy :: Proxy (DTypeName a))
                setAttr var "__shape__" (tshow $ NE.toList shape)
                setAttr var "__init__"  (tshow value)
                setAttr var "__dtype__" dtype
                setAttr var "__storage_type__" "default"
                setAttr var "__grad_req__" "null"
                return $ Symbol var
    named (RT.concat [name, ".sg"]) $ blockGrad sym

convolution :: (HasArgs "_Convolution" '(Symbol, a) args
                    '["kernel", "num_filter", "data", "stride", "dilate", "pad",
                      "num_group", "workspace", "layout",
                      "cudnn_tune", "cudnn_off", "no_bias"]
               ,WithoutArgs "_Convolution" '(Symbol, a) args
                    '["bias", "weight"]
               ,DType a)
            => ArgsHMap "_Convolution" _ args -> Layer (Symbol a)
convolution args = subscope_next_name $ do
    b <- parameter "bias"   ReqWrite
    w <- parameter "weight" ReqWrite

    name <- getNamePrefixed Nothing
    if args !? #no_bias == Just True
      then
        liftIO $ S._Convolution (#weight := w .& args) name
      else
        liftIO $ S._Convolution (#bias := b .& #weight := w .& args) name

convolutionShared :: (HasArgs "_Convolution" '(Symbol, a) args
                        '["kernel", "num_filter", "stride",
                          "dilate", "pad", "num_group", "workspace",
                          "layout", "cudnn_tune", "cudnn_off", "no_bias"]
                     ,WithoutArgs "_Convolution" '(Symbol, a) args
                        '["data", "bias", "weight"]
                     ,DType a)
                  => ArgsHMap "_Convolution" _ args -> Layer (Symbol a -> Layer (Symbol a))
convolutionShared args = subscope_next_name $ do
    b <- parameter "bias"   ReqWrite
    w <- parameter "weight" ReqWrite

    return $ \data_ -> do
        name <- getNextNamePrefixed
        if args !? #no_bias == Just True
          then
            liftIO $ S._Convolution (#data := data_ .& #weight := w .& args) name
          else
            liftIO $ S._Convolution (#data := data_ .& #bias := b .& #weight := w .& args) name

fullyConnected :: (HasArgs "_FullyConnected" '(Symbol, a) args
                    '["flatten", "no_bias", "data", "num_hidden"]
                  ,WithoutArgs "_FullyConnected" '(Symbol, a) args
                    '["bias", "weight"]
                  ,DType a)
              => ArgsHMap "_FullyConnected" _ args -> Layer (Symbol a)
fullyConnected args = subscope_next_name $ do
    b <- parameter "bias"   ReqWrite
    w <- parameter "weight" ReqWrite

    name <- getNamePrefixed Nothing
    if args !? #no_bias == Just True
    then
        liftIO $ S._FullyConnected (#weight := w .& args) name
    else
        liftIO $ S._FullyConnected (#bias := b .& #weight := w .& args) name

fullyConnectedShared :: (HasArgs "_FullyConnected" '(Symbol, a) args
                            '["flatten", "no_bias", "num_hidden"]
                        ,WithoutArgs "_FullyConnected" '(Symbol, a) args
                            '["bias", "weight"]
                        ,DType a)
                     => ArgsHMap "_FullyConnected" _ args
                     -> Layer (Symbol a -> Layer (Symbol a))
fullyConnectedShared args = subscope_next_name $ do
    b <- parameter "bias"   ReqWrite
    w <- parameter "weight" ReqWrite

    return $ \data_ -> do
        name <- getNextNamePrefixed
        if args !? #no_bias == Just True
        then
            liftIO $ S._FullyConnected (#data := data_ .& #weight := w .& args) name
        else
            liftIO $ S._FullyConnected (#data := data_ .& #bias := b .& #weight := w .& args) name

batchnorm :: (HasArgs "_BatchNorm" '(Symbol, a) args
                '["data", "eps", "momentum", "fix_gamma",
                  "use_global_stats", "output_mean_var", "axis",
                  "cudnn_off", "min_calib_range", "max_calib_range"]
             ,DType a)
          => ArgsHMap "_BatchNorm" _ args -> Layer (Symbol a)
batchnorm args = subscope_next_name $ do
    gamma    <- parameter "gamma"        ReqWrite
    beta     <- parameter "beta"         ReqWrite
    mov_mean <- parameter "running_mean" ReqNull
    mov_var  <- parameter "running_var"  ReqNull

    name <- getNamePrefixed Nothing
    liftIO $ S._BatchNorm (#gamma := gamma
                        .& #beta := beta
                        .& #moving_mean := mov_mean
                        .& #moving_var := mov_var
                        .& args) name

blockGrad :: DType a => Symbol a -> Layer (Symbol a)
blockGrad s = prim S._BlockGrad (#data := s .& Nil)

-- |
-- Note that: MakeLoss is compatible with numpy semantics. It is necessary to reshape
-- the loss to be (1,) if it is a scalar.
makeLoss :: DType a => Symbol a -> Float -> Layer (Symbol a)
makeLoss s a = prim S._MakeLoss (#data := s .& #grad_scale := a .& Nil)
