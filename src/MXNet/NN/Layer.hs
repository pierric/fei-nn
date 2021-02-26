{-# LANGUAGE MultiParamTypeClasses  #-}
{-# LANGUAGE PartialTypeSignatures  #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances   #-}
module MXNet.NN.Layer where

import qualified Data.UUID                   as UUID
import qualified Data.UUID.V4                as UUID
import           Formatting                  (formatToString, int, shown, stext,
                                              (%))
import           RIO
import qualified RIO.NonEmpty                as NE
import qualified RIO.State                   as ST
import qualified RIO.Text                    as RT
import           System.IO.Unsafe            (unsafePerformIO)

import           MXNet.Base
import qualified MXNet.Base.Operators.Tensor as S

runLayerBuilder :: MonadIO m => Layer a -> m a
runLayerBuilder = liftIO . flip ST.evalStateT []

type instance TensorMonad SymbolHandle  = Layer

instance PrimTensorOp SymbolHandle SymbolHandle where
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


variable :: Text -> Layer SymbolHandle
variable name = getNamePrefixed (Just name) >>= liftIO . mxSymbolCreateVariable

constant :: NonEmpty Int -> [Float] -> Layer SymbolHandle
constant shape value = do
    name <- getNextNamePrefixed
    let build = do var <- mxSymbolCreateVariable name
                   mxSymbolSetAttr var "__shape__" (tshow $ NE.toList shape)
                   mxSymbolSetAttr var "__init__"  (tshow value)
                   return var
    named (RT.concat [name, ".sg"]) $ blockGrad =<< liftIO build

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

