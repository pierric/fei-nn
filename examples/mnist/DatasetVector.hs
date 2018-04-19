module DatasetVector where

import MXNet.Core.Base
import Data.Typeable
import Control.Monad.Trans.Resource (MonadResource(..), MonadThrow(..))
import Control.Monad (liftM2, forM_, when)
import Control.Monad.IO.Class (liftIO)
import Control.Exception.Base
import qualified Data.Vector as V
import qualified Data.Vector.Mutable as VM
import qualified Data.Vector.Storable as VS
import qualified Data.ByteString as BS
import Data.Attoparsec.ByteString as AP

import Parse

type SymbolF = Symbol Float
type ArrayF  = NDArray Float

loadTrainingData :: MonadResource m => m (V.Vector (ArrayF, ArrayF))
loadTrainingData = liftM2 V.zip
    (sourceImages "examples/data/train-images-idx3-ubyte" >>= return . cBatchN 32 >>= cImageToNDArray)
    (sourceLabels "examples/data/train-labels-idx1-ubyte" >>= return . cBatchN 32 >>= cLabelToNDArray)

loadTestingData :: MonadResource m => m (V.Vector (ArrayF, ArrayF))
loadTestingData = liftM2 V.zip
    (sourceImages "examples/data/t10k-images-idx3-ubyte" >>= return . cBatchN 1 >>= cImageToNDArray)
    (sourceLabels "examples/data/t10k-labels-idx1-ubyte" >>= return . cBatchN 1 >>= cLabelToNDArray)

sourceImages :: MonadResource m => FilePath -> m (V.Vector Image)
sourceImages = parseFile fromImageFile
  where
    fromImageFile = do
        HeaderImg n w h <- header
        count n (image w h)

sourceLabels :: MonadResource m => FilePath -> m (V.Vector Label)
sourceLabels = parseFile fromLabelFile
  where
    fromLabelFile = do
        HeaderLbl n <- header
        count n label    

parseFile :: MonadResource m => Parser [a] -> FilePath -> m (V.Vector a)
parseFile parser fp = do
    content <- liftIO $ BS.readFile fp
    case AP.parseOnly parser content of
        Left msg -> throwM $ ParseError msg
        Right rt -> return $ V.fromList rt

cBatchN :: Int -> V.Vector a -> V.Vector (V.Vector a)
cBatchN chunksize vec = V.create $ do
    vec' <- VM.new (if remainder > 0 then quotient + 1 else quotient)
    forM_ [0..quotient-1] $ \ i -> do
        VM.write vec' i (V.slice (i*chunksize) chunksize vec)
    when (remainder > 0) $ do
        VM.write vec' quotient (V.slice (quotient*chunksize) remainder vec)
    return vec'
  where
    total = V.length vec
    (quotient, remainder) = divMod total chunksize

cImageToNDArray :: MonadResource m => V.Vector (V.Vector Image) -> m (V.Vector ArrayF)
cImageToNDArray = V.mapM (\dat -> liftIO $ makeNDArray [V.length dat, 1, 28, 28] contextCPU (VS.concat $ V.toList dat))
cLabelToNDArray :: MonadResource m => V.Vector (V.Vector Label) -> m (V.Vector ArrayF)
cLabelToNDArray = V.mapM (\dat -> liftIO $ makeNDArray [V.length dat] contextCPU (V.convert $ V.map fromIntegral dat))

data Exc = ParseError String
    deriving (Show, Typeable)
instance Exception Exc