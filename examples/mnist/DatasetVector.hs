module DatasetVector where

import MXNet.Base (Symbol, NDArray, makeNDArray, contextCPU)
import Data.Typeable
import Control.Monad.Trans.Resource (MonadThrow(..))
import Control.Monad (liftM2)
import Control.Exception.Base
import qualified Data.Vector as V
import qualified Data.Vector.Storable as VS
import qualified Data.ByteString as BS
import Data.Attoparsec.ByteString as AP

import MXNet.NN.DataIter.Class
import MXNet.NN.DataIter.Vec
import Parse

type SymbolF = Symbol Float
type ArrayF  = NDArray Float

loadTrainingData :: IO (DatasetVector IO (ArrayF, ArrayF))
loadTrainingData = do
    v1 <- batch 128 <$> sourceImages "examples/data/train-images-idx3-ubyte"
    v2 <- batch 128 <$> sourceLabels "examples/data/train-labels-idx1-ubyte"
    liftM2 zipD (mapMD cImageToNDArray v1) (mapMD cLabelToNDArray v2)

loadTestingData :: IO (DatasetVector IO (ArrayF, ArrayF))
loadTestingData = do
    v1 <- batch 100 <$> sourceImages "examples/data/t10k-images-idx3-ubyte"
    v2 <- batch 100 <$> sourceLabels "examples/data/t10k-labels-idx1-ubyte"
    liftM2 zipD (mapMD cImageToNDArray v1) (mapMD cLabelToNDArray v2)

sourceImages :: FilePath -> IO (DatasetVector IO Image)
sourceImages = parseFile $ do
    HeaderImg n w h <- header
    count n (image w h)

sourceLabels :: FilePath -> IO (DatasetVector IO Label)
sourceLabels = parseFile $ do
    HeaderLbl n <- header
    count n label

parseFile :: Parser [a] -> FilePath -> IO (DatasetVector IO a)
parseFile parser fp = do
    content <- BS.readFile fp
    case AP.parseOnly parser content of
        Left msg -> throwM $ ParseError msg
        Right rt -> return $ fromListD rt

batch :: Int -> DatasetVector IO a -> DatasetVector IO (V.Vector a)
batch n (DatasetVector vec) = (DatasetVector $ walk n vec)
  where
  walk n vec = V.unfoldr (\v -> if V.null v then Nothing else Just (V.splitAt n v)) vec

mapMD f (DatasetVector vec) = DatasetVector <$> V.mapM f vec

cImageToNDArray :: V.Vector Image -> IO ArrayF
cImageToNDArray dat = makeNDArray [V.length dat, 1, 28, 28] contextCPU (VS.concat $ V.toList dat)
cLabelToNDArray :: V.Vector Label -> IO ArrayF
cLabelToNDArray dat = makeNDArray [V.length dat] contextCPU (V.convert $ V.map fromIntegral dat)

data Exc = ParseError String
    deriving (Show, Typeable)
instance Exception Exc
