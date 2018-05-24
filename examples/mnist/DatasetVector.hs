module DatasetVector where

import MXNet.Core.Base
import Data.Typeable
import Control.Monad.Trans.Resource (MonadResource(..), MonadThrow(..))
import Control.Monad.IO.Class (liftIO)
import Control.Exception.Base
import qualified Data.Vector as V
import qualified Data.Vector.Storable as VS
import qualified Data.ByteString as BS
import Data.Attoparsec.ByteString as AP

import MXNet.NN.DataIter.LazyVec as VL
import Parse

type SymbolF = Symbol Float
type ArrayF  = NDArray Float

loadTrainingData :: (MonadResource m, MonadThrow m) => m (VL.LVec (ArrayF, ArrayF))
loadTrainingData = do
    v1 <- sourceImages "examples/data/train-images-idx3-ubyte" >>= liftIO . batch 128
    v2 <- sourceLabels "examples/data/train-labels-idx1-ubyte" >>= liftIO . batch 128
    return $ VL.zip (VL.map cImageToNDArray v1) (VL.map cLabelToNDArray v2)

loadTestingData :: (MonadResource m, MonadThrow m) => m (VL.LVec (ArrayF, ArrayF))
loadTestingData = do
    v1 <- sourceImages "examples/data/t10k-images-idx3-ubyte" >>= liftIO . batch 1
    v2 <- sourceLabels "examples/data/t10k-labels-idx1-ubyte" >>= liftIO . batch 1
    return $ VL.zip (VL.map cImageToNDArray v1) (VL.map cLabelToNDArray v2)

sourceImages :: (MonadResource m, MonadThrow m) => FilePath -> m (VL.LVec Image)
sourceImages = parseFile $ do
    HeaderImg n w h <- header
    count n (image w h)

sourceLabels :: (MonadResource m, MonadThrow m) => FilePath -> m (VL.LVec Label)
sourceLabels = parseFile $ do
    HeaderLbl n <- header
    count n label    

parseFile :: (MonadResource m, MonadThrow m) => Parser [a] -> FilePath -> m (VL.LVec a)
parseFile parser fp = do
    content <- liftIO $ BS.readFile fp
    case AP.parseOnly parser content of
        Left msg -> throwM $ ParseError msg
        Right rt -> return $ VL.fromVec $ V.fromList rt

cImageToNDArray :: V.Vector Image -> IO ArrayF
cImageToNDArray dat = makeNDArray [V.length dat, 1, 28, 28] contextCPU (VS.concat $ V.toList dat)
cLabelToNDArray :: V.Vector Label -> IO ArrayF
cLabelToNDArray dat = makeNDArray [V.length dat] contextCPU (V.convert $ V.map fromIntegral dat)

data Exc = ParseError String
    deriving (Show, Typeable)
instance Exception Exc