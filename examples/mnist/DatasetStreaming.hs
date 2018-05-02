{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedLists #-}
module DatasetStreaming where

import MXNet.Core.Base
import qualified MXNet.Core.Base.NDArray as A
import qualified MXNet.Core.Base.Internal.TH.NDArray as MXI
import Data.Typeable
import Data.Function ((&))
import Streaming
import Streaming.Prelude (Of(..))
import qualified Streaming.Prelude as S
import Data.Attoparsec.ByteString.Streaming as APS
import qualified Data.ByteString.Streaming as BSS
import Control.Monad.Trans.Resource (MonadResource(..), MonadThrow(..))
import qualified Data.Vector as NV
import qualified Data.Vector.Storable as SV
import Control.Exception.Base

import MXNet.NN.Types
import Parse

type SymbolF = Symbol Float
type ArrayF  = NDArray Float

type StreamProc a b m = Stream (Of a) m Int -> Stream (Of b) m Int

mappedOf :: Monad m => (a -> m b) -> StreamProc a b m
-- mappedOf f = S.sequence . maps (first f)
mappedOf = S.mapM

cImageToNDArray :: MonadIO m => StreamProc (Batched Image) ArrayF m
cImageToNDArray = mappedOf $ \dat -> liftIO $ do
  let sz = size dat
  makeNDArray [sz, 1, 28, 28] contextCPU $ SV.concat $ NV.toList $ _batch dat

cLabelToNDArray :: MonadIO m => StreamProc (Batched Label) ArrayF m
cLabelToNDArray = mappedOf $ \dat -> liftIO $ do
  let sz = size dat
  makeNDArray [sz] contextCPU (NV.convert $ NV.map fromIntegral $ _batch dat) :: IO ArrayF

cLabelToOnehotNDArray :: MonadIO m => StreamProc (Batched Label) ArrayF m
cLabelToOnehotNDArray = mappedOf $ \dat -> liftIO $ do
  let sz = size dat
  a <- makeNDArray [sz] contextCPU (NV.convert $ NV.map fromIntegral $ _batch dat) :: IO ArrayF
  b <- MXI.one_hot (A.getHandle a) 10 (add @"on_value" 1.0 $ add @"off_value" 0.0 nil)
  reshape (A.NDArray b) [sz, 10]

cBatchN :: (MonadIO m, MonadThrow m) => Int -> StreamProc a (Batched a) m
cBatchN n s = do 
  total <- mapped toBatch $ chunksOf n s
  let (r, m) = divMod total n 
  if m > 0 then effect $ throwM BadBatchSize else return r
  where
    toBatch seg = first (Batched . NV.fromList) <$> S.toList seg

trainingData :: MonadResource m => Stream (Of (ArrayF, ArrayF)) (TrainM Float m) Int
trainingData = S.zip
    (sourceImages "examples/data/train-images-idx3-ubyte" & cBatchN 32 & cImageToNDArray      )
    (sourceLabels "examples/data/train-labels-idx1-ubyte" & cBatchN 32 & cLabelToNDArray)

testingData :: MonadResource m => Stream (Of (ArrayF, ArrayF)) (TrainM Float m) Int
testingData = S.zip
    (sourceImages "examples/data/t10k-images-idx3-ubyte" & cBatchN 1 & cImageToNDArray      )
    (sourceLabels "examples/data/t10k-labels-idx1-ubyte" & cBatchN 1 & cLabelToNDArray)

newtype Batched a = Batched { _batch :: NV.Vector a }

size :: Batched a -> Int
size (Batched b) = NV.length b

sourceImages :: MonadResource m => FilePath -> Stream (Of Image) m Int
sourceImages fp = do
  (result, rest)<- lift $ APS.parse header (BSS.readFile fp)
  case result of
    Left (HeaderImg n w h) -> APS.parsed (image w h) rest >> return n
    _ -> effect $ throwM NotImageFile

sourceLabels :: MonadResource m => FilePath -> Stream (Of Label) m Int
sourceLabels fp = do
  (result, rest)<- lift $ APS.parse header (BSS.readFile fp)
  case result of
    Left (HeaderLbl n) -> APS.parsed label rest >> return n
    _ -> effect $ throwM NotLabelFile

data Exc = NotImageFile | NotLabelFile | BadBatchSize
    deriving (Show, Typeable)
instance Exception Exc