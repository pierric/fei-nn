{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedLists #-}
module Dataset where

import MXNet.Core.Base
import qualified MXNet.Core.Base.NDArray as A
import qualified MXNet.Core.Base.Internal.TH.NDArray as MXI
import Data.Function ((&))
import Streaming
import Streaming.Prelude (Of(..))
import qualified Streaming.Prelude as S
import Control.Monad.Trans.Resource (MonadResource(..))
import qualified Data.Vector as NV
import qualified Data.Vector.Storable as SV

import Parse

type SymbolF = Symbol Float
type ArrayF  = NDArray Float

device :: Context
device = contextCPU

type StreamProc a b m = Stream (Of a) m () -> Stream (Of b) m ()

mappedOf :: Monad m => (a -> m b) -> StreamProc a b m
-- mappedOf f = S.sequence . maps (first f)
mappedOf = S.mapM

cImageToNDArray :: MonadIO m => StreamProc (Batched Image) ArrayF m
cImageToNDArray = mappedOf $ \dat -> liftIO $ do
  let sz = size dat
  makeNDArray [sz, 28, 28] device $ SV.concat $ NV.toList $ _batch dat

cLabelToOnehotNDArray :: MonadIO m => StreamProc (Batched Label) ArrayF m
cLabelToOnehotNDArray = mappedOf $ \dat -> liftIO $ do
  let sz = size dat
  a <- array [sz] (NV.convert $ NV.map fromIntegral $ _batch dat) :: IO ArrayF
  b <- MXI.one_hot (A.getHandle a) 10 (add @"on_value" 1.0 $ add @"off_value" 0.0 nil)
  reshape (A.NDArray b) [sz, 10]

cBatchN :: MonadIO m => Int -> StreamProc a (Batched a) m
cBatchN n = mapped toBatch . chunksOf n
  where
    toBatch seg = first (Batched . NV.fromList) <$> S.toList seg

trainingData :: MonadResource m => Stream (Of (ArrayF, ArrayF)) m ()
trainingData = S.zip
    (sourceImages "examples/data/train-images-idx3-ubyte" & cBatchN 32 & cImageToNDArray      )
    (sourceLabels "examples/data/train-labels-idx1-ubyte" & cBatchN 32 & cLabelToOnehotNDArray)

testingData :: MonadResource m => Stream (Of (ArrayF, ArrayF)) m ()
testingData = S.zip
    (sourceImages "examples/data/t10k-images-idx3-ubyte" & cBatchN 1 & cImageToNDArray      )
    (sourceLabels "examples/data/t10k-labels-idx1-ubyte" & cBatchN 1 & cLabelToOnehotNDArray)

newtype Batched a = Batched { _batch :: NV.Vector a }

size :: Batched a -> Int
size (Batched b) = NV.length b

