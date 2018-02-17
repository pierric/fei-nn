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

type StreamProc a b m = Stream (Of a) m Int -> Stream (Of b) m Int

mappedOf :: Monad m => (a -> m b) -> StreamProc a b m
-- mappedOf f = S.sequence . maps (first f)
mappedOf = S.mapM

cImageToNDArray :: MonadIO m => Context -> StreamProc (Batched Image) ArrayF m
cImageToNDArray device = mappedOf $ \dat -> liftIO $ do
  let sz = size dat
  makeNDArray [sz, 1, 28, 28] device $ SV.concat $ NV.toList $ _batch dat

cLabelToOnehotNDArray :: MonadIO m => Context -> StreamProc (Batched Label) ArrayF m
cLabelToOnehotNDArray device = mappedOf $ \dat -> liftIO $ do
  let sz = size dat
  a <- makeNDArray [sz] device (NV.convert $ NV.map fromIntegral $ _batch dat) :: IO ArrayF
  b <- MXI.one_hot (A.getHandle a) 10 (add @"on_value" 1.0 $ add @"off_value" 0.0 nil)
  reshape (A.NDArray b) [sz, 10]

cBatchN :: MonadIO m => Int -> StreamProc a (Batched a) m
cBatchN n s = div' n <$> (mapped toBatch $ chunksOf n s)
  where
    toBatch seg = first (Batched . NV.fromList) <$> S.toList seg
    div' n t = let (r, m) = divMod t n in if m > 0 then r+1 else r  

trainingData :: MonadResource m => Context -> Stream (Of (ArrayF, ArrayF)) m Int
trainingData ctx = S.zip
    (sourceImages "examples/data/train-images-idx3-ubyte" & cBatchN 32 & cImageToNDArray ctx      )
    (sourceLabels "examples/data/train-labels-idx1-ubyte" & cBatchN 32 & cLabelToOnehotNDArray ctx)

testingData :: MonadResource m => Context -> Stream (Of (ArrayF, ArrayF)) m Int
testingData ctx = S.zip
    (sourceImages "examples/data/t10k-images-idx3-ubyte" & cBatchN 1 & cImageToNDArray ctx      )
    (sourceLabels "examples/data/t10k-labels-idx1-ubyte" & cBatchN 1 & cLabelToOnehotNDArray ctx)

newtype Batched a = Batched { _batch :: NV.Vector a }

size :: Batched a -> Int
size (Batched b) = NV.length b

