module Parse where

import Data.Attoparsec.ByteString as AP
import Data.Attoparsec.Binary as AP
import qualified Data.ByteString.Internal as BS
import qualified Data.Vector.Storable as SV

type Image = SV.Vector Float
type Label = Int

data Header = HeaderImg Int Int Int
            | HeaderLbl Int

header :: AP.Parser Header
header = do
  mc <- AP.anyWord32be
  case mc of
    0x00000803 -> do 
      [d1,d2,d3] <- AP.count 3 AP.anyWord32be
      return $ HeaderImg (fromIntegral d1) (fromIntegral d2) (fromIntegral d3)
    0x00000801 -> do 
      d1 <- AP.anyWord32be
      return $ HeaderLbl (fromIntegral d1)
    _ -> fail "Header type not recognised"

image :: Int -> Int -> AP.Parser Image
image w h = do
  BS.PS fp ofs len <- AP.take (w*h)
  let vw = SV.unsafeFromForeignPtr fp ofs len
  return $ SV.map ((/255) . fromIntegral) vw

label :: AP.Parser Label
label = fromIntegral <$> AP.anyWord8
