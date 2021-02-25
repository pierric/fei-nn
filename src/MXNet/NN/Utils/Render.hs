{-# LANGUAGE CPP             #-}
{-# LANGUAGE OverloadedLists #-}
module MXNet.NN.Utils.Render where

#ifdef USE_REPA
import           Data.Array.Repa             (Array, DIM3, U, Z (..), extent,
                                              toUnboxed, (:.) (..))
#endif

import           Codec.Picture.Types
import qualified Graphics.Rasterific         as G
import qualified Graphics.Rasterific.Texture as G (uniformTexture)
import           Graphics.Text.TrueType      (Font)
import           RIO
import qualified RIO.Text                    as T
import qualified RIO.Vector.Storable         as V

import           MXNet.Base


render :: (G.RenderablePixel px, ColorConvertible Pixel8 px) => Int -> Int -> G.Drawing px () -> Image px
render width height = G.renderDrawing width height (promotePixel (0 :: Pixel8))


drawImage :: Image px -> G.Drawing px ()
drawImage img = G.drawImage img 0 (G.V2 0 0)


drawBox :: px -> Float -> Float -> Float -> Float -> Float -> Maybe (Text, Font, Float, px) -> G.Drawing px ()
drawBox color stroke_width x0 y0 x1 y1 label = do
    G.withTexture (G.uniformTexture color) $
        G.stroke stroke_width G.JoinRound (G.CapRound, G.CapRound) $
            G.rectangle (G.V2 x0 y0) (x1 - x0) (y1 - y0)

    case label of
      Nothing -> return ()
      Just (text, font, size, text_color) ->
          G.withTexture (G.uniformTexture text_color) $
              let text_str = T.unpack text
               in G.printTextAt font (G.PointSize size) (G.V2 (x0+2) (y0+size+2)) text_str


data NotConvertible = NotConvertible (NonEmpty Int)
    deriving Show
instance Exception NotConvertible


imageFromNDArray :: (ColorConvertible PixelRGB8 px, HasCallStack)
                 => NDArray Float -> IO (Image px)
imageFromNDArray array = do
    shape <- ndshape array
    case shape of
      [height, width, 3] -> do
          vec <- toVector array
          let img = Image width height (V.map floor vec) :: Image PixelRGB8
          return $ promoteImage img
      _ -> throwM $ NotConvertible shape


#ifdef USE_REPA
imageFromRepa :: (ColorConvertible PixelRGB8 px, HasCallStack)
              => Array U DIM3 Float -> Image px
imageFromRepa array | c == 3 = promoted
                    | otherwise = impureThrow (NotConvertible shape)
    where
        Z :. h :. w :. c = extent array
        shape = [h, w, c]
        vec = V.convert $ toUnboxed array
        image = Image w h (V.map floor vec) :: Image PixelRGB8
        promoted = promoteImage image
#endif
