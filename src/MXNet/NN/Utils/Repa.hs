{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE Rank2Types #-}
module MXNet.NN.Utils.Repa where

import RIO
import RIO.List (splitAt)
import RIO.List.Partial (last)
import qualified RIO.Text as T (pack)
import qualified RIO.Vector.Boxed as V
import qualified RIO.Vector.Boxed.Partial as V (tail, foldl1')
import qualified RIO.Vector.Unboxed as VU
import qualified RIO.Vector.Unboxed.Partial as VU (maxIndex)
import Control.Exception (throw)
import Control.Lens
import Data.Array.Repa (Shape, Array, U, DIM1, DIM2, DIM3, All(..), Z(..), (:.)(..), extent, toUnboxed)
import qualified Data.Array.Repa as Repa
import Text.PrettyPrint.Leijen.Text (Pretty(..), (<+>), textStrict)

newtype PrettyArray u s e = PrettyArray (Array u s e)
instance (Pretty e, VU.Unbox e, Shape d) => Pretty (PrettyArray U d e) where
    pretty (PrettyArray arr) = textStrict (T.pack $ Repa.showShape $ extent arr) <+> pretty (VU.toList $ toUnboxed arr)

class IxedReadOnly m where
    ixr :: Index m -> Fold m (IxValue m)

type instance Index (Array u sh a) = sh
type instance IxValue (Array u sh a) = a

instance (Repa.Source u a, Shape sh) => IxedReadOnly (Array u sh a) where
    ixr i f a
        | not (Repa.inShapeRange Repa.zeroDim (extent a) i) = pure a
        | otherwise = f (Repa.unsafeIndex a i) *> pure a

newtype ArrayFlatten u sh a = ArrayFlatten {getArray :: Array u sh a}

type instance Index (ArrayFlatten u sh a) = Int
type instance IxValue (ArrayFlatten u sh a) = a

(^#!) :: (Repa.Source u a, Shape sh, HasCallStack) => Array u sh a -> Int -> a
a ^#! i = ArrayFlatten a ^?! ixr i

instance (Repa.Source u a, Shape sh) => IxedReadOnly (ArrayFlatten u sh a) where
    ixr i f aflt@(ArrayFlatten a)
        | not (i >= 0 && i < Repa.size (extent a)) = pure aflt
        | otherwise = f (Repa.unsafeLinearIndex a i) *> pure aflt

expandDim :: (Shape sh, VU.Unbox e) => Int -> Array U sh e -> Array U (sh :. Int) e
expandDim axis arr | axis >=0 && axis < rank = Repa.computeS $ Repa.reshape shape_new arr
                   | otherwise = error "Bad axis to expand."
  where
    shape = extent arr
    rank = Repa.rank shape
    (h, t) = splitAt (rank - axis) $ Repa.listOfShape shape
    shape_new = Repa.shapeOfList $ h ++ [1] ++ t


vstack :: (Shape sh, VU.Unbox e) => V.Vector (Array U sh e) -> Array U sh e
-- alternative definition:
-- vstack = Repa.transpose . V.foldl1 (Repa.++) . V.map Repa.transpose
vstack arrs = Repa.fromUnboxed shape_new $ VU.concat $ V.toList $ V.map toUnboxed arrs
  where
    sumShape sh1 sh2 = let a1:r1 = reverse $ Repa.listOfShape sh1
                           a2:r2 = reverse $ Repa.listOfShape sh2
                       in if r1 == r2
                          then Repa.shapeOfList $ reverse $ (a1+a2):r1
                          else error "Cannot stack array because of incompatible shapes"
    shape_new = V.foldl1' sumShape $ V.map extent arrs


vunstack :: (Unstackable sh, VU.Unbox e) => Array U sh e -> V.Vector (Array U (PredDIM sh) e)
vunstack arr = V.map (\i -> Repa.computeS $ Repa.slice arr (makeSliceAtAxis0 shape i)) range
  where
    shape = extent arr
    dim0 = last $ Repa.listOfShape shape
    range = V.enumFromN (0::Int) dim0

class (Shape sh,
       Shape (PredDIM sh),
       Repa.Slice (SliceAtAxis0 sh),
       Repa.FullShape (SliceAtAxis0 sh) ~ sh,
       Repa.SliceShape (SliceAtAxis0 sh) ~ PredDIM sh
      ) => Unstackable sh where
    type PredDIM sh
    type SliceAtAxis0 sh
    makeSliceAtAxis0 :: sh -> Int -> SliceAtAxis0 sh

instance Unstackable DIM2 where
    type PredDIM DIM2 = DIM1
    type SliceAtAxis0 DIM2 = Z:.Int:.All
    makeSliceAtAxis0 _ i = Z:.i:.All

instance Unstackable DIM3 where
    type PredDIM DIM3 = DIM2
    type SliceAtAxis0 DIM3 = Z:.Int:.All:.All
    makeSliceAtAxis0 (sh:._) i = makeSliceAtAxis0 sh i :. All


data ReshapeError = ReshapeMismatch (V.Vector Int) (V.Vector Int)
                  | ReshapeTooManyMinusOne (V.Vector Int)
  deriving Show
instance Exception ReshapeError

reshapeEx :: (Shape sh1, Shape sh2, VU.Unbox e) => sh2 -> Array U sh1 e -> Array U sh2 e
reshapeEx shape arr = Repa.computeS $ Repa.reshape real_new_shape arr
  where
    old_shape = V.reverse $ V.fromList $ Repa.listOfShape $ extent arr
    new_shape = V.reverse $ V.fromList $ Repa.listOfShape shape
    shapeMismatch, tooManyN1 :: forall a. a
    shapeMismatch = throw (ReshapeMismatch new_shape old_shape)
    tooManyN1 = throw (ReshapeTooManyMinusOne new_shape)

    sizeEqual sh = V.product old_shape == V.product sh
    replaceZ i v | v == 0 = case old_shape V.!? i of
                              Just v' -> v'
                              Nothing -> shapeMismatch
                  | otherwise = v
    new_shape_nz = V.imap replaceZ new_shape

    minus_n1s = V.elemIndices (-1) new_shape_nz
    filled_new_shape
        | V.null minus_n1s = if sizeEqual new_shape_nz then new_shape_nz else shapeMismatch
        | [s] <- V.toList minus_n1s = let (new_p1, new_p2) = V.splitAt s new_shape_nz
                                      in matchN1 new_p1 (V.tail new_p2) old_shape
        | otherwise = tooManyN1

    matchN1 sh1a sh1b sh2 | r == 0 = sh1a V.++ V.fromList [q] V.++ sh1b
                          | otherwise = shapeMismatch
      where size1 = V.product $ sh1a V.++ sh1b
            size2 = V.product sh2
            (q, r) = size2 `divMod` size1

    real_new_shape = Repa.shapeOfList $ V.toList $ V.reverse filled_new_shape

argMax :: (VU.Unbox e, Ord e)
       => Array U DIM2 e -> V.Vector Int
--argMax overlaps =
--    let Z :. m :. n = extent overlaps
--        findMax row = VU.maxIndex $ toUnboxed $ Repa.computeS $ Repa.slice overlaps (Z :. row :. All)
--    in V.map findMax $ V.enumFromN (0 :: Int) m
argMax arr = V.map (VU.maxIndex . toUnboxed) (vunstack arr)
