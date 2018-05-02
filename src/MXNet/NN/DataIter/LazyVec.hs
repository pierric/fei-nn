{-# LANGUAGE FlexibleInstances #-}
module MXNet.NN.DataIter.LazyVec where

import Prelude hiding (zip)
import Data.Vector (Vector)
import qualified Data.Vector as V
import qualified Data.Vector.Mutable as VM
import Data.IORef
import Control.Monad
import Control.Monad.IO.Class
import MXNet.Core.Base (NDArray)
import MXNet.NN.DataIter.Class

data Lazy a = Direct a | Make (() -> IO a)

instance Functor Lazy where
    fmap f (Direct a) = Direct (f a)
    fmap f (Make g)   = Make (g >=> return . f)

force :: Lazy a -> IO a
force (Direct a) = return a
force (Make f)   = f ()

-- type LVec a = Lazy (Vector (Lazy a))

-- fromVector :: Vector a -> LVec a
-- fromVector = Direct . V.map Direct

-- toVector :: LVec a -> IO (Vector a)
-- toVector v = join $ force $ fmap (V.mapM force) v

type LVec a = Lazy (Vector a)

fromVec :: Vector a -> LVec a
fromVec = Direct

toVec :: LVec a -> IO (Vector a)
toVec = force 

batch :: Int -> LVec a -> IO (LVec (Vector a))
batch chunksize vec = do
    pos <- newIORef 0
    return $ case vec of 
      Direct v -> makeChunk pos v
      Make   f -> Make $ f >=> (toVec . makeChunk pos)
  where
    makeChunk cur_pos vector =
        let total = V.length vector
            (quotient, remainder) = divMod total chunksize
            new_vec_size = if remainder > 0 then quotient + 1 else quotient    
        in Make $ \_ -> do
            vec' <- VM.new new_vec_size
            forM_ [0..new_vec_size-1] $ \ i -> do
                j <- readIORef cur_pos
                if j + chunksize >= total 
                    then do 
                        let rst = total - j
                            rnd = chunksize - rst
                        VM.write vec' i (V.slice j rst vector V.++ V.slice 0 rnd vector)
                        writeIORef cur_pos rnd
                    else do
                        VM.write vec' i (V.slice j chunksize vector)
                        writeIORef cur_pos (j+chunksize)
            V.freeze vec'    

zip :: LVec a -> LVec b -> LVec (a,b)
zip (Direct a) (Direct b) = Direct (V.zip a b)
zip (Direct a) (Make   f) = Make (f >=> return . V.zip a)
zip (Make   f) (Direct b) = Make (f >=> return . flip V.zip b)
zip (Make   f) (Make   g) = Make (\_ -> liftM2 V.zip (f ()) (g ()))

map :: (a -> IO b) -> LVec a -> LVec b
map f v = case fmap (V.mapM f) v of 
            Direct a -> Make (\_ -> a)
            Make   m -> Make (join . m)

type instance DatasetConstraint (LVec (NDArray t, NDArray t)) m2 = MonadIO m2

instance Dataset (LVec (NDArray t, NDArray t)) where
    type DatType (LVec (NDArray t, NDArray t)) = t
    size dat = liftIO $ (toVec dat >>= return . V.length)
    forEach  dat proc = do
        vec <- liftIO $ toVec dat
        ret <- V.imapM (\i (x,y) -> proc (i+1) x y) vec
        return $ V.toList ret        
    forEach' dat proc = do
        vec <- liftIO $ toVec dat
        let t = V.length vec
        ret <- V.imapM (\i (x,y) -> proc (i+1,t) x y) vec
        return $ V.toList ret