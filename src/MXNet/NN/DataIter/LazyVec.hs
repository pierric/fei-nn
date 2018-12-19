{-# Language MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
module MXNet.NN.DataIter.LazyVec where

import Prelude hiding (zip)
import Data.Vector (Vector)
import qualified Data.Vector as V
import qualified Data.Vector.Mutable as VM
import Data.IORef
import Control.Monad
import Control.Monad.IO.Class
import Control.Exception.Base (assert)
import MXNet.NN.DataIter.Class

data Lazy a = Direct a | Make (() -> IO a)

instance Functor Lazy where
    fmap f (Direct a) = Direct (f a)
    fmap f (Make g)   = Make (g >=> return . f)

force :: Lazy a -> IO a
force (Direct a) = return a
force (Make f)   = f ()

data LVec a = LVec { size :: Int, unLVec :: Lazy (Vector a)}

fromVec :: Vector a -> LVec a
fromVec v = LVec (V.length v) (Direct v)

toVec :: LVec a -> IO (Vector a)
toVec = force . unLVec

batch :: Int -> LVec a -> IO (LVec (Vector a))
batch chunksize vec = do
    pos <- newIORef 0
    return $ case unLVec vec of 
      Direct v -> makeChunk pos v
      Make   f -> LVec new_vec_size . Make $ f >=> (toVec . makeChunk pos)
  where
    total = size vec
    (quotient, remainder) = divMod total chunksize
    new_vec_size = if remainder > 0 then quotient + 1 else quotient    
    makeChunk cur_pos vector =
        assert (V.length vector == total) $ 
        LVec new_vec_size . Make $ \_ -> do
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
zip (LVec n1 (Direct a)) (LVec n2 (Direct b)) = LVec (min n1 n2) (Direct (V.zip a b))
zip (LVec n1 (Direct a)) (LVec n2 (Make   f)) = LVec (min n1 n2) (Make (f >=> return . V.zip a))
zip (LVec n1 (Make   f)) (LVec n2 (Direct b)) = LVec (min n1 n2) (Make (f >=> return . flip V.zip b))
zip (LVec n1 (Make   f)) (LVec n2 (Make   g)) = LVec (min n1 n2) (Make (\_ -> liftM2 V.zip (f ()) (g ())))

map :: (a -> IO b) -> LVec a -> LVec b
map f v = case fmap (V.mapM f) (unLVec v) of 
            Direct a -> LVec (size v) $ Make (\_ -> a)
            Make   m -> LVec (size v) $ Make (join . m)

type instance DatasetConstraint LVec m = MonadIO m

instance Dataset LVec where
    fromListD = fromVec . V.fromList
    zipD      = zip
    sizeD dat = return $ size dat
    forEachD dat proc = do
        vec <- liftIO $ toVec dat
        ret <- V.mapM proc vec
        return $ V.toList ret

    -- LVec does not support infinite stream, so we override the 
    -- default implementations
    forEachD_i  dat = forEachD (zipD (fromListD [1..size dat]) dat)

    foldD dat ele proc = do
        vec <- liftIO $ toVec dat
        V.foldM proc ele vec