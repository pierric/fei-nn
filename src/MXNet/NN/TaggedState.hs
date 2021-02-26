{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE PolyKinds           #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeOperators       #-}
module MXNet.NN.TaggedState where

import           Control.Lens      (makeLenses)
import           Data.Type.Index
import           Data.Type.Product
import qualified GHC.TypeLits      as L
import           RIO
import           RIO.State         (StateT (..))

newtype Tagged a (t :: L.Symbol) = Tagged {_untag :: a} deriving Show
makeLenses ''Tagged


-- liftSub :: forall k (f :: k -> *) ss t m a. (Elem ss t, Monad m) => ReaderT (f t) m a -> ReaderT (Prod f ss) m a
-- liftSub (ReaderT m) = ReaderT (m . index elemIndex)
liftSub :: forall k (f :: k -> *) s1 s2 m a. (Elem s2 s1, Monad m) => StateT (f s1) m a -> StateT (Prod f s2) m a
liftSub (StateT m1) = StateT $ \s -> do
    (a, si) <- m1 $ index elemIndex s
    let new_s = modify elemIndex si s
    new_s `seq` return (a, new_s)


modify :: Index as a -> f a -> Prod f as -> Prod f as
modify IZ new (_ :< remainder)        = new :< remainder
modify (IS s) new (item :< remainder) = item :< modify s new remainder


toPair :: forall t a. L.KnownSymbol t => Tagged a t -> (String, a)
toPair (Tagged a)= (L.symbolVal (Proxy :: Proxy t), a)


-- a1 :: StateT (Tagged Int "A") IO ()
-- a1 = put (Tagged 4)
--
-- a2 :: StateT (Tagged String "B") IO ()
-- a2 = put (Tagged "hi")
--
-- a3 :: StateT (ProdI '[Tagged Int "A", Tagged String "B"]) IO ()
-- a3 = do
--     liftT a1
--     liftT a2

-- runStateT a3 (Identity (Tagged 0) :> Identity (Tagged ""))

