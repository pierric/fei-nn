{-# Language DataKinds, TypeOperators #-}
module MXNet.NN.TaggedState where

import GHC.TypeLits (Symbol)
import Data.Type.Product
import Data.Type.Index
import Control.Monad.State (StateT(..))
import Data.Functor.Identity (Identity(..))


newtype Tagged (t :: Symbol) a = Tagged a deriving Show 
type ProdI = Prod Identity

liftT :: (Monad m, Elem s2 s1) => StateT s1 m a -> StateT (ProdI s2) m a
liftT (StateT m1) = StateT $ \s -> do
    (a, si) <- m1 $ runIdentity $ index elemIndex s
    let new_s = modify elemIndex (Identity si) s
    new_s `seq` return (a, new_s)

modify :: Index as a -> f a -> Prod f as -> Prod f as
modify IZ new (_ :< remainder) = new :< remainder
modify (IS s) new (first :< remainder) = first :< modify s new remainder



-- a1 :: StateT (Tagged "A" Int) IO ()
-- a1 = put (Tagged 4)
--
-- a2 :: StateT (Tagged "B" String) IO ()
-- a2 = put (Tagged "hi")
--
-- a3 :: StateT (ProdI '[Tagged "A" Int, Tagged "B" String]) IO ()
-- a3 = do
--     liftT a1
--     liftT a2

-- runStateT a3 (Identity (Tagged 0) :> Identity (Tagged ""))
