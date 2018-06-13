{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
module MXNet.NN.LrScheduler where

import MXNet.Core.Base.HMap
import Data.Maybe (fromMaybe)

class LrScheduler sch where
    getLR :: sch -> Int -> Float

instance LrScheduler Float where
    getLR = const

data Const = Const Float
instance LrScheduler Const where
    getLR (Const lr) = const lr

lrOfConst :: Float -> Const
lrOfConst = Const

data FactorScheduler = Factor Float Float Int Float
instance LrScheduler FactorScheduler where
    getLR (Factor base factor step stop) nup = 
        let lr = base * factor ^ (nup `div` step)
        in if lr < stop then stop else lr

lrOfFactor :: (MatchKVList kvs '["base" ':= Float, "stop" ':= Float], QueryKV kvs)
           => Float -> Int -> HMap kvs -> FactorScheduler
lrOfFactor factor step args = Factor base factor step stop
  where 
    base = fromMaybe 0.01 (query @"base" args :: Maybe Float)
    stop = fromMaybe 1e-8 (query @"stop" args :: Maybe Float)

data MultifactorScheduler = Multifactor Float Float [Int]
instance LrScheduler MultifactorScheduler where
    getLR (Multifactor base factor steps) nup = base * factor ^ (index nup steps)
      where
        index a bs = go a bs (0 :: Int)
        go _ [] n = n
        go a (b:bs) n = if b > a then n else go a bs (n+1)

lrOfMultifactor :: (MatchKVList kvs '["base" ':= Float], QueryKV kvs)
                => Float -> [Int] -> HMap kvs -> MultifactorScheduler
lrOfMultifactor factor steps args = Multifactor base factor steps
  where 
    base = fromMaybe 0.01 (query @"base" args :: Maybe Float)

data PolyScheduler = Poly Float Float Int
instance LrScheduler PolyScheduler where
    getLR (Poly base power maxnup) nup =
        if nup < maxnup 
          then base * (1 - fromIntegral nup / fromIntegral maxnup) ** power
          else 0

lrOfPoly :: (MatchKVList kvs '["base" ':= Float, "power" ':= Float], QueryKV kvs)
           => Int -> HMap kvs -> PolyScheduler
lrOfPoly maxnup args = Poly base power maxnup
  where 
    base = fromMaybe 0.01 (query @"base"  args :: Maybe Float)
    power= fromMaybe 2    (query @"power" args :: Maybe Float)
