{-# LANGUAGE FlexibleContexts #-}
module MXNet.NN.LrScheduler where

class LrScheduler sch where
    getLR :: sch -> Int -> Float

instance LrScheduler Float where
    getLR = const

data Const = Const Float
instance LrScheduler Const where
    getLR (Const lr) = const lr

data FactorScheduler = Factor Float Float Int Float
instance LrScheduler FactorScheduler where
    getLR (Factor base factor step stop) nup = 
        let lr = base * factor ^ (nup `div` step)
        in if lr < stop then stop else lr

data MultifactorScheduler = Multifactor Float Float [Int]
instance LrScheduler MultifactorScheduler where
    getLR (Multifactor base factor steps) nup = base * factor ^ (index nup steps)
      where
        index a bs = go a bs (0 :: Int)
        go _ [] n = n
        go a (b:bs) n = if b > a then n else go a bs (n+1)

data PolyScheduler = Poly Float Float Int
instance LrScheduler PolyScheduler where
    getLR (Poly base power maxnup) nup =
        if nup < maxnup 
          then base * (1 + fromIntegral nup / fromIntegral maxnup) ** power
          else 0