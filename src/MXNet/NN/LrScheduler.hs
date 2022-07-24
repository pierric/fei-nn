{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE FlexibleContexts #-}
module MXNet.NN.LrScheduler where

import           MXNet.Base.Spec.Operator
import           RIO                      hiding (Const)

class Show sch => LrScheduler sch where
    baseLR :: sch -> Float
    getLR  :: sch -> Int -> Float

instance LrScheduler Float where
    baseLR = id
    getLR = const

newtype Const = Const Float
    deriving Show
instance LrScheduler Const where
    baseLR (Const lr) = lr
    getLR  (Const lr) = const lr

lrOfConst :: Float -> Const
lrOfConst = Const

data FactorScheduler = Factor Float Float Int Float
    deriving Show
instance LrScheduler FactorScheduler where
    baseLR (Factor base _ _ _) = base
    getLR  (Factor base factor step stop) nup =
        let lr = base * factor ^ (nup `div` step)
        in if lr < stop then stop else lr

type instance ParameterList "lrOfFactor" () =
    '[ '("factor", 'AttrReq Float), '("step", 'AttrReq Int),
       '("base", 'AttrOpt Float), '("stop", 'AttrOpt Float)]

lrOfFactor :: Fullfilled "lrOfFactor" () args
           => ArgsHMap "lrOfFactor" () args -> FactorScheduler
lrOfFactor args = Factor base factor step stop
  where
    factor = args ! #factor
    step   = args ! #step
    base   = fromMaybe 0.01 (args !? #base)
    stop   = fromMaybe 1e-8 (args !? #stop)

data MultifactorScheduler = Multifactor Float Float [Int]
    deriving Show
instance LrScheduler MultifactorScheduler where
    baseLR (Multifactor base _ _) = base
    getLR  (Multifactor base factor steps) nup = base * factor ^ (index nup steps)
      where
        index a bs = go a bs (0 :: Int)
        go _ [] n     = n
        go a (b:bs) n = if b > a then n else go a bs (n+1)

type instance ParameterList "lrOfMultifactor" () =
    '[ '("factor", 'AttrReq Float), '("steps", 'AttrReq [Int]), '("base", 'AttrOpt Float)]

lrOfMultifactor :: Fullfilled "lrOfMultifactor" () args
                => ArgsHMap "lrOfMultifactor" () args -> MultifactorScheduler
lrOfMultifactor args = Multifactor base factor steps
  where
    factor = args ! #factor
    steps  = args ! #steps
    base = fromMaybe 0.01 (args !? #base)

data PolyScheduler = Poly Float Float Int
    deriving Show
instance LrScheduler PolyScheduler where
    baseLR (Poly base _ _) = base
    getLR  (Poly base power maxnup) nup =
        if nup < maxnup
          then base * (1 - fromIntegral nup / fromIntegral maxnup) ** power
          else 0

type instance ParameterList "lrOfPoly" () =
    '[ '("maxnup", 'AttrReq Int), '("power", 'AttrReq Float), '("base", 'AttrOpt Float)]

lrOfPoly :: Fullfilled "lrOfPoly" () args
           => ArgsHMap "lrOfPoly" () args -> PolyScheduler
lrOfPoly args = Poly base power maxnup
  where
    maxnup = args ! #maxnup
    base   = fromMaybe 0.01 (args !? #base)
    power  = fromMaybe 2    (args !? #power)

data WarmupScheduler a = WarmupScheduler Int a
    deriving Show
instance LrScheduler a => LrScheduler (WarmupScheduler a) where
    baseLR (WarmupScheduler _ sch) = baseLR sch
    getLR  (WarmupScheduler warmup_steps sch) nup =
        let base = baseLR sch
         in if nup >= warmup_steps
            then getLR sch (nup - warmup_steps)
            else base / fromIntegral warmup_steps * fromIntegral nup

