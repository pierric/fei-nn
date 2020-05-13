module MXNet.NN.DataIter.ConduitAsync where

import RIO
import qualified Data.Conduit as C
import qualified Data.Conduit.Combinators as C
import qualified Data.Conduit.List as CL
import qualified Data.Conduit.Async as CA

import MXNet.NN.DataIter.Class
import qualified MXNet.NN.DataIter.Conduit as DC


newtype ConduitAsyncData m a = ConduitAsyncData (DC.ConduitData m a)

asyncConduit :: Maybe Int -> C.ConduitM () a m () -> ConduitAsyncData m a
asyncConduit sz cc = ConduitAsyncData (DC.ConduitData sz cc)


instance Dataset ConduitAsyncData where
    type DatasetMonadConstraint ConduitAsyncData m = MonadUnliftIO m
    fromListD = ConduitAsyncData . fromListD
    zipD (ConduitAsyncData d1) (ConduitAsyncData d2) = ConduitAsyncData $ zipD d1 d2
    sizeD (ConduitAsyncData d) = sizeD d
    forEachD (ConduitAsyncData d) proc = CA.runCConduit $ DC.getConduit d CA.=$=& (CL.mapM proc C..| CL.consume)
    foldD proc unit (ConduitAsyncData d) = CA.runCConduit $ DC.getConduit d CA.=$=& C.foldM proc unit
    takeD n (ConduitAsyncData d) = ConduitAsyncData (takeD n d)
    liftD (ConduitAsyncData d) = ConduitAsyncData (liftD d)
