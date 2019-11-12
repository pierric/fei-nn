module MXNet.NN.Callback where

import Control.Lens (use)
import Text.Printf (printf)
import Control.Monad.IO.Class (liftIO)
import Data.IORef
import Data.Time.Clock (UTCTime, getCurrentTime, diffUTCTime)

import MXNet.NN.Session
import MXNet.NN.TaggedState (untag)

-- | Learning rate
data DumpLearningRate = DumpLearningRate

instance CallbackClass DumpLearningRate where
    endOfBatch _ _ _ = do
        lr <- use (untag . mod_statistics . stat_last_lr)
        liftIO $ putStr $ printf "<lr: %0.6f> " lr

-- | Throughput
data DumpThroughputEpoch = DumpThroughputEpoch {
    _tp_begin_time :: IORef UTCTime,
    _tp_end_time :: IORef UTCTime,
    _tp_total_sample :: IORef Int
}

instance CallbackClass DumpThroughputEpoch where
    begOfBatch _ n (DumpThroughputEpoch _ _ totalRef) = do
        liftIO $ modifyIORef totalRef (+n)
    begOfEpoch _ _ (DumpThroughputEpoch tt1Ref _ _) =
        liftIO $ getCurrentTime >>= writeIORef tt1Ref
    endOfEpoch _ _ (DumpThroughputEpoch _ tt2Ref _) = do
        liftIO $ getCurrentTime >>= writeIORef tt2Ref
    endOfVal   _ _ (DumpThroughputEpoch tt1Ref tt2Ref totalRef) = liftIO $ do
        tbeg <- readIORef tt1Ref
        tend <- readIORef tt2Ref
        let diff = realToFrac $ diffUTCTime tend tbeg :: Float
        total <- readIORef totalRef
        putStr $ printf "Throughput: %d samepls/sec " (floor $ fromIntegral total / diff :: Int)
        writeIORef totalRef 0

dumpThroughputEpoch :: IO Callback
dumpThroughputEpoch = do
    t0 <- getCurrentTime
    r0 <- newIORef t0
    r1 <- newIORef t0
    r2 <- newIORef 0
    return $ Callback $ DumpThroughputEpoch r0 r1 r2

-- | Checkpoint
data Checkpoint = Checkpoint String

instance CallbackClass Checkpoint where
    endOfVal i _ (Checkpoint path) = do
        state <- use untag
        let filename = printf "%s/epoch_%d" path i
        liftIO $ saveState (i == 0) filename state
