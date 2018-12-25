module MXNet.NN.Callback where

import Control.Monad.State.Strict (lift)
import Control.Lens (use)
import Text.Printf (printf)
import Control.Monad.IO.Class (liftIO)
import Data.IORef
import Data.Time.Clock (UTCTime, getCurrentTime, diffUTCTime)

import MXNet.NN.Types

-- | Learning rate
data DumpLearningRate = DumpLearningRate

instance CallbackClass DumpLearningRate where
    endOfBatch _ _ _ = do
        lr <- lift $ use stat_last_lr
        liftIO $ do
            putStr $ printf "<lr: %0.6f>" lr

-- | Throughput 
data DumpThroughputEpoch = DumpThroughputEpoch {
    tp_begin_time :: IORef UTCTime,
    tp_total_sample :: IORef Int
}

instance CallbackClass DumpThroughputEpoch where
    begOfBatch _ n (DumpThroughputEpoch _ totalRef) = do
        liftIO $ modifyIORef totalRef (+n)
    begOfEpoch _ (DumpThroughputEpoch timestampRef _) =
        liftIO $ getCurrentTime >>= writeIORef timestampRef
    endOfEpoch _ (DumpThroughputEpoch timestampRef totalRef) = do
        liftIO $ do
            tend <- getCurrentTime
            tbeg <- readIORef timestampRef
            let diff = diffUTCTime tend tbeg
            total <- readIORef totalRef
            putStr $ printf "<throughtput: %d samepls/sec>" (floor $ fromIntegral total / realToFrac diff :: Int)
            writeIORef totalRef 0

dumpThroughputEpoch :: IO Callback
dumpThroughputEpoch = do
    t0 <- getCurrentTime
    r0 <- newIORef t0
    r1 <- newIORef 0
    return $ Callback $ DumpThroughputEpoch r0 r1