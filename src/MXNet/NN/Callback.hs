module MXNet.NN.Callback where

import Control.Monad.State.Strict (lift)
import Control.Lens (use)
import Text.Printf (printf)
import Control.Monad.IO.Class (liftIO)
import Control.Applicative (Alternative(..))
import Data.IORef
import Data.Dynamic (fromDynamic)
import Data.Maybe (fromMaybe)
import Data.Monoid (Alt(..))
import Data.Time.Clock (UTCTime, getCurrentTime, diffUTCTime)
import qualified Data.HashMap.Strict as M

import MXNet.NN.Types
import MXNet.NN.Utils

-- | Learning rate
data DumpLearningRate = DumpLearningRate

instance CallbackClass DumpLearningRate where
    endOfBatch _ _ _ = do
        lr <- lift $ use stat_last_lr
        liftIO $ do
            putStr $ printf "<lr: %0.6f> " lr

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
        store <- use sess_store
        let getKey key = fromMaybe (0 :: Float) $ getAlt $
                            (Alt $ M.lookup ("val_" ++ key)   store >>= fromDynamic) <|>
                            (Alt $ M.lookup ("train_" ++ key) store >>= fromDynamic)
            acc  = getKey "acc"
            loss = getKey "loss"
            filename = printf "%s/epoch_%d_acc_%.2f_loss_%.2f" path i acc loss
        saveSession filename