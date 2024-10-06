from pathlib import Path
import sys
from utils.job import Job
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.logger import HParam
from utils.callbacks import MemoryCallback, HParamCallback, MultiBarCallback

# from nett.utils.train import compute_train_performance

from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit

def initialize_callbacks(job: Job) -> CallbackList:

    hparam_callback = HParamCallback() # TODO: Are we using the tensorboard that this creates? See https://www.tensorflow.org/tensorboard Appears to be responsible for logs/events.out.. files

    callback_list = [hparam_callback]

    if job.estimate_memory:
        callback_list.extend([
            # creates the parallel progress bars
            MultiBarCallback(job.index, "Estimating Memory Usage"), # TODO: Add progress bars to test aswell
            # creates the memory callback for estimation of memory for a single job
            MemoryCallback(job.device, save_path=job.paths["base"])
            ])
    else:
        # creates the parallel progress bars
        callback_list.append(MultiBarCallback(job.index, f"{job.condition}-{job.brain_id}", job.iterations["train"])) # TODO: Add progress bars to test aswell

    if job.save_checkpoints:
        callback_list.append(CheckpointCallback(
            save_freq=job.checkpoint_freq, # defaults to 30_000 steps
            save_path=job.paths["checkpoints"],
            save_replay_buffer=True,
            save_vecnormalize=True))

    return CallbackList(callback_list)
