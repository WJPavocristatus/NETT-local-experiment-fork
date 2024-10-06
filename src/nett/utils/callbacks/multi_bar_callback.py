
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.logger import HParam
import sys
from tqdm import tqdm

# from nett.utils.train import compute_train_performance

from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit

class MultiBarCallback(BaseCallback):


    def __init__(self, index: int, label: str, num_steps: int = None) -> None:
        super().__init__()
        # where on the screen the progress bar will be displayed
        self.index = index
        # label to prefix the progress bar
        self.label = label
        # progress bar object
        self.pbar = None
        # number of steps to be done
        self.num_steps = num_steps

    def _on_training_start(self) -> None:
        # if num_steps is None, this means that memory estimation is being done, so the length of a single rollout will be used
        num_steps = self.num_steps if self.num_steps is not None else self.model.n_steps
        # Initialize progress bar
        # Remove timesteps that wer4e done in previous training sessions
        self.pbar = tqdm(total=(num_steps), position=self.index, dynamic_ncols=True, desc=self.label, file=sys.stdout)
        pass

    def _on_step(self) -> bool:
        # Update progress bar, we do num_envs steps per call to `env.step()`
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self) -> None:
        self.pbar.refresh()
        self.pbar.close()
        pass
