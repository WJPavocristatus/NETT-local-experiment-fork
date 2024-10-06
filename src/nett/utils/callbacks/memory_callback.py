from pathlib import Path
from tqdm import tqdm

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.logger import HParam

# from nett.utils.train import compute_train_performance

from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit

class MemoryCallback(BaseCallback):

    def __init__(self, device: int, save_path: str) -> None:
        super().__init__()
        self.device = device
        self.save_path = save_path
        self.close = False
        nvmlInit()

    def _on_step(self) -> bool:

        if self.close:
            # Create a temporary directory to store the memory usage
            # os.makedirs("./.tmp", exist_ok=True)
            # Grab the memory being used by the GPU
            used_memory = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(self.device)).used
            # Write the used memory to a file
            with open(Path.joinpath(self.save_path, "mem.txt"), "w") as f:
                f.write(str(used_memory))
            # Close the callback
            return False
        return True

    def _on_rollout_end(self) -> None:
   
        self.close = True
        pass
