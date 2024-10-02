from pathlib import Path
from typing import Final, Any
from nett import logger

class Job:

  _MODES: Final = ("train", "test", "full")

  @classmethod
  def initialize(cls, mode: str, output_dir: Path | str, steps_per_episode: int, save_checkpoints: bool, checkpoint_freq: int,  reward: str, batch_mode: bool, iterations: dict[str, int]) -> None:
    cls.mode = cls._validate_mode(mode)
    cls.steps_per_episode: int = steps_per_episode
    cls.checkpoint_freq: int = checkpoint_freq
    cls.output_dir: Path = output_dir
    cls.reward: str = reward
    cls.save_checkpoints: bool = save_checkpoints
    cls.batch_mode: bool = batch_mode
    cls.iterations: dict[str, int] = iterations

  def __init__(self, brain_id: int, condition: str, device: int, index: int, port: int, estimate_memory: bool = False) -> None:
    self.device: int = device
    self.condition: str = condition
    self.brain_id: int = brain_id

    self.paths: dict[str, Path] = self._configure_paths()
    self.index: int = index
    self.port: int = port
    self.estimate_memory: bool = estimate_memory

    # Initialize logger

    self.logger = logger.getChild(__class__.__name__+"."+condition+"."+str(brain_id))


  def _configure_paths(self) -> dict[str, Path]:
    paths: dict[str, Path] = {
      "base": Path.joinpath(self.output_dir, self.condition, f"brain_{self.brain_id}")
      }
    SUBDIRS = ["model", "checkpoints", "plots", "logs", "env_recs", "env_logs"]
    for subdir in SUBDIRS:
      paths[subdir] = Path.joinpath(paths["base"], subdir)

    return paths

  def env_kwargs(self) -> dict[str, Any]:
    return {
      "rewarded": bool(self.reward == "supervised"),
      "rec_path": str(self.paths["env_recs"]),
      "log_path": str(self.paths["env_logs"]),
      "condition": self.condition,
      "brain_id": self.brain_id,
      "device": self.device,
      "episode_steps": self.steps_per_episode,
      "batch_mode": self.batch_mode
    }

  @staticmethod
  def _validate_mode(mode: str) -> str:
    if mode not in Job._MODES:
      raise ValueError(f"Unknown mode type {mode}, should be one of {Job._MODES}")
    return mode