import os
from typing import Any, Optional
from pathlib import Path
import inspect
import torch
import stable_baselines3
import sb3_contrib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common import results_plotter
from nett.brain import algorithms, policies, encoder_dict
from nett.brain import encoders
from nett.utils.callbacks import initialize_callbacks
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import gym
from nett.utils.job import Job

from nett import logger

class Brain:

    def __init__(
        self,
        policy: Any | str,
        algorithm:  str | OnPolicyAlgorithm | OffPolicyAlgorithm,
        encoder: Any | str = None,
        embedding_dim: Optional[int] = None,
        reward: str = "supervised",
        batch_size: int = 512,
        buffer_size: int = 2048,
        train_encoder: bool = True,
        seed: int = 12,
        custom_encoder_args: dict[str, str]= {},
        custom_policy_arch: Optional[list[int|dict[str,list[int]]]] = None
    ) -> None:
    
       

        self.logger = logger.getChild(__class__.__name__)

        # Set attributes
        self.algorithm = self._validate_algorithm(algorithm)
        self.policy = self._validate_policy(policy)
        self.train_encoder = train_encoder
        self.encoder = self._validate_encoder(encoder) if encoder else None
        self.reward = self._validate_reward(reward) if reward else None

        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.seed = seed
        self.custom_encoder_args = custom_encoder_args
        self.custom_policy_arch = custom_policy_arch

    def train(self, env: "gym.Env", job: "Job"):

        # importlib.reload(stable_baselines3)
        # validate environment
        env = self._validate_env(env)

        # initialize environment
        envs = make_vec_env(
            env_id=lambda: env, 
            n_envs=1, 
            # seed=self.seed, # Commented out as seed does not work
            monitor_dir=str(job.paths["env_logs"])) #TODO: Switch to multi-processing for parallel environments with vec_envs #TODO: Add custom seed function for seeding env, see https://stackoverflow.com/questions/47331235/how-should-openai-environments-gyms-use-env-seed0

        # build model
        policy_kwargs = {
            "features_extractor_class": self.encoder,
            "features_extractor_kwargs": {
                "features_dim": self.embedding_dim or inspect.signature(self.encoder).parameters["features_dim"].default,
            }
        } if self.encoder is not None else {}

        if len(self.custom_encoder_args) > 0:
            policy_kwargs["features_extractor_kwargs"].update(self.custom_encoder_args)
        
        if self.custom_policy_arch:
            policy_kwargs["net_arch"] = self.custom_policy_arch
            
        self.logger.info(f'Training with {self.algorithm.__name__}')
        try:
            model = self.algorithm(
                self.policy,
                envs,
                batch_size=self.batch_size,
                n_steps=self.buffer_size,
                verbose=1,
                policy_kwargs=policy_kwargs,
                device=torch.device("cuda", job.device))
            
        except Exception as e:
            self.logger.exception(f"Failed to initialize model with error: {str(e)}")
            raise e

        # setup tensorboard logger and attach to model
        tb_logger = configure(str(job.paths["logs"]), ["stdout", "csv", "tensorboard"])
        model.set_logger(tb_logger)
        
        self.logger.info(f"Tensorboard logs saved at {str(job.paths['logs'])}")
        # set encoder as eval only if train_encoder is not True
        if not self.train_encoder:
            model = self._set_encoder_as_eval(model)
            self.logger.warning(f"Encoder training is set to {str(self.train_encoder).upper()}")

        # initialize callbacks
        self.logger.info("Initializing Callbacks")
        callback_list = initialize_callbacks(job)

        # train
        self.logger.info(f"Total number of training steps: {job.iterations['train']}")
        model.learn(
            total_timesteps=job.iterations["train"],
            tb_log_name=self.algorithm.__name__,
            progress_bar=False,
            callback=[callback_list])
        self.logger.info("Training Complete")

        # nothing else is needed for memory estimation
        if job.estimate_memory:
            return

        # save
        ## create save directory
        job.paths["model"].mkdir(parents=True, exist_ok=True)
        self.save_encoder_policy_network(model.policy, job.paths["model"])
        print("Saved feature extractor")
        
        save_path = f"{job.paths['model'].joinpath('latest_model.zip')}"
        model.save(save_path)
        self.logger.info(f"Saved model at {save_path}")

        # plot reward graph
        self.plot_results(iterations=job.iterations["train"],
                        model_log_dir=job.paths["env_logs"],
                        plots_dir=job.paths["plots"],
                        name="reward_graph")   

    def test(self, env: "gym.Env", job: "Job"):
   
        # load previously trained model from save_dir, if it exists
        model: OnPolicyAlgorithm | OffPolicyAlgorithm = self.algorithm.load(
            job.paths['model'].joinpath('latest_model.zip'), 
            device=torch.device('cuda', job.device))

        # validate environment
        env = self._validate_env(env)

        # initialize environment
        num_envs = 1
        envs = make_vec_env(
            env_id=lambda: env, 
            n_envs=num_envs, 
            # seed=self.seed # Commented out as seed does not work
            )
        obs = envs.reset() #TODO: try to use envs. This will return a list of obs, rather than a single obs #see https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html for details on conversion

        self.logger.info(f'Testing with {self.algorithm.__name__}')

        ## record - test video
        try:
            # vr = VideoRecorder(env=envs,
            # path="{}/agent_{}.mp4".format(job.paths["env_recs"], \
            #     str(index)), enabled=True)
            
            # for when algorithm is RecurrentPPO
            iterations: int = job.iterations["test"]
            if issubclass(self.algorithm, RecurrentPPO):
                self.logger.info(f"Total number of episodes: {iterations}")
                #iterations = 20*50 # 20 episodes of 50 conditions  each
                t = tqdm(total=iterations, desc=f"Condition {job.index}", position=job.index)
                for _ in range(iterations):
                    # cell and hidden state of the LSTM 
                    done, lstm_states = False, None
                    # episode start signals are used to reset the lstm states
                    episode_starts = np.ones((num_envs,), dtype=bool)
                    episode_length = 0
                    while not done:
                        action, lstm_states = model.predict(
                            obs,
                            state=lstm_states,
                            episode_start=episode_starts,
                            deterministic=True)
                        obs, _, done, _ = envs.step(action) # obs, rewards, done, info #TODO: try to use envs. This will return a list for each of obs, rewards, done, info rather than single values. Ex: done = [False, False, False, False, False] and not False
                        t.update(1)
                        episode_starts = done
                        episode_length += 1
                        envs.render(mode="rgb_array") #TODO: try to use envs. This will return a list of obs, rewards, done, info rather than single values
                        # vr.capture_frame()    

                # vr.close()
                # vr.enabled = False

            # for all other algorithms
            else:
            #iterations = 50*20*200 # 50 conditions of 20 steps each
                self.logger.info(f"Total number of testing steps: {iterations}")
                t = tqdm(total=iterations, desc=f"Condition {job.index}", position=job.index)
                for _ in range(iterations):
                    action, _ = model.predict(obs, deterministic=True) # action, states
                    obs, _, done, _ = envs.step(action) # obs, reward, done, info #TODO: try to use envs. This will return a list of obs, rewards, done, info rather than single values
                    t.update(1)
                    if done:
                        envs.reset()
                    envs.render(mode="rgb_array")
                    # vr.capture_frame()
        except Exception as e:
            self.logger.exception(f"Failed to test model with error: {str(e)}")
            raise e
        # finally:
            # vr.close()
            # vr.enabled = False
        
        t.close()
    
    @staticmethod
    def save_encoder_policy_network(policy, path: Path):
           
        ## save policy
        path.mkdir(parents=True, exist_ok=True)
        policy.save(os.path.join(path, "policy.pkl"))
        
        ## save encoder
        encoder = policy.features_extractor.state_dict()
        save_path = os.path.join(path, "feature_extractor.pth")
        torch.save(encoder, save_path)

        return

    @staticmethod
    def plot_results(
        iterations: int,
        model_log_dir: Path,
        plots_dir: Path,
        name: str
    ) -> None:
       
        results_plotter.plot_results([str(model_log_dir)],
            iterations,
            results_plotter.X_TIMESTEPS,
            name)
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_dir.joinpath(f"{name}.png"))
        plt.clf()

    @staticmethod
    def _validate_encoder(encoder: Any | str) -> BaseFeaturesExtractor:
     
        # for when encoder is a string
        if isinstance(encoder, str):
            if encoder not in encoder_dict.keys():
                raise ValueError(f"If a string, should be one of: {encoder_dict.keys()}")
            encoder = getattr(encoders, encoder_dict[encoder])

        # for when encoder is a custom PyTorch encoder
        if isinstance(encoder, BaseFeaturesExtractor):
            # TODO (v0.3) pass dummy torch.tensor on "meta" device to validate embedding dim
            pass

        return encoder

    @staticmethod
    def _validate_algorithm(algorithm: str | OnPolicyAlgorithm | OffPolicyAlgorithm) -> OnPolicyAlgorithm | OffPolicyAlgorithm:
   
        # for when policy is a string
        if isinstance(algorithm, str):
            if algorithm not in algorithms:
                raise ValueError(f"If a string, should be one of: {algorithms}")
            # check for the passed policy in stable_baselines3 as well as sb3-contrib
            # at this point in the code, it is guaranteed to be in either of the two
            try:
                algorithm = getattr(stable_baselines3, algorithm)
                
            except:
                algorithm = getattr(sb3_contrib, algorithm)

        # for when policy algorithm is custom
        elif isinstance(algorithm, OnPolicyAlgorithm) or isinstance(algorithm, OffPolicyAlgorithm):
            # TODO (v0.4) determine appropriate validation checks to be performed before passing
            pass

        else:
            raise ValueError(f"Policy Algorithm should be either one of {algorithms} or a subclass of [{OnPolicyAlgorithm}, {OffPolicyAlgorithm}]")

        
        return algorithm

    @staticmethod
    def _validate_policy(policy: str | BasePolicy) -> str | BasePolicy:
       
        # for when policy is a string
        if isinstance(policy, str):
            # support tested for PPO and RecurrentPPO only
            if policy not in policies:
                raise ValueError(f"If a string, should be one of: {policies}")

        # for when policy is custom
        elif isinstance(policy, BasePolicy):
            # TODO (v0.4) determine appropriate validation checks to be performed before passing
            pass

        else:
            raise ValueError(f"Policy Model should be either one of {policies} or a subclass of [{BasePolicy}]")

        return policy

    @staticmethod
    def _validate_reward(reward: str) -> str:
     
        # for when reward is a string
        if not isinstance(reward, str) or reward not in ['supervised', 'unsupervised']:
            raise ValueError("If a string, should be one of: ['supervised', 'unsupervised']")
        return reward

    @staticmethod
    def _validate_env(env: "gym.Env") -> "gym.Env":
       
        try:
            check_env(env)
        except Exception as ex:
            raise ValueError(f"Failed training env check with {str(ex)}")
        return env

    @staticmethod
    def _set_encoder_as_eval(model: OnPolicyAlgorithm | OffPolicyAlgorithm) -> OnPolicyAlgorithm | OffPolicyAlgorithm:
     
        model.policy.features_extractor.eval()

        for param in model.policy.features_extractor.parameters():
            param.requires_grad = False
        return model
    
    def __repr__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if k != 'logger'}
        return f"{self.__class__.__name__}({attrs!r})"

    def __str__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if k != 'logger'}
        return f"{self.__class__.__name__}({attrs!r})"