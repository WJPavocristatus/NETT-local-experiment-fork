import gym

import torch as th
import timm
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SegmentAnything(BaseFeaturesExtractor):


    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 384) -> None:
        super(SegmentAnything, self).__init__(observation_space, features_dim)
        self.n_input_channels = observation_space.shape[0]
        self.transforms = Compose([Resize(size=256,
                                          interpolation=InterpolationMode.BICUBIC,
                                          max_size=None,
                                          antialias=True),
                                   CenterCrop(size=(224, 224)),
                                   Normalize(mean=th.tensor([0.485, 0.456, 0.406]),
                                             std=th.tensor([0.229, 0.224, 0.225]))])

        n_input_channels = observation_space.shape[0]
        print("N_input_channels", n_input_channels)

        self.model = timm.create_model("samvit_base_patch16.sa1b", pretrained=True,
                                       num_classes=0)  # remove classifier th.nn.Linear)

    def forward(self, observations: th.Tensor) -> th.Tensor:
   
        # Cut off image
        # reshape to from vector to W*H
        # gray to color transform
        # application of ResNet
        # Concat features to the rest of observation vector
        # return
        return self.model(self.transforms(observations))
