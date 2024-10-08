import gym
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNNLSTM(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256) -> None:
        super(CNNLSTM, self).__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        # define LSTM layer
        hidden_size = 512
        self.lstm = nn.LSTM(input_size=n_flatten, hidden_size=hidden_size,
                            num_layers=2, batch_first=True)
        
        # outputs
        self.linear = nn.Sequential(nn.Linear(hidden_size, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor):
        x = observations # original shape -> (length, batchsize, obs_size)
        # T,B, *_ = x.shape

        # Pass through CNN layers
        x = self.cnn(x)

        # Flatten the output for LSTM
        x = x.view(x.size(0), x.size(1), -1)

        # Pass through LSTM layer
        x, _ = self.lstm(x)

        # Get the last time step's output and apply the fully connected layer
        x = self.linear(x[:, -1, :])

        return x

class Identity(nn.Module):
    def __init__(self) -> None:
        super(Identity, self).__init__()

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x
