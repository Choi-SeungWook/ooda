"""PPO 정책망(특징 추출기)과, 학습된 모델을 FSM 과 같은 인터페이스로 쓰는 RLPursuer.

특징 추출기: 격자(8채널 20x20) → 작은 CNN, 스칼라 → 그대로 이어 붙임.
  conv3x3(32) → conv3x3(64) → conv3x3 stride2(64) → 10x10x64 → 선형 256
  20x20 은 Atari 용 NatureCNN(입력 36 이상 필요)에 비해 작아서 직접 정의한다.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from .config import EnvConfig
from .gridmap import GridMap
from .obs import N_CHANNELS, ObsBuilder


class GridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        n_vec = observation_space["vec"].shape[0]
        super().__init__(observation_space, features_dim + n_vec)
        C, H, W = observation_space["grid"].shape
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, C, H, W)).shape[1]
        self.fc = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, obs: dict) -> torch.Tensor:
        return torch.cat([self.fc(self.cnn(obs["grid"])), obs["vec"]], dim=1)


POLICY_KWARGS = dict(features_extractor_class=GridCNN,
                     features_extractor_kwargs=dict(features_dim=256),
                     net_arch=dict(pi=[128], vf=[128]))


class RLPursuer:
    """학습된 PPO 모델을 FSM 과 같은 인터페이스(reset/act)로 감싼다 → runner.run_episode 로 평가.
    결정적(deterministic) 행동을 쓴다."""

    def __init__(self, model, cfg: EnvConfig = EnvConfig(), deterministic: bool = True):
        self.model, self.cfg, self.deterministic = model, cfg, deterministic
        # 모델이 학습한 관측 채널 수로 기억 채널 사용 여부를 정한다 (결정 15 이전 모델은 8채널)
        self.memory = model.observation_space["grid"].shape[0] == N_CHANNELS

    @classmethod
    def load(cls, path: str, cfg: EnvConfig = EnvConfig(), device: str = "cpu", **kw):
        from stable_baselines3 import PPO
        return cls(PPO.load(path, device=device, custom_objects={"policy_kwargs": POLICY_KWARGS}), cfg, **kw)

    def reset(self, m: GridMap, rng: np.random.Generator) -> None:
        self.builder = ObsBuilder(m, self.cfg, memory=self.memory)

    def act(self, obs) -> int:
        o = self.builder.update(obs)
        a, _ = self.model.predict(o, deterministic=self.deterministic)
        return int(a)
