import random

import gym
import numpy as np

from carla_env import CarlaEnv


class SingleCarWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 255,
                                                (3, 80, 160),
                                                dtype=np.uint8)
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

    def observation(self, observation):
        # (N_car, N_sensor, C, H, W) --> (C, H, W)
        return observation.reshape(-1, observation.shape[-2], observation.shape[-1])


class MeanReward(gym.RewardWrapper):
    def reward(self, rew):
        return np.mean(rew)


class BiasedAction(gym.Wrapper):
    def __init__(self, env, bias=[0.5, 0.0]):
        super().__init__(env)
        self.bias = np.array(bias, dtype=np.float32)

    def step(self, action):
        return self.env.step(action + self.bias)


def create_wrapped_carla_single_car_env(
        time_limit: int = 1200,
        **kwargs
):
    # create env
    env = CarlaEnv(**kwargs)
    # env = BiasedAction(env)
    env = MeanReward(env)
    env = SingleCarWrapper(env)

    if time_limit is not None:
        env = gym.wrappers.TimeLimit(env, time_limit)

    return env
