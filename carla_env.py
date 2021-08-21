from manager.server import ServerManager
from manager.world import WorldManager
from manager.car import CarManager
from manager.perception import PerceptionManager

import abc
import psutil
import os
import signal
import random

import numpy as np
import carla
import gym


def kill_all_servers():
    """Kill all PIDs that start with Carla"""
    processes = [p for p in psutil.process_iter() if "carla" in p.name().lower()]
    for process in processes:
        os.kill(process.pid, signal.SIGKILL)


class CarlaEnv(gym.Env, abc.ABC):
    def __init__(self,
                 global_config: dict,
                 gpu_index: int = 0):
        super().__init__()
        self.global_config = global_config
        self.gpu_index = gpu_index

        # create instance managers
        self.initialized = False
        self.server_manager = None
        self.world_manager = None
        self.car_manager = None
        self.perception_manager = None

    def init(self):
        if not self.initialized:
            # seed random generators
            np.random.seed(random.randint(0, 1000000))

            self.server_manager = ServerManager(self.global_config, self.gpu_index)
            self.world_manager = WorldManager(self.global_config, self.server_manager)
            self.car_manager = CarManager(self.global_config, self.world_manager)
            self.perception_manager = PerceptionManager(self.global_config, self.gpu_index)

            self.initialized = True

    def __del__(self):
        # delete in sequence
        if self.initialized:
            del self.perception_manager
            del self.car_manager
            del self.world_manager
            del self.server_manager

    def _get_observation(self):
        return self.perception_manager.infer(*self.car_manager.get_observation())

    def reset(self):
        if not self.initialized:
            self.init()

        # destroy all existing cars
        self.car_manager.destroy_all_cars()

        # reset managers
        self.server_manager.reset()
        self.world_manager.reset()
        self.car_manager.reset(self.server_manager.get(), self.server_manager.tm_port)

        # get observation
        return self._get_observation()

    def step(self, action):
        # apply control for cars
        self.car_manager.apply_control(action)

        # tick world
        self.car_manager.sync_before_tick()
        self.world_manager.get().tick()
        self.car_manager.sync_after_tick()

        # get obs, reward, done, info
        return (self._get_observation(), *self.car_manager.get_reward_done(), {})
