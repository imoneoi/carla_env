from manager.server import ServerManager
from manager.world import WorldManager
from manager.car import CarManager

import abc

import carla
import gym


class CarlaEnv(gym.Env, abc.ABC):
    def __init__(self,
                 global_config: dict,
                 gpu_index: int = 0):
        # create instance managers
        self.server_manager = ServerManager(global_config, gpu_index)
        self.world_manager = WorldManager(global_config, self.server_manager.get())
        self.car_manager = CarManager(global_config)

    def __del__(self):
        # delete in sequence
        del self.car_manager
        del self.world_manager
        del self.server_manager

    def reset(self):
        # destroy all existing cars
        self.car_manager.destroy_all_cars()

        # reset managers
        self.server_manager.reset()
        self.world_manager.reset()
        self.car_manager.reset(self.world_manager.get(), self.server_manager.get(), self.server_manager.tm_port)

        # get observation
        return self.car_manager.get_observation()

    def step(self, action):
        # apply control for cars
        self.car_manager.apply_control(action)

        # tick world
        self.car_manager.sync_before_tick()
        self.world_manager.get().tick()
        self.car_manager.sync_after_tick()

        # get obs, reward, done, info
        return self.car_manager.get_observation(), self.car_manager.get_reward(), self.car_manager.get_done(), {}
