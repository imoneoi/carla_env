import json

import carla
import numpy as np


class CarManagerInitializer:
    def __init__(self,
                 global_options: dict):
        # default options
        self.options = {
            "num_cars": 1
        }

        self.options.update(global_options.get("car_manager", {}))

        # cars
        self.cars = None

    def get(self):
        return self.cars

    def reset(self, world: carla.World):
        # destroy existing cars
        if self.cars:
            for car in self.cars:


        # select car
        blueprint_library = world.get_blueprint_library()
        # filter only 4 wheel cars (to remove bicycles)
        car_blueprint_list = list(filter(lambda vehicle: int(vehicle.get_attribute("number_of_wheels")) == 4,
                                  blueprint_library.filter("vehicle.*")))

        # spawn cars
        spawn_point_list = world.get_map().get_spawn_points()
        np.random.shuffle(spawn_point_list)

        for _