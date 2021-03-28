from car.car import Car

import json

import carla
import numpy as np


class CarManager:
    def __init__(self,
                 global_options: dict):
        # default options
        self.options = {
            "num_controlled_cars": 1,
            "num_auto_cars": 0,

            "car_blueprint_list": None
        }

        self.options.update(global_options.get("car_manager", {}))

        # car options
        self.car_options = global_options.get("car", {})
        self.car_reward_weights = global_options.get("car_reward_weights", {})

        # cars
        self.cars = []
        self.uncontrolled_cars = []

        # debug info
        print("Car Manager Options: ", json.dumps(self.options))

    def __del__(self):
        self.destroy_all_cars()

    def get(self):
        return self.cars

    def reset(self,
              world: carla.World,
              client: carla.Client,
              tm_port: int):
        # destroy existing cars
        self.destroy_all_cars()

        # select car
        blueprint_library = world.get_blueprint_library()

        selected_list = self.options["car_blueprint_list"]
        if selected_list is None:
            # filter only 4 wheel cars (to remove bicycles)
            car_blueprint_list = list(filter(lambda vehicle: int(vehicle.get_attribute("number_of_wheels")) == 4,
                                      blueprint_library.filter("vehicle.*")))
        else:
            car_blueprint_list = [blueprint_library.find(name) for name in selected_list]

        # spawn cars (controlled)
        spawn_point_list = world.get_map().get_spawn_points()
        np.random.shuffle(spawn_point_list)

        for _ in range(self.options["num_controlled_cars"]):
            car_blueprint = np.random.choice(car_blueprint_list)
            # mark ego vehicle for hybrid physics mode
            car_blueprint.set_attribute("role_name", "hero")

            # try to find a spawn point
            car_actor = None
            spawn_point = None

            for pt in spawn_point_list:
                spawn_point = pt
                car_actor = world.try_spawn_actor(car_blueprint, spawn_point)
                if car_actor is not None:
                    break

            # if cannot spawn
            assert car_actor is not None, "Car cannot be spawned, spawn points ran out."

            # remove spawn point
            spawn_point_list.remove(spawn_point)

            # create car instance
            self.cars.append(Car(self.car_options, self.car_reward_weights, car_actor, world))

        # spawn cars (other)
        batch_op = []
        for _ in range(self.options["num_auto_cars"]):
            car_blueprint = np.random.choice(car_blueprint_list)
            car_blueprint.set_attribute("role_name", "autopilot")

            if not spawn_point_list:
                break

            spawn_point = spawn_point_list.pop()

            batch_op.append(
                carla.command.SpawnActor(car_blueprint, spawn_point)
                    .then(carla.command.SetAutopilot(carla.command.FutureActor, True, tm_port)))

        client.apply_batch_sync(batch_op, True)

        # update world
        world.tick()

    def destroy_all_cars(self):
        # destroy existing cars
        [car.destroy() for car in self.cars]
        [car.destroy() for car in self.uncontrolled_cars]

        self.cars = []
        self.uncontrolled_cars = []

    # synchronization
    def sync_before_tick(self):
        return [car.sync_before_tick() for car in self.cars]

    def sync_after_tick(self):
        return [car.sync_after_tick() for car in self.cars]

    # control
    def apply_control(self, action):
        return [car.apply_control(act) for car, act in zip(self.cars, action)]

    def get_observation(self):
        return [car.get_observation() for car in self.cars]

    def get_reward(self):
        return [car.get_reward() for car in self.cars]

    # done
    def get_done(self):
        return False
