from car.car import Car
from manager.world import WorldManager

import json

import carla
import numpy as np


class CarManager:
    def __init__(self,
                 global_options: dict,
                 world_manager: WorldManager):
        # world manager
        self.world_manager = world_manager
        # default options
        self.options = {
            "num_controlled_cars": 1,
            "num_auto_cars": 100,
            "num_walkers": 200,

            "car_blueprint_list": None,
            "car_blueprint_blacklist": ["vehicle.chargercop2020.chargercop2020",
                                        "vehicle.charger2020.charger2020",
                                        "vehicle.mercedesccc.mercedesccc"]  # FIXME: See docs/known_bugs.md
        }

        self.options.update(global_options.get("car_manager", {}))

        # car options
        self.car_options = global_options.get("car", {})
        self.car_reward_weights = global_options.get("car_reward_weights", {})

        self.client = None
        # cars
        self.cars = []
        self.uncontrolled_cars = []

        # walkers
        self.walkers = []
        self.walker_controllers = []

        # debug info
        print("Car Manager Options: ", json.dumps(self.options))

    def __del__(self):
        self.destroy_all_cars()

    def get(self):
        return self.cars

    def reset(self,
              client: carla.Client,
              tm_port: int):
        # destroy existing cars
        self.destroy_all_cars()
        # assign client
        self.client = client

        # select car
        world = self.world_manager.get()
        map = self.world_manager.map

        blueprint_library = world.get_blueprint_library()

        selected_list = self.options["car_blueprint_list"]
        if selected_list is None:
            # filter only 4 wheel cars (to remove bicycles)
            # car_blueprint_list = [item for item in blueprint_library.filter("vehicle.*")
            #                       if int(item.get_attribute("number_of_wheels")) == 4]

            # do not filter, include bicycles & motorcycles
            car_blueprint_list = list(blueprint_library.filter("vehicle.*"))
        else:
            car_blueprint_list = [blueprint_library.find(name) for name in selected_list]

        # remove blacklist
        car_blueprint_list = [item for item in car_blueprint_list
                              if item.id not in self.options["car_blueprint_blacklist"]]

        # spawn cars (controlled)
        spawn_point_list = map.get_spawn_points()
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

        result = client.apply_batch_sync(batch_op, True)

        # record uncontrolled cars
        self.uncontrolled_cars = [item.actor_id for item in result if not item.error]

        # spawn walkers
        walker_blueprint_list = world.get_blueprint_library().filter("walker.pedestrian.*")
        batch_op = []
        for _ in range(self.options["num_walkers"]):
            walker_blueprint = np.random.choice(walker_blueprint_list)
            if walker_blueprint.has_attribute("is_invincible"):
                walker_blueprint.set_attribute("is_invincible", "false")
            batch_op.append(carla.command.SpawnActor(walker_blueprint,
                                                     carla.Transform(world.get_random_location_from_navigation())))

        result = client.apply_batch_sync(batch_op, True)

        self.walkers = [item.actor_id for item in result if not item.error]
        # spawn walker controllers
        walker_controller_blueprint = world.get_blueprint_library().find("controller.ai.walker")
        batch_op = []
        for walker in self.walkers:
            batch_op.append(carla.command.SpawnActor(walker_controller_blueprint, carla.Transform(), walker))

        result = client.apply_batch_sync(batch_op, True)

        self.walker_controllers = [item.actor_id for item in result if not item.error]
        [controller.start() for controller in world.get_actors(self.walker_controllers)]

        # update world
        world.tick()

    def destroy_all_cars(self):
        # destroy existing cars
        [car.destroy() for car in self.cars]

        self.cars = []

        # destroy uncontrolled cars
        if self.client:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.uncontrolled_cars])

        self.uncontrolled_cars = []

        # destroy walkers
        if self.client:
            for controller in self.world_manager.get().get_actors(self.walker_controllers):
                controller.stop()
            self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in self.walker_controllers])
            self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in self.walkers])

        self.walkers = []
        self.walker_controllers = []

        # clear client
        self.client = None

    # synchronization
    def sync_before_tick(self):
        return [car.sync_before_tick() for car in self.cars]

    def sync_after_tick(self):
        return [car.sync_after_tick() for car in self.cars]

    # control
    def apply_control(self, action):
        if isinstance(action, np.ndarray):
            action = action.reshape(len(self.cars), -1)

        return [car.apply_control(act) for car, act in zip(self.cars, action)]

    def get_observation(self):
        result = None
        for car in self.cars:
            obs = car.get_observation()
            if result is None:
                result = [[] for _ in range(len(obs))]
            for idx, v in enumerate(obs):
                result[idx].append(v)

        return result

    def get_reward_done(self):
        rewards = []
        done = True
        for car in self.cars:
            r, d = car.get_reward_done(self.world_manager.map)

            rewards.append(r)
            done &= d

        return rewards, done
