import carla
import numpy as np


class CarReward:
    def __init__(self, car, weights: dict, options: dict):
        self.car = car

        # init weights
        self.weights = {
            "collision": -100,

            "lane_invasion_solid": -10,
            "lane_invasion_double_solid": -20,

            "over_speed_per_tick": -0.5,
            
            "stay_per_tick": -0.1,
            "forward": 0.25,  # every meter
        }
        self.weights.update(weights)

        # init options
        self.options = {
            "speed_limit": 10,  # m/s
            "stay_speed": 0.1,  # m/s
            "collision_debounce_ticks": 30,  # 1 collision / 3sec
        }
        self.options.update(options)

        # collision debounce
        self.last_collision_elapsed = self.options["collision_debounce_ticks"]

        # forward
        self.last_location = None

    def get_reward(self):
        reward = 0.0

        # Part 1. Safety
        # collision penalty
        is_collision = len(self.car.collision_events) > 0
        if is_collision:
            if self.last_collision_elapsed >= self.options["collision_debounce_ticks"]:
                reward += self.weights["collision"]

            self.last_collision_elapsed = 0
        else:
            self.last_collision_elapsed += 1

        # Part 2. Traffic Rules
        # lane invasion collision
        is_invasion_solid = False
        is_invasion_double_solid = False
        for event in self.car.lane_invasion_events:
            for marking in event.crossed_lane_markings:
                is_invasion_solid |= marking.type == carla.LaneMarkingType.Solid
                is_invasion_double_solid |= marking.type == carla.LaneMarkingType.SolidSolid

        reward += is_invasion_solid * self.weights["lane_invasion_solid"]
        reward += is_invasion_double_solid * self.weights["lane_invasion_double_solid"]

        # red light
        # WIP

        # over speed
        velocity = self.car.actor.get_velocity()
        velocity = np.sqrt(velocity.x ** 2 + velocity.y ** 2)
        is_over_speed = velocity > self.options["speed_limit"]

        reward += is_over_speed * self.weights["over_speed_per_tick"]

        # stay
        is_stay = velocity < self.options["stay_speed"]
        reward += is_stay * self.weights["stay_per_tick"]

        # Part 3. Forward
        location = self.car.actor.get_location()
        if self.last_location is not None:
            forward_dist = np.sqrt((location.x - self.last_location.x) ** 2 + (location.y - self.last_location.y) ** 2)
            reward += self.weights["forward"] * forward_dist

        self.last_location = location

        return reward
