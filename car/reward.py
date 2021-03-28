import carla


class CarReward:
    def __init__(self, car, weights: dict):
        self.car = car

        # init weights
        self.weights = {
            "collision": -100,

            "lane_invasion_solid": -10,
            "lane_invasion_double_solid": -20,
        }
        self.weights.update(weights)

        # collision debounce
        self.last_collision = False

    def get_reward(self):
        reward = 0

        # Part 1. Safety
        # collision penalty
        is_collision = len(self.car.collision_events) > 0
        if is_collision != self.last_collision:
            self.last_collision = is_collision
            if is_collision:
                reward += self.weights["collision"]

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

        return reward
