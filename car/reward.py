import carla
import numpy as np


def vec3d_to_np(v):
    return np.array([v.x, v.y, v.z])


def distance_to_line(A, B, p):
    num   = np.linalg.norm(np.cross(B - A, A - p))
    denom = np.linalg.norm(B - A)
    if np.isclose(denom, 0):
        return np.linalg.norm(p - A)
    return num / denom


class CarReward:
    def __init__(self, car, weights: dict, options: dict):
        self.car = car

        # init weights
        self.weights = {
            "collision": -100,

            "lane_invasion_solid": -10,
            "lane_invasion_double_solid": -20,

            "speed": 1,
            "dist_center": 1,
            "angle": 1
        }
        self.weights.update(weights)

        # init options
        self.options = {
            "speed_stay": 0.3,  # m/s
            "speed_min": 5,  # m/s
            "speed_target": 6,  # m/s
            "speed_max": 7,  # m/s

            "angle_max": 30.0 / 180.0 * np.pi,  # rad
            "dist_center_max": 3.0,  # m

            "waypoint_precision": 0.5,
            "collision_debounce_ticks": 30,  # 1 collision / 3sec
        }
        self.options.update(options)

        # collision debounce
        self.last_collision_elapsed = self.options["collision_debounce_ticks"]

    def get_reward_done(self, world_map: carla.Map):
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

        # Part 3. Shaped reward
        car_location = vec3d_to_np(self.car.actor.get_location())
        car_vel = vec3d_to_np(self.car.actor.get_velocity())
        car_heading = vec3d_to_np(self.car.actor.get_transform().get_forward_vector())
        car_waypoint = world_map.get_waypoint(
            self.car.actor.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        car_next_waypoint = car_waypoint.next(self.options["waypoint_precision"])
        car_next_waypoint = car_next_waypoint[0] if len(car_next_waypoint) > 0 else car_waypoint


        # speed
        car_speed = np.linalg.norm(car_vel)
        if car_speed < self.options["speed_min"]:
            reward_speed = car_speed / self.options["speed_min"]
        elif car_speed > self.options["speed_target"]:
            reward_speed = 1.0 - (car_speed - self.options["speed_target"]) / (self.options["speed_max"] - self.options["speed_target"])
        else:
            reward_speed = 1.0
        # dist to center
        dist_to_center = distance_to_line(
            vec3d_to_np(car_waypoint.transform.location),
            vec3d_to_np(car_next_waypoint.transform.location),
            car_location)
        reward_dist_center = np.clip(1.0 - dist_to_center / self.options["dist_center_max"], 0.0, 1.0)
        # angle
        waypoint_forward = vec3d_to_np(car_waypoint.transform.rotation.get_forward_vector())
        angle = np.dot(waypoint_forward, car_heading) / (np.linalg.norm(waypoint_forward) * np.linalg.norm(car_heading))
        angle = np.arccos(np.clip(angle, -1, 1))  # improve numerical stability

        reward_angle = np.clip(1.0 - angle / self.options["angle_max"], 0.0, 1.0)

        # no shaped reward if stay
        if car_speed < self.options["speed_stay"]:
            reward_dist_center = 0
            reward_angle = 0

        # total reward
        reward += self.weights["speed"] * reward_speed + \
                  self.weights["dist_center"] * reward_dist_center + \
                  self.weights["angle"] * reward_angle
        assert not np.isnan(reward)

        print("speed {:.2f} dist {:.2f} angle {:.2f}".format(reward_speed, reward_dist_center, reward_angle))

        # Done: critical infraction
        done = is_collision

        return reward, done
