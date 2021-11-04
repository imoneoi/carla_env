import os

import carla
import cv2
import numpy as np


class CarMap:
    def __init__(self,
                car,
                world: carla.World,
                options: dict):
        self.options = {
            # Map
            "pixels_per_meter": 10,

            # Localization
            "gps_rate": 1.0,
            "gps_error_std": 5.0
        }
        self.options.update(options)

        # car and world
        self.car = car
        self.world = world

        # map
        self.pixels_per_meter = self.options["pixels_per_meter"]

        self.map = None
        self.map_size = None
        self.world_offset = None

        # load / draw map
        map_filename = os.path.join(".cache", "static_map_{}.npy".format(self.world.get_map().name))
        if os.path.isfile(map_filename):
            self.map = np.load(map_filename)
        else:
            self.create_map(self.world.get_map())
            np.save(map_filename, self.map)

    def _world_to_pixel(self, location):
        """Converts the world coordinates to pixel coordinates"""
        x = self.pixels_per_meter * (location.x - self.world_offset[0])
        y = self.pixels_per_meter * (location.y - self.world_offset[1])
        return [int(x), int(y)]

    def create_map(self, carla_map: carla.Map, precision: float = 0.05):
        # map drawing functions
        def lateral_shift(transform, shift):
            """Makes a lateral shift of the forward vector of a transform"""
            transform.rotation.yaw += 90
            return transform.location + shift * transform.get_forward_vector()

        def draw_topology(img, carla_topology, index):
            topology = [x[index] for x in carla_topology]
            topology = sorted(topology, key=lambda w: w.transform.location.z)
            set_waypoints = []
            for waypoint in topology:
                waypoints = [waypoint]

                # Generate waypoints of a road id. Stop when road id differs
                nxt = waypoint.next(precision)
                if len(nxt) > 0:
                    nxt = nxt[0]
                    while nxt.road_id == waypoint.road_id:
                        waypoints.append(nxt)
                        nxt = nxt.next(precision)
                        if len(nxt) > 0:
                            nxt = nxt[0]
                        else:
                            break
                set_waypoints.append(waypoints)

            # Draw Roads
            for waypoints in set_waypoints:
                road_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
                road_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

                polygon = road_left_side + [x for x in reversed(road_right_side)]
                polygon = [self._world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    cv2.fillPoly(img, np.array([polygon]), 1)

        def get_map_size():
            waypoints = carla_map.generate_waypoints(2)
            margin = 50
            max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
            max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
            min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
            min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

            self.map_size = int((max_y - min_y) * self.pixels_per_meter), int((max_x - min_x) * self.pixels_per_meter)
            self.world_offset = (min_x, min_y)

        # get map size
        get_map_size()

        # create map
        self.map = np.zeros(self.map_size, dtype=np.uint8)

        # draw topology
        draw_topology(self.map, carla_map.get_topology(), 0)
