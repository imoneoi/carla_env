import carla
import cv2
import numpy as np


class CarMap:
    def __init__(self, car, options: dict, dt: float):
        self.options = {
            # Map
            "pixels_per_meter": 10,

            # Localization
            "gps_rate": 1.0,
            "gps_error_std": 3.0
        }
        self.options.update(options)

        # map
        self.map = None
        self.map_size = None
        self.world_offset = None



    def create_map(self, carla_map: carla.Map):
        # get map size
        waypoints = carla_map.generate_waypoints(2)
        margin = 50
        max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
        max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

        self.map_size = max_y - min_y, max_x - min_x
        self.world_offset = (min_y, min_x)

        # create map
        self.map = np.zeros(self.map_size, dtype=np.uint8)
