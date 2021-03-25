import json

import carla
import numpy as np


class WorldInitializer:
    def __init__(self,
                 global_options: dict,
                 client: carla.Client):
        # default options
        self.options = {
            "dt": 0.1,

            "map_list": client.get_available_maps(),
            "map_lifetime": 10
        }

        self.options.update(global_options.get("world", {}))

        # world
        self.client = client
        self.world = None

        # internal
        self.map_age = 0

        # debug info
        print("World Options: ", json.dumps(self.options))

    def get(self):
        return self.world

    def reset(self):
        # change map
        if (self.world is None) or (self.map_age >= self.options["map_lifetime"]):
            # create world
            self.world = self.client.load_world(np.random.choice(self.options["map_list"]))
            # set sync mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.options["dt"]
            self.world.apply_settings(settings)

            # clear map age
            self.map_age = 0

        self.map_age += 1
