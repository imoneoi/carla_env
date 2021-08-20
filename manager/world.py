import json

import carla
import numpy as np

from manager.server import ServerManager


class WorldManager:
    def __init__(self,
                 global_options: dict,
                 server_manager: ServerManager):
        # default options
        self.options = {
            "dt": 0.1,

            "map_list": [x for x in server_manager.get().get_available_maps()
                         if (not x.endswith("_Opt")) and (not "HD" in x)],
            "map_lifetime": 5,
            "server_lifetime": 10,

            "weather_list": [k for k, v in vars(carla.WeatherParameters).items()
                             if isinstance(v, carla.WeatherParameters)]
        }

        self.options.update(global_options.get("world", {}))

        # world
        self.server_manager = server_manager
        self.world = None

        # internal
        self.map_age = 0
        self.server_age = 0

        # debug info
        print("World Options: ", json.dumps(self.options))

    def get(self):
        return self.world

    def reset(self):
        # change map
        if (self.world is None) or (self.map_age >= self.options["map_lifetime"]):
            # reset server if required
            self.server_age += 1
            if self.server_age >= self.options["server_lifetime"]:
                self.server_manager.cleanup()
                exit(-1)

                self.server_age = 0

            # create world
            self.world = self.server_manager.get().load_world(np.random.choice(self.options["map_list"]))
            # set sync mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.options["dt"]

            settings.no_rendering_mode = True
            self.world.apply_settings(settings)

            # update world
            self.world.tick()

            # clear map age
            self.map_age = 0

        self.map_age += 1

        # change weather
        weather_name = np.random.choice(self.options["weather_list"])
        self.world.set_weather(getattr(carla.WeatherParameters, weather_name))
