import json
import re
import numpy as np
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

from manager.server import ServerManager

import carla
from carla import ColorConverter as cc

sys.path.append("..")
from agents.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.basic_agent import BasicAgent  # pylint: disable=import-error

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


class WorldManager:
    def __init__(self,
                global_options: dict,
                server_manager: ServerManager):
        # default options
        self.options = {
            "dt": 0.1,

            # TODO: choose maps from the list (No layered or HD Map)
            "map_list": [x for x in server_manager.get().get_available_maps()
                        if (not x.endswith("_Opt")) and (not "HD" in x)],
            "map_lifetime": 5,
            # TODO: prevent the memory leakage
            "server_lifetime": 10,

            "weather_blacklist": ['ClearNight', 'CloudyNight', 'WetNight', 'WetCloudyNight', 'SoftRainNight', 'MidRainyNight', 'HardRainNight'],
            # "weather_list": [k for k, v in vars(carla.WeatherParameters).items()
            #                 if isinstance(v, carla.WeatherParameters)]
        }

        self.options.update(global_options.get("world", {}))
        # self.options.update(global_options)

        # world
        self.server_manager = server_manager
        self.world = None
        self.map = None

        # internal
        self.map_age = 0
        self.server_age = 0

        # debug info
        print("World Options: ", json.dumps(self.options))

    def get(self):
        return self.world

    def reset(self):
        # TODO: change map when world is None or map age > map lifetime
        if (self.world is None) or (self.map_age >= self.options["map_lifetime"]):
            # reset server if required
            self.server_age += 1
            print("server age: ", self.server_age)
            if self.server_age >= self.options["server_lifetime"]:
                print("server age reaches the life limit!")
                self.server_manager.cleanup()
                exit(-1)

                self.server_age = 0

            # create world
            self.world = self.server_manager.get().load_world(np.random.choice(self.options["map_list"]))
            print("get world")
            self.map = self.world.get_map()
            print("get map")
            # set sync mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.options["dt"]

            settings.no_rendering_mode = True
            self.world.apply_settings(settings)
            print("set sync mode")

            # update world
            self.world.tick()
            print("update world")

            # clear map age
            self.map_age = 0

        
        self.map_age += 1

        # change weather
        weather_whitelist = [k for k, v in vars(carla.WeatherParameters).items()
                if isinstance(v, carla.WeatherParameters) and k not in self.options["weather_blacklist"]]
        print(weather_whitelist)
        # [item for item in self.options["weather_list"] if item.id not in self.options["car_blueprint_blacklist"]]
        weather_name = np.random.choice(weather_whitelist)
        self.world.set_weather(getattr(carla.WeatherParameters, weather_name))
