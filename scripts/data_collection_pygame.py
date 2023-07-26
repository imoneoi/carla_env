import argparse
import os
import multiprocessing as mp
import random
import datetime
import sys
import re

from wrapped_carla_env import create_wrapped_carla_single_car_env

import carla
from tqdm import tqdm
import cv2
import json
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')
import ipdb
import pdb

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


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)
            
#

def record_dataset(
        save_path: str,
        n_steps: int,
        timeout: int,
        gpu_index: int,

        eps: float,
        rand_action_range: float,
):
    # set gpu index
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    
    # fix the map
    global_config = {
        "server": {
            "resolution_x": 640, 
            "resolution_y": 640, 
            "quality": "Low"
            }, 
        "world": {
            "dt": 0.05,
            "map_list" : ['Town02'], 
            "map_lifetime" : n_steps
            }, 
        "car_manager": {
            "num_auto_cars": 0, 
            "num_walkers": 0, 
            "car_blueprint_list": ["vehicle.audi.tt"]# ["vehicle.tesla.model3"]
            }, 
        "perception": {
            "target_size": (640, 640)
        },
        "car": {
            "camera_x": 640, 
            "camera_y": 640, 
            "camera_postprocess": False, 
            "bev_camera_x": 640, 
            "bev_camera_y": 640, 
            "bev_fov": 90, 
            "bev_camera_height": 40.0
            }
        }

    # step env
    env = create_wrapped_carla_single_car_env(global_config=global_config, gpu_index=gpu_index)
    
    # step calculator bridging this process and the last exiting one
    step_file = "./step.txt"
    try:
        # Try opening the existing step file
        with open(step_file, "r+") as f:
            digits_only = re.sub(r'\D', '', f.read())
            if digits_only:
                prev_step = int(digits_only)
            else:
                prev_step = 0
            f.truncate(prev_step)
    except FileNotFoundError:
        # Create a new step file if it doesn't exist
        with open(step_file, "w") as f:
            f.truncate(0)
            prev_step = 0
            
    pygame.init()
    pygame.font.init()
    world = None
            
    obs, bev_obs, bev_seg_obs = env.reset()
    print("initial reset!")
    for step in tqdm(range(1, n_steps + 1)):
        if random.random() < eps:
            # eps-greedy act
            act = env.action_space.sample() * rand_action_range

            env.unwrapped.server_manager.client.apply_batch_sync([
                carla.command.SetAutopilot(env.unwrapped.car_manager.cars[0].actor.id, False, env.unwrapped.server_manager.tm_port)
            ])
            (next_obs, next_bev_obs, next_bev_seg_obs), rew, done, info = env.step([act])
            env.unwrapped.server_manager.client.apply_batch_sync([
                carla.command.SetAutopilot(env.unwrapped.car_manager.cars[0].actor.id, True, env.unwrapped.server_manager.tm_port)
            ])
        else:
            (next_obs, next_bev_obs, next_bev_seg_obs), rew, done, info = env.step([None])
            # get act
            car_control = env.unwrapped.car_manager.cars[0].actor.get_control()
            act = [car_control.throttle - car_control.brake, car_control.steer]

            # save obs, bev_obs & act
            # obs_cv = cv2.cvtColor(obs.transpose((1, 2, 0)), cv2.COLOR_RGB2BGR)
            # bev_obs_cv = cv2.cvtColor(bev_obs.transpose((1, 2, 0)), cv2.COLOR_RGB2BGR)
            bev_seg_obs_cv = cv2.cvtColor(bev_seg_obs.transpose((1, 2, 0)), cv2.COLOR_RGB2BGR)
            # cv2.imwrite(os.path.join(save_path, "front_rgb_{}.jpg".format(prev_step + step)), obs_cv)
            # cv2.imwrite(os.path.join(save_path, "bev_rgb_{}.jpg".format(prev_step + step)), bev_obs_cv)
            cv2.imwrite(os.path.join(save_path, "seg_bev_rgb_{}.jpg".format(prev_step + step)), bev_seg_obs_cv)
            
            # do not erase the previously recorded data
            with open(os.path.join(save_path, "{}.json".format(prev_step + step)), "wt") as f:
                ego_location = env.unwrapped.car_manager.cars[0].actor.get_location()
                position = {
                            "x": ego_location.x,
                            "y": ego_location.y
                            # "z": ego_location.z
                        }
                ego_velocity = env.unwrapped.car_manager.cars[0].actor.get_velocity()
                velocity = {
                            "x": ego_velocity.x,
                            "y": ego_velocity.y
                            # "z": ego_velocity.z
                        }
                ego_heading = env.unwrapped.car_manager.cars[0].actor.get_transform().rotation.yaw
                terminal = done or (step % timeout == 0)
                json.dump({"act": act, "pos": position, "yaw": ego_heading, "vel": velocity, "traj_terminal": terminal}, f, indent=4)
                f.close()

        # next & done reset
        obs, bev_obs, bev_seg_obs = next_obs, next_bev_obs, next_bev_seg_obs
        # Save the collection step to a file before potential exiting
        with open(step_file, "w") as f:
            f.write(str(prev_step + step))
        if terminal:
            print("done!")
            obs, bev_obs, bev_seg_obs = env.reset()
            print("reset!")


def main():
    nowTime = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", default="bev_23070700", type=str, help="Path to save dataset")
    parser.add_argument("--n", type=int, default=int(2e1), help="Number of steps to collect")
    parser.add_argument("--n_jobs", type=int, default=10, help="Number of worker processes")
    parser.add_argument("--timeout", type=int, default=1000, help="Reset at timeout steps")
    parser.add_argument("--devices", type=str, default="0", help="GPUs to use")
    parser.add_argument("--eps", type=float, default=0.2, help="Probability to randomly perturb AutoPilot actions")
    parser.add_argument("--rand_action_range", type=float, default=0.1, help="Perturbation range")

    args = parser.parse_args()

    # start workers
    devices = list(map(int, args.devices.split(",")))
    n_jobs = args.n_jobs
    step_per_job = args.n // n_jobs

    processes = []
    for job_id in range(n_jobs):
        save_path = os.path.join("../dataset/", args.save_path, str(job_id))
        os.makedirs(save_path, exist_ok=True)

        gpu_index = devices[job_id % len(devices)]

        processes.append(mp.Process(target=record_dataset, kwargs={
            "save_path": save_path,
            "n_steps": step_per_job,
            "timeout": args.timeout,
            "gpu_index": gpu_index,
            "eps": args.eps,
            "rand_action_range": args.rand_action_range
        }))

    # start & wait all processes
    [proc.start() for proc in processes]
    [proc.join() for proc in processes]


if __name__ == "__main__":
    main()
