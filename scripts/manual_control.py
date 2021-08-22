from scripts.util.joystick import Joystick
from carla_env import CarlaEnv
from wrapped_carla_env import BiasedAction

import time
import carla
import pygame

import numpy as np


class ManualInterface:
    def __init__(self, env: CarlaEnv):
        # create env
        self.env = env

        self.obs = None
        self.rew = None
        self.done = False
        self.reset_env()

        self.total_reward = None

        # init pygame
        pygame.init()

        self.running = True
        self.surface = None

        # font
        self.font = pygame.font.Font(pygame.font.get_default_font(), 30)

        # control
        self.joysticks = None
        self.reset_flag = False

        # tps counter
        self.tps_total_time = 0
        self.tps_total_frame = 0

    def __del__(self):
        pygame.quit()

    def reset_env(self):
        self.obs = self.env.reset()
        self.rew = None
        self.done = False

        self.total_reward = None

    def get_action(self):
        # init joysticks
        if self.joysticks is None:
            self.joysticks = [Joystick(fn) for fn in Joystick.list_devices()]

        # pump events before get
        pygame.event.pump()

        # get joysticks
        act = []
        for js in self.joysticks:
            accel = -js.axes["ry"]  # Left lever, L <--> R
            steer = js.axes["x"]   # Right lever, U <--> D
            reverse = js.buttons["tl"]    # LB

            # act.append(carla.VehicleControl(
            #     throttle=max(0.0, accel),
            #     brake=-min(0.0, accel),
            #
            #     steer=steer,
            #     reverse=reverse
            # ))
            act.append(np.array([accel, steer], dtype=np.float32))

        # check if reset
        is_reset = sum([js.buttons["y"] for js in self.joysticks])
        is_reset |= self.done
        if is_reset:
            if not self.reset_flag:
                self.reset_env()
                self.reset_flag = True
        else:
            self.reset_flag = False

        return act

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return

    def on_update(self):
        # update env
        act = self.get_action()

        start_time = time.time()
        self.obs, self.rew, self.done, _, = self.env.step(act)
        elapsed = time.time() - start_time

        # update total reward
        if self.total_reward is None:
            self.total_reward = np.array(self.rew)
        else:
            self.total_reward += np.array(self.rew)

        # tps counter
        self.tps_total_time += elapsed
        self.tps_total_frame += 1
        if self.tps_total_frame >= 100:
            print("TPS: {}".format(self.tps_total_frame / self.tps_total_time))

            self.tps_total_frame = 0
            self.tps_total_time = 0

    def on_render(self):
        _, h, w = self.obs[0][0].shape

        # init surface
        if self.surface is None:
            n_cars = len(self.obs)
            n_cameras = len(self.obs[0])

            self.surface = pygame.display.set_mode((w * n_cameras, h * n_cars), pygame.HWSURFACE | pygame.DOUBLEBUF)

        # show images
        y = 0
        for cam, rew in zip(self.obs, self.total_reward):
            # draw car cam images
            x = 0
            for cam_img in cam:
                if cam_img.shape[0] < 3:
                    # pad channel
                    padded_cam_img = np.concatenate([
                        cam_img,
                        np.zeros((1, *cam_img.shape[1:]), dtype=cam_img.dtype)], axis=0)
                else:
                    padded_cam_img = cam_img

                cam_surf = pygame.surfarray.make_surface(padded_cam_img.transpose(2, 1, 0))
                self.surface.blit(cam_surf, (x, y))

                x += w

            # draw reward
            rew_surf = self.font.render("Reward: {:.2f}".format(rew), True, (0, 0, 255))
            self.surface.blit(rew_surf, (10, y + 10))

            y += h

        # update display
        pygame.display.update()

    def run(self, fps: int = None):
        clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                self.on_event(event)

            self.on_update()
            self.on_render()

            if fps:
                clock.tick(fps)


def main():
    dt = 0.1
    global_options = {
        "world": {
            "dt": dt
        }
    }
    env = CarlaEnv(global_options, 0)

    ui = ManualInterface(env)
    ui.run(int(1.0 / dt))


if __name__ == '__main__':
    main()
