from scripts.wrapped_carla_env import create_wrapped_carla_single_car_env

import cv2
import time


def main():
    env = create_wrapped_carla_single_car_env(
        global_config={},
        gpu_index=0
    )
    obs = env.reset()
    while True:
        act = env.action_space.sample()
        t_start = time.time()
        obs_next, rew, done, _ = env.step(act)
        t_elapsed = time.time() - t_start

        obs = obs_next
        if done:
            obs = env.reset()

        print(t_elapsed)


if __name__ == "__main__":
    main()
