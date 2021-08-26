import argparse
import os

from wrapped_carla_env import create_wrapped_carla_single_car_env

import numpy as np
from tqdm import tqdm
import cv2


def record_dataset(
        save_path: str,
        n_steps: int = int(2e3)
):
    # save dataset path
    image_path = os.path.join(save_path, "obs/")
    os.makedirs(image_path, exist_ok=True)

    act_file_path = os.path.join(save_path, "act.npy")

    # action array
    act_array = np.zeros((n_steps, 2), dtype=np.float32)

    # step env
    env = create_wrapped_carla_single_car_env(global_config={}, gpu_index=0)
    obs = env.reset()
    for step in tqdm(range(n_steps)):
        next_obs, rew, done, info = env.step([None])
        # get act
        car_control = env.unwrapped.car_manager.cars[0].actor.get_control()
        act = np.array([car_control.throttle - car_control.brake, car_control.steer], dtype=np.float32)

        # save obs & act
        obs_cv = cv2.cvtColor(obs.transpose((1, 2, 0)), cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path + "{}.jpg".format(step), obs_cv)
        act_array[step] = act

        # next & done reset
        obs = next_obs
        if done:
            obs = env.reset()

    # write act
    np.save(act_file_path, act_array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("save_path", type=str, help="Path to save dataset")

    args = parser.parse_args()

    record_dataset(args.save_path)


if __name__ == "__main__":
    main()
