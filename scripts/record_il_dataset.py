import argparse
import os
import multiprocessing as mp
import random

from wrapped_carla_env import create_wrapped_carla_single_car_env

import carla
from tqdm import tqdm
import cv2
import json


def record_dataset(
        save_path: str,
        n_steps: int,
        gpu_index: int,

        eps: float = 0.2,
        rand_action_range: float = 0.5,
):
    # set gpu index
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    # step env
    env = create_wrapped_carla_single_car_env(global_config={}, gpu_index=gpu_index)
    obs = env.reset()
    for step in tqdm(range(n_steps)):
        if random.random() < eps:
            # eps-greedy act
            act = env.action_space.sample() * rand_action_range

            env.unwrapped.server_manager.client.apply_batch_sync([
                carla.command.SetAutopilot(env.unwrapped.car_manager.cars[0].actor.id, False,
                                           env.unwrapped.server_manager.tm_port)
            ])
            next_obs, rew, done, info = env.step([act])
            env.unwrapped.server_manager.client.apply_batch_sync([
                carla.command.SetAutopilot(env.unwrapped.car_manager.cars[0].actor.id, True,
                                           env.unwrapped.server_manager.tm_port)
            ])
        else:
            next_obs, rew, done, info = env.step([None])
            # get act
            car_control = env.unwrapped.car_manager.cars[0].actor.get_control()
            act = [car_control.throttle - car_control.brake, car_control.steer]

            # save obs & act
            obs_cv = cv2.cvtColor(obs.transpose((1, 2, 0)), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_path, "{}.jpg".format(step)), obs_cv)
            with open(os.path.join(save_path, "{}.json".format(step)), "wt") as f:
                json.dump({"act": act}, f)
                f.close()

        # next & done reset
        obs = next_obs
        if done:
            obs = env.reset()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("save_path", type=str, help="Path to save dataset")
    parser.add_argument("--n", type=int, default=int(2e5), help="Number of steps to collect")
    parser.add_argument("--n_jobs", type=int, default=15, help="Number of worker processes")
    parser.add_argument("--devices", type=str, default="0,1,2", help="GPUs to use")

    args = parser.parse_args()

    # start workers
    devices = list(map(int, args.devices.split(",")))
    n_jobs = args.n_jobs
    step_per_job = args.n // n_jobs

    processes = []
    for job_id in range(n_jobs):
        save_path = os.path.join(args.save_path, str(job_id))
        os.makedirs(save_path, exist_ok=True)

        gpu_index = devices[job_id % len(devices)]

        processes.append(mp.Process(target=record_dataset, kwargs={
            "save_path": save_path,
            "n_steps": step_per_job,
            "gpu_index": gpu_index
        }))

    # start & wait all processes
    [proc.start() for proc in processes]
    [proc.join() for proc in processes]


if __name__ == "__main__":
    main()
