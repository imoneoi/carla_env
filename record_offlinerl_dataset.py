import argparse
import os
import multiprocessing as mp
import random

from wrapped_carla_env import create_wrapped_carla_single_car_env
from car.reward import CarReward

import carla
from tqdm import tqdm
import cv2
import json

def record_dataset(
        save_path: str,
        n_steps: int,
        gpu_index: int,

        eps: float = 0.2,
        rand_action_range: float = 0.1,
):
    # set gpu index
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    global_config = {}
    # TODO: when recording real dataset, freeze the map and ensure its lifetime to be enough
    if save_path.split('/')[-1] == "real":
        global_config = {"world":{"map_list" : ['Town01'], "map_lifetime" : n_steps}}

    # # TODO: when recording real dataset, fix the map and ensure its lifetime to be enough
    # global_config = {"world":{"map_list" : ['Town01'], "map_lifetime" : n_steps}}
    
    # TODO: create an env for single car
    env = create_wrapped_carla_single_car_env(global_config=global_config, gpu_index=gpu_index)
    obs = env.reset()
    for step in tqdm(range(n_steps)):
        if random.random() < eps:
            # eps-greedy act
            act = env.action_space.sample() * rand_action_range

            # TODO: Close Autopilot and choose action randomly
            env.unwrapped.server_manager.client.apply_batch_sync([
                carla.command.SetAutopilot(env.unwrapped.car_manager.cars[0].actor.id, False, env.unwrapped.server_manager.tm_port)
            ])
            next_obs, rew, done, info = env.step([act])
            # TODO: Open Autopilot anew
            env.unwrapped.server_manager.client.apply_batch_sync([
                carla.command.SetAutopilot(env.unwrapped.car_manager.cars[0].actor.id, True, env.unwrapped.server_manager.tm_port)
            ])
        else:
            next_obs, rew, done, info = env.step([None])
            # TODO: get action and reward from Autopilot
            car_control = env.unwrapped.car_manager.cars[0].actor.get_control()
            act = [car_control.throttle - car_control.brake, car_control.steer]
            map = env.unwrapped.world_manager.get().get_map()
            rew, _ = env.unwrapped.car_manager.cars[0].get_reward_done(map)

            # save observation(jpg) & action & reward
            obs_cv = cv2.cvtColor(obs.transpose((1, 2, 0)), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_path, "s{}.jpg".format(step)), obs_cv)
            with open(os.path.join(save_path, "a{}-r{}.json".format(step, step)), "wt") as f:
                json.dump({"action": act, "reward": rew}, f)
                f.close()

        # next & done reset
        obs = next_obs
        # save next observation(jpg)
        next_obs_cv = cv2.cvtColor(obs.transpose((1, 2, 0)), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_path, "s{}_.jpg".format(step)), next_obs_cv)
        if done:
            obs = env.reset()


def main():
    parser = argparse.ArgumentParser()
    # TODO: path for dataset recording
    parser.add_argument("save_path", type=str, default="real", help="Path to save dataset")
    parser.add_argument("--n", type=int, default=int(2e5), help="Number of steps to collect")
    parser.add_argument("--n_jobs", type=int, default=10, help="Number of worker processes")
    parser.add_argument("--devices", type=str, default="0,1", help="GPUs to use")

    args = parser.parse_args()

    # start workers
    devices = list(map(int, args.devices.split(",")))
    n_jobs = args.n_jobs
    step_per_job = args.n // n_jobs

    processes = []
    for job_id in range(n_jobs):
        save_path = os.path.join("./dataset", args.save_path, str(job_id))
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
