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
import ipdb
import pdb

# import sys

# # Get the step number from the command line argument
# if len(sys.argv) > 1:
#     INITIAL_STEP = int(sys.argv[1])
# else:
#     INITIAL_STEP = 0

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
