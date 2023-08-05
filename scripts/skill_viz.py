"""Parking State Information (state observationn appended with goal):

State Size: 12
0: x
1: y
2: vx
3: vy
4: cos_h
5: sin_h
6: goal x
7: goal y
8: goal vx
9: goal vy
10: goal cos_h
11: goal sin_h

Parking Spots: 
0.14 or -0.14 y, 
+/- 0.26, 0.22, 0.18, 0.14, 0.3, 0.1, 0.02

"""


from collections import defaultdict, Counter
import torch
from time import time 
from torch.distributions import Categorical
import gym
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import dataset, DataLoader
import json
import matplotlib.colors as mcolors
import random
import skill_utils
import pickle
from arguments import args
import ipdb


MIN_SKILL_LENGTH = 2
# TODO: change rollout dataset directory
ROLLOUTS_DIR = "../../../CORL2017ImitationLearningData/AgentHuman/SeqVal"

#set random seeds
torch.manual_seed(args.random_seed)
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed) 


# def plot_parking_spots():
#     for spot in range(15):
#         x = (spot - 15 // 2) * (4.0) - 2.0
#         plt.vlines(x=x-2, ymin=10, ymax=18)
#         plt.vlines(x=x+2, ymin=10, ymax=18)
#         plt.vlines(x=x-2, ymin=-10, ymax=-18)
#         plt.vlines(x=x+2, ymin=-10, ymax=-18)


def plot_traj(args_dict, states, lengths, fn_timestamp):
    for episode in range(states.shape[0]):
        plt.plot([100*state[0] for state in states[episode][:lengths[episode]]], [100*state[1] for state in states[episode][:lengths[episode]]], color="blue", alpha=0.75)
    # plot_parking_spots()
    plt.savefig("../figures/"+args_dict["env_name"]+"/trajectories_"+fn_timestamp, dpi=300)
    plt.close()

#filter by max peak reward as heuristic
def plot_rollouts(args_dict):
    if(args_dict["sample"]):
        logs = skill_utils.get_sample_rollouts(args_dict)
    else:
        logs = skill_utils.get_target_rollouts(args_dict)
    logs_fn = "samples/"+args_dict["env_name"]+"_"+str(skill_utils.NUM_TYPE_SKILLS)+"_skills_sample_"+str(args_dict["sample"])+".pkl"
    pickle.dump(logs, open(logs_fn, "wb"))
    for skill in logs["states"]:
        for demo_episode in logs["states"][skill]:
            plt.plot([state[0]*100 for state in demo_episode], [state[1]*100 for state in demo_episode], color="blue", alpha=0.5)
            # plt.scatter([state[-6]*100 for state in demo_episode], [state[-5]*100 for state in demo_episode], color="green", label="goal", alpha=0.5, s=150)
    # plot_parking_spots()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    return logs_fn


def plot_skills(args_dict):
    fn_timestamp = str(int(time()))
    latent_skills_dict, states, lengths = skill_utils.get_all_skills(args_dict)
    plot_traj(args_dict, states, lengths, fn_timestamp)

    for skill in latent_skills_dict["states"].keys():
        for idx in range(len(latent_skills_dict["states"][skill])):
            traj = latent_skills_dict["states"][skill][idx]
            x1, x2 = latent_skills_dict["compile"][skill][idx]
            # ipdb.set_trace()

            # Extract the x and y coordinates of the starting and terminal points
            start_point = np.array(traj[x1])
            terminal_point = np.array(traj[x2-1])

            # Extract the x and y coordinates of the middle points inside the segment
            middle_points_x = np.array([state[0] for state in traj[x1:x2]])
            middle_points_y = np.array([state[1] for state in traj[x1:x2]])
            # Get the speed values of each middle point
            middle_points_v = np.array([state[2] for state in traj[x1:x2]])

            # Set the colormap function
            cmap = plt.cm.viridis

            if args_dict["viz_type"] == "pos_agnostic":
                # Plot the middle points, setting colors based on the speed values
                plt.scatter(middle_points_x - start_point[0], middle_points_y - start_point[1], c=middle_points_v, cmap=cmap, alpha=0.75, s=1)

                # Plot the starting and terminal points with distinct colors
                plt.plot(0, 0, marker='o', markersize=2, color='mediumvioletred')
                plt.plot(terminal_point[0] - start_point[0], terminal_point[1] - start_point[1], marker='o', markersize=2, color='blue')

                # # Plot the middle points inside the segment
                # plt.plot(middle_points_x, middle_points_y, color="gray", alpha=0.75)
            else:
                # Plot the middle points, setting colors based on the speed values
                plt.scatter(100 * middle_points_x, 100 * middle_points_y, c=middle_points_v, cmap=cmap, alpha=0.75, s=1)

                # Plot the starting and terminal points with distinct colors
                plt.plot(100 * start_point[0], 100 * start_point[1], marker='o', markersize=2, color='mediumvioletred')
                plt.plot(100 * terminal_point[0], 100 * terminal_point[1], marker='o', markersize=2, color='blue')

                # # Plot the middle points inside the segment
                # plt.plot(100 * middle_points_x, 100 * middle_points_y, color="gray", alpha=0.75)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Latent: " + str(skill) + ", Num Episodes: " + str(len(latent_skills_dict["states"][skill])))
        # Add a color bar to the plot to represent the speed range
        plt.colorbar(label='Speed')
        plt.savefig("../figures/" + args_dict["env_name"] + "/latent_" + str(skill) + "_" + str(args_dict["compile_dir"].split("/")[-1]) + "_" + fn_timestamp + ".png", dpi=300)
        plt.close()

    return


def viz_logs(args_dict, logs_fn, skill_type="compile"):
    logs = pickle.load(open(logs_fn, "rb"))
    num_skills = len(logs["states"])
    for skill in range(num_skills):
        states_list = logs["states"][skill]
        for demo in range(len(states_list)):
            frames = logs[skill_type][skill][demo]
            x1 = frames[0]
            x2 = frames[1]
            traj = logs["states"][skill][demo]
            plt.plot([100*state[0] for state in traj[x1:x2+1]], [100*state[1] for state in traj[x1:x2+1]], color="olive", alpha=0.75)
        plt.title("Demos for Skill "+str(skill)+" "+skill_type)
        # plot_parking_spots()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        plt.close()


args_dict = {"min_skill_length":MIN_SKILL_LENGTH,
    "env_name":"parking",
    "rollouts_dir":ROLLOUTS_DIR,
    "num_episodes":3,
    "same_traj":False,
    "sample":True,
    "viz_type":"pos_agnostic", # pos_based
    "chosen_skills":[1,8,15], #Make sure latents are always the same :)
    "idx_for_fixed_skills":{1:[0], 8:[0], 15:[0]},
    # TODO: change saved model directory
    "compile_dir":"results/CompILE_oriXY_snorm_CIL_statediff_iteration=200_latent_dim=12_maxSegNum=4_expectedSegLength=30_beta_b=0.1_beta_z=0.1_beta_s=1.0_23-08-03-22-10-14",
    "min_skill_length":MIN_SKILL_LENGTH}
# logs_fn = plot_rollouts(args_dict)
logs_fn = plot_skills(args_dict)
skill_utils.get_eval_seeds(args_dict)
# viz_logs(args_dict, logs_fn, skill_type="compile")
# viz_logs(args_dict, logs_fn, skill_type="time")



