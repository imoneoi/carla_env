import pickle
import h5py
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
from arguments import device

def pad_collate(batch):
  (obss, actss, rewss, seeds) = zip(*batch)
  lens = torch.Tensor([len(obs) for obs in obss]).type(torch.int64).to(device)

  obss_pad = pad_sequence(obss, batch_first=True, padding_value=0)
  actss_pad = pad_sequence(actss, batch_first=True, padding_value=0)
  rewss_pad = pad_sequence(rewss, batch_first=True, padding_value=0)

  return obss_pad, actss_pad, rewss_pad, lens, seeds

import os
import json
import torch
from torch.utils.data import Dataset
import random


class CILTrajectoryDatasetwithVectorState(Dataset):
    def __init__(self, traj_dir, args, train=True):
        self.args = args
        self.traj_dir = traj_dir
        self.trajectories = []

        # Load JSON files as data samples
        h5_files = [filename for filename in os.listdir(traj_dir)]
        
        for filename in h5_files:
            file_path = os.path.join(traj_dir, filename)
            f = h5py.File(file_path, 'r')
            measurement_data = list(f[u'targets'])
            ct = 0
            trajectory = []
            
            position = (float(measurement_data[0][8]),float(measurement_data[0][9]))
            last_position = position
            for data_point in measurement_data:        
                if ct >= 5 and ct < len(measurement_data) - 1:
                    last_position = position
                    position = (data_point[8],data_point[9])
                    dist = np.linalg.norm(np.array(position)-np.array(last_position))
                    if dist > 147 and len(trajectory) > 1:
                        self.trajectories.append(trajectory)
                        trajectory = []
                    else:
                        trajectory.append(data_point)     
                ct += 1
                
            last_position = (float(measurement_data[-3][8]),float(measurement_data[-3][9]))
            position = (float(measurement_data[-2][8]),float(measurement_data[-2][9]))
            dist = np.linalg.norm(np.array(position)-np.array(last_position))
            if dist<107 and len(trajectory)>5:
                self.trajectories.append(trajectory)
                trajectory = []
                

        # Randomly select a subset of trajectories
        if train:
            num_selected_trajectories = int(len(self.trajectories) * self.args.train_eval_ratio)
        else:
            num_selected_trajectories = int(len(self.trajectories) * (1 - self.args.train_eval_ratio))
        selected_indices = random.sample(range(len(self.trajectories)), num_selected_trajectories)
        self.trajectories = [self.trajectories[idx] for idx in selected_indices]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]

        # Process the trajectory data here
        obs_list, actions_list = [], []
        for data_point in trajectory:
            # obs_list.append(torch.Tensor(data_point[8: 11]).to(self.args.device))
            obs_list.append(torch.Tensor([data_point[8], data_point[9], data_point[10], data_point[21], data_point[22]]).to(self.args.device))
            # obs_list.append(torch.Tensor([data_point[10], data_point[21], data_point[22], data_point[23]]).to(self.args.device))
            # pos_list.append(torch.Tensor([data_point[10], data_point[21], data_point[22], data_point[23]]).to(self.args.device))
            actions_list.append(torch.Tensor([data_point[0], data_point[1] - data_point[2]]).to(self.args.device))

        obs = torch.stack(obs_list, dim=0)
        actions = torch.stack(actions_list, dim=0)
        rews = torch.zeros((len(actions_list), 1)).to(self.args.device)  # Set all rewards to 0
        seed = torch.Tensor([self.args.random_seed]).to(self.args.device)
        # obs = obs[:-1]  # End goal

        if self.args.action_type == "discrete":
            out = [obs, (actions + 1).type(torch.int64), rews, seed]  # So that we can use padding 0
        else:
            out = [obs, actions, rews, seed]

        return out


class TrajectoryDatasetwithPosition(Dataset):
    def __init__(self, traj_dir, args, train=True):
        self.args = args
        self.traj_dir = traj_dir
        self.trajectories = []

        # Load JSON files as data samples
        json_files = [filename for filename in os.listdir(traj_dir) if filename.endswith('.json')]

        # Group data points into trajectories based on "traj_terminal" key
        trajectory = []
        for filename in sorted(json_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])):
            with open(os.path.join(traj_dir, filename), "r") as f:
                data_point = json.load(f)
                trajectory.append(data_point)

                if data_point.get("traj_terminal", False):
                    self.trajectories.append(trajectory)
                    trajectory = []

        # Randomly select a subset of trajectories
        if train:
            num_selected_trajectories = int(len(self.trajectories) * self.args.train_eval_ratio)
        else:
            num_selected_trajectories = int(len(self.trajectories) * (1 - self.args.train_eval_ratio))
        selected_indices = random.sample(range(len(self.trajectories)), num_selected_trajectories)
        self.trajectories = [self.trajectories[idx] for idx in selected_indices]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]

        # Process the trajectory data here
        obs_list, actions_list = [], []
        for data_point in trajectory:
            obs_list.append(torch.Tensor(list(data_point["pos"].values()) + list(data_point["vel"].values())).to(self.args.device))
            actions_list.append(torch.Tensor(data_point["act"]).to(self.args.device))

        obs = torch.stack(obs_list, dim=0)
        actions = torch.stack(actions_list, dim=0)
        rews = torch.zeros((len(actions_list), 1)).to(self.args.device)  # Set all rewards to 0
        seed = torch.Tensor([self.args.random_seed]).to(self.args.device)
        # obs = obs[:-1]  # End goal

        if self.args.action_type == "discrete":
            out = [obs, (actions + 1).type(torch.int64), rews, seed]  # So that we can use padding 0
        else:
            out = [obs, actions, rews, seed]

        return out



class TrajectoryDatasetWithRew(Dataset):
    def __init__(self, traj_dir, args):
        self.args = args
        self.traj_dir = traj_dir
        with open(self.traj_dir, "rb") as f:
            self.trajectories = pickle.load(f)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        obs = torch.Tensor(self.trajectories[idx].obs).to(self.args.device)
        actions = torch.Tensor(self.trajectories[idx].acts).to(self.args.device)
        rews = torch.Tensor(self.trajectories[idx].rews).to(self.args.device)
        seed = torch.Tensor([self.trajectories[idx].infos[0]["seed"]]).to(self.args.device)
        obs = obs[:-1] #end goal 
        if self.args.action_type == "discrete":
            out = [obs, (actions+1).type(torch.int64), rews, seed] #so that we can use padding 0
        else:
            out = [obs, actions, rews, seed]
        return out

class TrajectoryDatasetActionSeedOnly(Dataset):
    #This should only be run for the server
    def __init__(self, action_list, seed_list):
        self.trajectories = torch.Tensor(action_list).to("cpu")
        self.seeds = seed_list

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        num_actions = len(self.trajectories[idx])
        obs = torch.zeros(num_actions+1, 12).to("cpu")
        actions = torch.Tensor(self.trajectories[idx]).to("cpu")
        rews = torch.Tensor(np.zeros(num_actions)).to("cpu")
        seed = torch.Tensor([self.seeds[idx]]).to("cpu")
        obs = obs[:-1] #end goal 
        out = [obs, actions, rews, seed]
        return out

class OneTrajectoryDatasetWithRew(Dataset):
    def __init__(self, one_traj, args):
        self.args = args
        self.one_traj = one_traj

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        obs = torch.Tensor(self.one_traj.obs).to(self.args.device)
        actions = torch.Tensor(self.one_traj.acts).to(self.args.device)
        rews = torch.Tensor(self.one_traj.rews).to(self.args.device)
        seed = torch.Tensor([self.one_traj.infos[0]["seed"]]).to(self.args.device)
        obs = obs[:-1] #end goal 
        if self.args.action_type == "discrete":
            out = [obs, (actions+1).type(torch.int64), rews, seed] #so that we can use padding 0
        else:
            out = [obs, actions, rews, seed]
        return out
