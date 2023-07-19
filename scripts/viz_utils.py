from collections import defaultdict, Counter
import torch
import matplotlib.pyplot as plt
import numpy as np
import skill_extraction
from torch.utils.data import dataset, DataLoader
from data_parser import TrajectoryDatasetwithPosition, pad_collate
import json


class DotDict(dict):
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            self[key] = val

def load_model_and_batch(viz_args_dict, batch_num=1000):
    args = DotDict()
    if(viz_args_dict["compile_dir"]):
        args_dict = json.load(open(viz_args_dict["compile_dir"] + "/config.json"))
    else:
        args_dict = {}
    for key, value in args_dict.items():
        args.__setattr__(key, value)
    args.device = 'cpu'
    if viz_args_dict["env_name"] == "lunar":
        args.cont_action_dim = 0
        args.action_type = 'discrete'
    elif viz_args_dict["env_name"] == "parking":
        args.action_type = 'continuous'
    if viz_args_dict["compile_dir"]:
        model = skill_extraction.CompILE(args).to(args.device)
        # chooose your model
        model.load_state_dict(torch.load(args.save_dir + "/model_0.pth", map_location=torch.device('cpu')))
    else:
        model = None
    args.rollouts_path = viz_args_dict["rollouts_dir"]
    dl = DataLoader(TrajectoryDatasetwithPosition(args.rollouts_path, args, train=False), collate_fn=pad_collate, batch_size=batch_num)
    batch = next(iter(dl))
    return model, batch, args

def get_boundaries(ep_idx, seg_idx, all_b, lengths):
    if seg_idx == 0:
        start = 0
    else:
        temp_starts = []
        for temp_seg_idx in range(0,seg_idx):
            temp_starts.append(torch.where(all_b['samples'][temp_seg_idx][ep_idx] == 1)[0].item())
        start = max(temp_starts)
    if seg_idx + 1 == len(all_b['samples']):
        end = lengths[ep_idx].item()
    else:
        end = torch.where(all_b['samples'][seg_idx][ep_idx] == 1)[0].item()
    if(start > end):
        return 0,0
    return start, end

def get_latent_info(outputs, lengths, args):
    all_encs, all_recs, all_masks, all_b, all_z = outputs
    z = torch.stack(all_z['samples'])
    z_idx = z.argmax(dim=-1)
    boundaries_by_latent = defaultdict(list)
    segments_by_latent = defaultdict(list)
    latents_by_segment = defaultdict(list)
    
    for z_j in range(args.latent_dim):
        for seg_idx, ep_idx in zip(*torch.where(z_idx == z_j)):
            seg_idx = seg_idx.item()
            ep_idx = ep_idx.item()
            latents_by_segment[seg_idx].append(z_j)
            boundaries = (ep_idx, get_boundaries(ep_idx, seg_idx, all_b, lengths))
            boundaries_by_latent[z_j].append(boundaries)
            segments_by_latent[z_j].append(seg_idx)

    boundaries_by_episode = []
    for ep_idx in range(len(all_b['samples'][0])):
        boundaries_by_episode.append([])
        for seg_idx in range(len(all_b['samples'])):
            boundaries_by_episode[ep_idx].append(get_boundaries(ep_idx, seg_idx, all_b, lengths))

    return z, z_idx, boundaries_by_latent, segments_by_latent, latents_by_segment, boundaries_by_episode


def get_latent_and_segment_std(segments_by_latent):
    latent_std = []
    for latent_key in segments_by_latent.keys():
        latent_std.append((latent_key, np.std(segments_by_latent[latent_key])))
    return latent_std








