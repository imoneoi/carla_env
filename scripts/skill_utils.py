import viz_utils
import torch
from collections import defaultdict
import numpy as np
NUM_TYPE_SKILLS = 3
MAX_T = {'parking':25}

def entropy_actions(action_list):
    change_action = 0
    curr_action = action_list[0]
    for a in action_list:
        if(curr_action != a):
            change_action += 1
            curr_action = a
    return change_action / len(action_list)

def entropy_rewards(reward_list):
    return reward_list[-1]

def score_episode(args_dict, action_list, reward_list):
    if(args_dict["env_name"] == "lunar"):
        return entropy_actions(reward_list)
    elif(args_dict["env_name"] == "parking"):
        return entropy_rewards(reward_list)
    else:
        return None

def close_state(target_x, target_y, curr_x, curr_y):
    target_x = int(target_x*100)
    target_y = int(target_y*100)
    curr_x = int(curr_x*100)
    curr_y = int(curr_y*100)
    return (target_x == curr_x) and (target_y == curr_y)

def get_goal_xy_from_state(target_state):
    return target_state[6], target_state[7]

def get_goal_hxy_from_state(target_state):
    return target_state[10], target_state[11]

def get_state_hxy(target_state):
    return target_state[4], target_state[5]  

def get_state_xy(target_state):
    return target_state[0], target_state[1] 
    
def filter_state_conditions(target_state): 
    filter_parking_spots = [(0.26, 0.14), (0.22, 0.14), (0.18, 0.14)] # filter to select spots
    true_spot = False
    for parking_spot in filter_parking_spots:
        gx, gy = get_goal_xy_from_state(target_state)
        if(close_state(gx, gy, parking_spot[0], parking_spot[1])):
             true_spot = True 
    target_hx, target_hy = get_state_hxy(target_state)
    goal_hx, goal_hy = get_goal_hxy_from_state(target_state)
    return true_spot #and target_hx < 0  #target_hy > 0.2

def apply_parking_filter(states, actions, rews, seeds, lengths):
    new_states = []
    new_actions = []
    new_rews = []
    new_seeds = []
    new_lengths = []
    for episode in range(len(states)):
        all_states = states[episode][:lengths[episode]]
        if(not filter_state_conditions(all_states[0])):
                continue
        new_states.append(states[episode].numpy())
        new_actions.append(actions[episode].numpy())
        new_rews.append(rews[episode].numpy())
        new_seeds.append(seeds[episode].numpy())
        new_lengths.append(lengths[episode].item())
    return torch.FloatTensor(new_states), torch.FloatTensor(new_actions), torch.FloatTensor(new_rews), torch.FloatTensor(new_seeds),torch.LongTensor(new_lengths)


def get_target_rollouts(args_dict):
    model, batch, args = viz_utils.load_model_and_batch(args_dict)
    logs = defaultdict(lambda: defaultdict(list))
    states, actions, rews, lengths, seeds = batch 
    if(args_dict["env_name"] == "parking"):
        print("Applying Parking Filter")
        states, actions, rews, seeds, lengths = apply_parking_filter(states, actions, rews, seeds, lengths)
    for skill in range(NUM_TYPE_SKILLS):
        for ei in range(args_dict["num_episodes"]):
            if(args_dict["same_traj"] and not args_dict["filter_episodes"]):
                episode = 0
            elif(args_dict["filter_episodes"]):
                episode = args_dict["filter_episodes"][ei % len(args_dict["filter_episodes"])]
            else:
                episode = ei
            all_states = states[episode][:lengths[episode]]
            all_actions= actions[episode][:lengths[episode]]
            all_rews = rews[episode][:lengths[episode]]
            seed = int(seeds[episode].item())
            if(args_dict["env_name"] == "lunar"):
                logs["states"][skill].append(all_states.tolist())
                logs["rews"][skill].append(all_rews.tolist())
                logs["actions"][skill].append([a-1 for a in all_actions])
            elif(args_dict["env_name"] == "parking"):
                logs["states"][skill].append(all_states)
                logs["rews"][skill].append(all_rews)
                logs["actions"][skill].append(all_actions)
            logs["seed"][skill].append(seed)
            if(args_dict["compile_dir"]):
                logs["compile"][skill].append(get_target_compile_skills(args_dict, episode)[skill])
                logs["time"][skill].append(get_target_time_skills(args_dict, episode, NUM_TYPE_SKILLS)[skill])
            else:
                logs["compile"][skill].append(None)
                logs["time"][skill].append(None)
    return dict(logs)

def get_sample_rollouts(args_dict):
    latent_skills_dict, states, lengths = get_all_skills(args_dict)
    print(lengths[:5])
    logs = defaultdict(lambda: defaultdict(list))
    # filter by specific skill
    if(args_dict["chosen_skills"]):
        for skill_idx in range(NUM_TYPE_SKILLS):
            chosen_skill = args_dict["chosen_skills"][skill_idx]
            for idx in args_dict["idx_for_fixed_skills"][chosen_skill]:
                logs["states"][skill_idx].append(latent_skills_dict["states"][chosen_skill][idx])
                logs["actions"][skill_idx].append(latent_skills_dict["actions"][chosen_skill][idx])
                logs["rews"][skill_idx].append(latent_skills_dict["rews"][chosen_skill][idx])
                logs["seed"][skill_idx].append(latent_skills_dict["seed"][chosen_skill][idx])
                logs["compile"][skill_idx].append(latent_skills_dict["compile"][chosen_skill][idx])
                episode = latent_skills_dict["episode"][chosen_skill][idx]
                time_frames = get_target_time_skills(args_dict, episode, NUM_TYPE_SKILLS)
                logs["time"][skill_idx].append(time_frames[skill_idx])
    # no skill filter
    else:
        for skill_idx in range(NUM_TYPE_SKILLS):
            for idx in range(args_dict["num_episodes"]):
                logs["states"][skill_idx].append(latent_skills_dict["states"][skill_idx][idx])
                logs["actions"][skill_idx].append(latent_skills_dict["actions"][skill_idx][idx])
                logs["rews"][skill_idx].append(latent_skills_dict["rews"][skill_idx][idx])
                logs["seed"][skill_idx].append(latent_skills_dict["seed"][skill_idx][idx])
                logs["compile"][skill_idx].append(latent_skills_dict["compile"][skill_idx][idx])
                episode = latent_skills_dict["episode"][skill_idx][idx]
                time_frames = get_target_time_skills(args_dict, episode, NUM_TYPE_SKILLS)
                logs["time"][skill_idx].append(time_frames[skill_idx])
    return dict(logs)


#Get Compile Skills for a Target Episode, helper function for full rollouts
def get_target_compile_skills(args_dict, episode):
    print("Getting Compile Skills for Rollout")
    model, batch, compile_args = viz_utils.load_model_and_batch(args_dict)
    states, actions, rews, seeds, lengths  = batch
    if(args_dict["env_name"] == "parking"):
        print("Applying Parking Filter")
        states, actions, rews, seeds, lengths = apply_parking_filter(states, actions, rews, seeds, lengths) 
    model.training = False
    outputs = model.forward(states, actions, lengths)
    z, z_idx, boundaries_by_latent, segments_by_latent, latents_by_segment, boundaries_by_episode = viz_utils.get_latent_info(outputs=outputs, lengths=lengths, args=compile_args)
    target_boundaries = [b for b in boundaries_by_episode[episode] if b[1]-b[0] >= args_dict["min_skill_length"]]
    print(target_boundaries[::-1])
    return target_boundaries[::-1]

#Get Temporal Skills for a Target Episode, helper function for full rollouts
def get_target_time_skills(args_dict, episode, num_split):
    print("Getting Time Skills for Rollout")
    model, batch, compile_args = viz_utils.load_model_and_batch(args_dict)
    states, actions, rews, lengths, seeds  = batch
    if(args_dict["env_name"] == "parking"):
        print("Applying Parking Filter")
        states, actions, rews, seeds, lengths = apply_parking_filter(states, actions, rews, seeds, lengths) 
    episode_len = lengths[episode].item()
    target_boundaries = get_time_split_frames(episode_len, num_split)
    return target_boundaries[::-1]


def score_latent_skills(latent_skills_dict):
    print("Scoring Skills!")
    for latent in latent_skills_dict['states'].keys():
        print((latent, len(latent_skills_dict['states'][latent])))
        


def get_all_skills(args_dict):
    model, batch, args = viz_utils.load_model_and_batch(args_dict)
    states, actions, rews, lengths, seeds  = batch
    if(args_dict["env_name"] == "parking"):
        print("Applying Parking Filter")
        states, actions, rews, seeds, lengths = apply_parking_filter(states, actions, rews, seeds, lengths) 
    model.training = False
    outputs = model.forward(states, actions, lengths)
    z, z_idx, boundaries_by_latent, segments_by_latent, latents_by_segment, boundaries_by_episode = viz_utils.get_latent_info(outputs, lengths, args)
    latent_skills_dict = defaultdict(lambda: defaultdict(list))
    for latent in viz_utils.get_latent_and_segment_std(segments_by_latent):
        boundaries = boundaries_by_latent[latent[0]]
        boundary_diff_bool = [b[1][1]-b[1][0] < args_dict["min_skill_length"] for b in boundaries]
        if(all(boundary_diff_bool)):
            continue
        for b in boundaries:
            episode = b[0]
            if(b[1][1]-b[1][0] < args_dict["min_skill_length"]+3):
                continue
            x1 = b[1][0] 
            x2 = b[1][1]
            all_states = states[episode][:lengths[episode]]
            all_actions = actions[episode][:lengths[episode]]
            all_rews = rews[episode][:lengths[episode]]
            seed = int(seeds[episode].item())
            if(args_dict["env_name"] == "lunar"):
                latent_skills_dict["states"][latent[0]].append(all_states.tolist())
                latent_skills_dict["actions"][latent[0]].append([a-1 for a in all_actions])
                latent_skills_dict["rews"][latent[0]].append(all_rews.tolist())
            elif(args_dict["env_name"] == "parking"):
                latent_skills_dict["states"][latent[0]].append(all_states)
                latent_skills_dict["actions"][latent[0]].append(all_actions)
                latent_skills_dict["rews"][latent[0]].append(all_rews)
            latent_skills_dict["seed"][latent[0]].append(seed)
            latent_skills_dict["compile"][latent[0]].append([x1, x2])
            latent_skills_dict["episode"][latent[0]].append(episode)
    score_latent_skills(latent_skills_dict)
    return latent_skills_dict, states, lengths

def get_eval_seeds(args_dict):
    model, batch, args = viz_utils.load_model_and_batch(args_dict)
    states, actions, rews, lengths, seeds  = batch
    if(args_dict["env_name"] == "parking"):
        print("Applying Parking Filter")
        states, actions, rews, seeds, lengths = apply_parking_filter(states, actions, rews, seeds, lengths) 
    print(seeds[:20])

def get_time_split_frames(max_t, num_split):
    split_size = int(max_t/num_split)       
    frames = []
    all_frames = []
    for x in range(split_size, max_t+1, split_size):
        frames.append([x-split_size, x])
    # go to end of trajectory
    frames[-1][-1] = max_t
    return frames


