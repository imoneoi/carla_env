""" Adapted from https://github.com/tkipf/compile
Example calls: 
python train_compile.py  --iterations=5 --rollouts_path_train=../dataset/bev_071317_T10/0 --rollouts_path_eval=../dataset/bev_071317_T10/0 --latent_dist concrete --latent_dim 4 --num_segments 4 --cont_action_dim 2 --prior_rate 10 --mode state+action --run_name driving --state_dim 4 --beta_s 1
python train_compile.py --rollouts_path_train expert-rollouts/drawing_train_1686529312.pkl --rollouts_path_eval expert-rollouts/drawing_eval_1686529312.pkl  --latent_dist concrete --latent_dim 16 --num_segments 8 --iterations 1 --mode action --run_name drawing-concrete-16d-action0 --batch_size 50 --learning_rate 0.01 --beta_s 0.0
"""

import os
import numpy as np
import random
from tqdm import trange
import wandb
import datetime
import torch
from torch.utils.data import Dataset, DataLoader

import compile_utils
import skill_extraction
# TODO: dataset and args
from data_parser import TrajectoryDatasetwithPosition, pad_collate
from arguments import args, device

wandb.init(project="AI_assisted_driving", save_code=False)
wandb.config.update(args)
nowTime = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
wandb.run.name = f"CompILE_{args.run_name}_iteration={args.iterations}_latent_dim={args.latent_dim}_maxSegNum={args.num_segments}_beta_b={args.beta_b}_beta_z={args.beta_z}_beta_s={args.beta_s}_{nowTime}"

#set random seeds
torch.manual_seed(args.random_seed)
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)  

model = skill_extraction.CompILE(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
dl_train = DataLoader(TrajectoryDatasetwithPosition(args.rollouts_path_train, args), collate_fn=pad_collate, batch_size=args.batch_size)
dl_eval = DataLoader(TrajectoryDatasetwithPosition(args.rollouts_path_eval, args, train=False), collate_fn=pad_collate, batch_size=args.batch_size)


# Train model.
print('Training model...')
print("Device: " + str(device))
best_valid_loss_same = 0
curr_valid_loss = float('inf')
for step in trange(args.iterations):
    train_loss = 0
    batch_num = 0
    dl_iter_train = iter(dl_train)
    for batch in dl_iter_train:
        states, actions, rewards, lengths, seeds = batch

        # Run forward pass.
        model.train()
        outputs = model.forward(states, actions, lengths)
        loss, nll, kl_z, kl_b = compile_utils.get_losses(states, actions, outputs, args)

        train_loss += nll.item() # This is just the NLL loss (without regularizers) - #TODO: Log all the terms
        batch_num += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        wandb.log(
            {"loss_per_step": loss,
            "NLL loss": nll,
            "KL-z": kl_z,
            "KL-b": kl_b}
        )
    
    train_loss /= batch_num
    wandb.log({"avg_train_loss": train_loss})

    if step % args.log_interval == 0:
        # Run evaluation.
        model.eval()
        
        dl_iter_eval = iter(dl_eval)
        total_nll = 0.
        total_acc = 0.
        total_count = 0
        for batch in dl_iter_eval:
            states, actions, rewards, lengths, seeds = batch
            count = len(states)
            outputs = model.forward(states, actions, lengths)
            _, nll, _, _ = compile_utils.get_losses(states, actions, outputs, args)
            acc, rec = compile_utils.get_reconstruction_accuracy(states, actions, outputs, args)
            total_nll += nll.item() * count
            total_acc += acc.item() * count
            total_count += count

        # Accumulate metrics.
        eval_acc = total_acc / count
        eval_nll = total_nll / count
        if(eval_nll <= curr_valid_loss):
            curr_valid_loss =  eval_nll
            best_valid_loss_same = 0
        else:
            best_valid_loss_same += 1
            
        wandb.log(
            {"eval_acc": eval_acc,
            "eval_nll": eval_nll}
        )
    if step % args.save_interval == 0:
        torch.save(model.state_dict(), os.path.join(args.save_dir, "model.pth"))

    if(best_valid_loss_same >  50):
        break
