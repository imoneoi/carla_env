# import torch
# from tap import Tap
# import time
# import os
# import shutil


# class Arguments(Tap):
#     iterations: int = 2000 # Number of training iterations
#     learning_rate: float = 1e-3 # Learning rate
#     hidden_dim: int = 64 # Number of hidden units
#     latent_dim: int = 32 # Dimensionality of latent variables.
#     latent_dist: str = 'gaussian' # Latent variable type -> "gaussian" or "concrete"
#     batch_size: int = 100 # Mini-batch size (for averaging gradients)
#     train_eval_ratio: float = 0.8 # training set out of the whole dataset
#     state_dim: int = 12 # Dimensionality of state
#     num_actions: int = 5 # Number of distinct actions in data generation
#     cont_action_dim: int = 2 # Dimensionality of the continuous action space (or 0 if discrete)
#     num_segments: int = 4 # Number of segments in data generation
#     prior_rate: int = 50  # Expected length of segments
#     log_interval: int = 5 # Logging interval
#     save_interval: int = 5 # Saving interval
#     rollouts_path_train: str = None # Path to training rollouts
#     rollouts_path_eval: str = None # Path to evaluation rollouts
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     run_name: str = None # Name of run
#     save_dir: str = None # Path to save (computed)
#     beta_b: float = 0.1 # Weight on KL term for boundaries
#     beta_z: float = 0.1 # Weight on KL term for latents
#     beta_s: float = 0 # Weight on state reconstruction
#     mode: str = "action" # What to embed/reconstruct -> action or state+action or statediff+action
#     action_type: str = "continuous" # Action type -> discrete or continuous
#     wb: bool = False # Record to wandb
#     random_seed: int = 0 #Random seed

# parser = Arguments()
# args = parser.parse_args()
import argparse
import torch
import time
import datetime
import json
import os
import shutil

# Use argparse.ArgumentParser to define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=2000, help="Number of training iterations")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--hidden_dim", type=int, default=64, help="Number of hidden units")
parser.add_argument("--latent_dim", type=int, default=32, help="Dimensionality of latent variables.")
parser.add_argument("--latent_dist", type=str, default="gaussian", help="Latent variable type -> 'gaussian' or 'concrete'")
parser.add_argument("--batch_size", type=int, default=100, help="Mini-batch size (for averaging gradients)")
parser.add_argument("--train_eval_ratio", type=float, default=0.8, help="Training set out of the whole dataset")
parser.add_argument("--state_dim", type=int, default=12, help="Dimensionality of state")
parser.add_argument("--num_actions", type=int, default=5, help="Number of distinct actions in data generation")
parser.add_argument("--cont_action_dim", type=int, default=2, help="Dimensionality of the continuous action space (or 0 if discrete)")
parser.add_argument("--num_segments", type=int, default=4, help="Number of segments in data generation")
parser.add_argument("--prior_rate", type=int, default=50, help="Expected length of segments")
parser.add_argument("--log_interval", type=int, default=5, help="Logging interval")
parser.add_argument("--save_interval", type=int, default=5, help="Saving interval")
parser.add_argument("--rollouts_path_train", type=str, default=None, help="Path to training rollouts")
parser.add_argument("--rollouts_path_eval", type=str, default=None, help="Path to evaluation rollouts")
parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device (cuda or cpu)")
parser.add_argument("--run_name", type=str, default=None, help="Name of run")
parser.add_argument("--save_dir", type=str, default=None, help="Path to save (computed)")
parser.add_argument("--beta_b", type=float, default=0.1, help="Weight on KL term for boundaries")
parser.add_argument("--beta_z", type=float, default=0.1, help="Weight on KL term for latents")
parser.add_argument("--beta_s", type=float, default=0, help="Weight on state reconstruction")
parser.add_argument("--mode", type=str, default="action", help="What to embed/reconstruct -> action or state+action or statediff+action")
parser.add_argument("--action_type", type=str, default="continuous", help="Action type -> discrete or continuous")
parser.add_argument('--wandb', type=bool, default=True)
parser.add_argument("--wb", action="store_true", help="Record to wandb")
parser.add_argument("--random_seed", type=int, default=0, help="Random seed")

args = parser.parse_args()

# Set default values if necessary
if args.run_name is None:
    args.run_name = "test_" + time.strftime("%H%M%S-%Y%m%d")

args.action_type = "continuous" if args.cont_action_dim > 0 else "discrete"

# Create a directory for saving results
nowTime = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
dir = "results/" + "CompILE_{}_iteration={}_latent_dim={}_maxSegNum={}_beta_b={}_beta_z={}_beta_s={}_{}".format(args.run_name, args.iterations, args.latent_dim, args.num_segments, args.beta_b, args.beta_z, args.beta_s, nowTime)
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)
args.save_dir = dir

# Save the arguments to a JSON file
with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
    json.dump(vars(args), f)

device = args.device