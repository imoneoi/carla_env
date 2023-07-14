import os
import glob
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import datetime
import ipdb

def plot_trajectories(data_path):
    all_x_values = []
    all_y_values = []
    trajectory_indices = []

    # Get the list of JSON files in the directory
    json_files = glob.glob("{}/*.json".format(data_path))

    # Iterate over the JSON files
    for filename in sorted(json_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])):
        # Read the JSON file
        with open(filename, "r") as f:
            data = json.load(f)

        # Extract the x and y values from the JSON data
        x = data["pos"]["x"]
        y = data["pos"]["y"]

        # Check if the trajectory is terminal
        if data.get("traj_terminal"):
            # Store the trajectory indices
            trajectory_indices.append(len(all_x_values))

        # Append the x and y values to the respective lists
        all_x_values.append(x)
        all_y_values.append(y)
    print(trajectory_indices)

    # Plot all the data points within each trajectory with a single color
    colormap = plt.cm.get_cmap('rainbow')
    colors = colormap(np.linspace(0, 1, len(trajectory_indices)))
    for i, index in enumerate(trajectory_indices):
        start_index = 0 if i == 0 else trajectory_indices[i-1] + 1
        end_index = index
        # print(start_index, end_index)
        # ipdb.set_trace()
        plt.scatter(all_x_values[start_index:end_index+1], all_y_values[start_index:end_index+1], s=1, color=colors[i])

    # Set the axis labels and title
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectories")

    # Save the figure with a timestamp in the filename
    plt.savefig("../figures/Town10_100k_pos_{}.png".format(datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')))
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="bev_23070700", type=str, help="Path to save dataset")
    parser.add_argument("--n_jobs", type=int, default=10, help="Number of worker processes")
    args = parser.parse_args()

    n_jobs = args.n_jobs
    for job_id in range(n_jobs):
        data_path = os.path.join("../dataset/", args.data_path, str(job_id))

    plot_trajectories(data_path)

if __name__ == "__main__":
    main()
