
import os
import glob
import argparse
import json
import matplotlib.pyplot as plt

def plot_trajectories(data_path):
    x_values = []
    y_values = []

    # Get the list of JSON files in the directory
    json_files = glob.glob("{}/*.json".format(data_path))

    # Iterate over the JSON files
    for filename in json_files:
        # Read the JSON file
        with open(filename, "r") as f:
            data = json.load(f)

        # Extract the x and y values from the JSON data
        x = data["pos"]["x"]
        y = data["pos"]["y"]

        # Append the x and y values to the respective lists
        x_values.append(x)
        y_values.append(y)

    # Plot the curve using matplotlib
    # plt.plot(x_values, y_values)
    plt.scatter(x_values, y_values, s=1)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Position Curve")
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
