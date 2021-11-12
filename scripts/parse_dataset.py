from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import os

def parse():
    parser = argparse.ArgumentParser()
    # TODO: path for dataset recording
    parser.add_argument("dataset_path", type=str, default="real_1", help="Path to save dataset")
    args = parser.parse_args()

    dataset_path = os.path.join("../dataset", args.dataset_path)
    actions = []
    rewards = []
    with os.scandir(dataset_path) as processes:
        for process in processes:
            process_path = os.path.join(dataset_path, process.name)
            with os.scandir(process_path) as trajectories:
                for trajectory in trajectories:
                    if trajectory.name.split('.')[-1] == 'json':
                        trajectory_path = os.path.join(process_path, trajectory.name)
                        with open(trajectory_path, 'r') as f:
                            data = eval(f.read())
                            actions.append(data["action"])
                            rewards.append(data["reward"])
                    else:
                        continue

    print(len(actions), len(rewards))
    accel_max = max(np.array(actions)[:,0])
    accel_min = min(np.array(actions)[:,0])
    steer_rightmax = max(np.array(actions)[:,1])
    steer_leftmax = min(np.array(actions)[:,1])
    rew_max = max(rewards)
    rew_min = min(rewards)
    print(accel_max)
    print(accel_min)
    print(steer_rightmax)
    print(steer_leftmax)
    print(rew_max)
    print(rew_min)

    # # TODO: plot actions distribution in the dataset
    # a = pd.DataFrame(actions, columns=['Accelerate', 'Steer'])
    # sns.set_palette(sns.color_palette())
    # # kdeplot
    # # kdeplot = sns.kdeplot(x=a['Accelerate'], y=a['Steer'], kind="kde", cbar=True, cmap='RdYlBu_r', shade=True)

    # # # kinds: hist | scatter | kde | hex | reg
    # # ax = sns.jointplot(x=a['Accelerate'], y=a['Steer'], kind="hist", cbar=True)
    # kdeplot = sns.jointplot(x=a['Accelerate'], y=a['Steer'], kind="kde", cbar=True, cmap='Blues', shade=True, fill=True, space=0, ratio=15)
    # plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
    # # get the current positions of the joint ax and the ax for the marginal x
    # pos_joint_ax = kdeplot.ax_joint.get_position()
    # pos_marg_x_ax = kdeplot.ax_marg_x.get_position()
    # # reposition the joint ax so it has the same width as the marginal x ax
    # kdeplot.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
    # # reposition the colorbar using new x positions and y positions of the joint ax
    # kdeplot.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])
    
    # kdeplot.set_axis_labels(xlabel=r'Acceleration($m/s^{-2}$)', ylabel='Steering angle(rad)', fontsize=20)
    # # plt.rcParams["axes.labelsize"] = 20
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.title('Joint Distribution of Acceleration and Steering Angle in the dataset', loc='right', y=1.0, pad=40, fontsize=20, fontweight='heavy')

    # TODO: plot rewards distribution
    distplot = sns.distplot(rewards)
    distplot.set_xlabel('Reward', fontsize=20)
    distplot.set_ylabel('Density', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Distribution of Reward in the dataset', y=1.0, fontsize=20, fontweight='heavy')


    plt.show()

    # # Heatmap
    # fig = plt.figure(facecolor='w')
    # ax = fig.add_subplot(1,1,1)
    # ax = sns.heatmap(actions)
    # plt.show()





if __name__ == "__main__":
    parse()