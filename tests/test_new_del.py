from carla_env import CarlaEnv

import numpy as np


def main():
    n = 100
    for it in range(n):
        print("iter {}".format(it))

        env = CarlaEnv({})
        env.reset()
        del env


if __name__ == "__main__":
    main()