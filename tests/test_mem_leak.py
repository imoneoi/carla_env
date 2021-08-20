from wrapped_carla_env import create_wrapped_carla_single_car_env


def main():
    env = create_wrapped_carla_single_car_env(global_config={}, gpu_index=0)
    while True:
        env.reset()


if __name__ == "__main__":
    main()
