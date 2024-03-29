from carla_env import CarlaEnv

import carla
import numpy as np


def main():
    # options
    dt = 0.1
    test_num_cars = 10
    test_times = 10000
    test_ticks = 10000
    test_reward_thresh = -100

    # test all cars
    car_name_list = ['vehicle.audi.a2', 'vehicle.audi.tt', 'vehicle.bmw.grandtourer', 'vehicle.carlamotors.carlacola', 'vehicle.dodge_charger.police', 'vehicle.tesla.cybertruck', 'vehicle.mercedesccc.mercedesccc', 'vehicle.chargercop2020.chargercop2020', 'vehicle.chevrolet.impala', 'vehicle.mustang.mustang', 'vehicle.volkswagen.t2', 'vehicle.bmw.isetta', 'vehicle.citroen.c3', 'vehicle.charger2020.charger2020', 'vehicle.audi.etron', 'vehicle.nissan.micra', 'vehicle.lincoln.mkz2017', 'vehicle.tesla.model3', 'vehicle.lincoln2020.mkz2020', 'vehicle.seat.leon', 'vehicle.toyota.prius', 'vehicle.nissan.patrol', 'vehicle.mini.cooperst', 'vehicle.mercedes-benz.coupe', 'vehicle.jeep.wrangler_rubicon']

    for car_name in car_name_list:
        # construct env
        global_options = {
            "world": {
                "dt": dt
            },
            "car_manager": {
                "car_blueprint_list": None, #  [car_name],

                "num_controlled_cars": test_num_cars
            }
        }
        env = CarlaEnv(global_options)

        # test 10 times
        is_fail = False
        test_iter = 0

        for i in range(test_times):
            # one episode
            test_iter += 1
            print("Testing {} episode {}".format(car_name, test_iter))

            is_fail = True
            min_reward = None
            env.reset()
            for step in range(test_ticks):
                # go straight
                next_obs, rew, done, _ = env.step([carla.VehicleControl(throttle=1.0) for _ in range(test_num_cars)])
                rew = np.array(rew)

                if min_reward is None:
                    min_reward = rew
                else:
                    min_reward = np.minimum(min_reward, rew)

                # check threshold
                if (min_reward <= test_reward_thresh).all():
                    is_fail = False
                    break

            if is_fail:
                break

        if is_fail:
            print("FAIL: {}".format(car_name))

        # delete env
        del env


if __name__ == "__main__":
    main()
