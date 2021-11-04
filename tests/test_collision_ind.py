import carla
import numpy as np
import tqdm

COLLISION_STATE = None


def _collision_callback(car_id, event):
    global COLLISION_STATE

    COLLISION_STATE[car_id] = True

def test_collision_sensor(
        num_cars: int = 100,
        num_test_rounds: int = 100,

        max_timestep: int = 5000,
        dt: float = 0.1
):
    global COLLISION_STATE

    # Create client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)  # seconds

    for r in range(num_test_rounds):
        print("Test round {}".format(r))

        # Clear collision state
        COLLISION_STATE = np.zeros(num_cars, dtype=np.bool)

        # Load world
        world = client.load_world('Town04')

        # Synchronous tick mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = dt

        settings.no_rendering_mode = True
        world.apply_settings(settings)

        # Create cars with sensors
        car_list = []
        sensor_list = []
        car_name_list = []

        car_blueprints = world.get_blueprint_library().filter("vehicle.*")
        car_spawn_points = world.get_map().get_spawn_points()
        for car_id in range(num_cars):
            # get car blueprint
            car_bp = np.random.choice(car_blueprints)

            # get spawn point
            car_pos = np.random.choice(car_spawn_points)
            car_spawn_points.remove(car_pos)

            # spawn car
            car = world.spawn_actor(car_bp, car_pos)
            car_list.append(car)
            car_name_list.append(car_bp.id)

            # create collision sensor
            sensor_blueprint = world.get_blueprint_library().find("sensor.other.collision")
            sensor = world.spawn_actor(sensor_blueprint, carla.Transform(), attach_to=car)
            sensor.listen(lambda event, id=car_id: _collision_callback(id, event))
            sensor_list.append(sensor)

        world.tick()

        # Tick
        passed = False
        for step in tqdm.tqdm(range(max_timestep)):
            if COLLISION_STATE.all():
                passed = True
                break

            # apply controls
            [car.apply_control(carla.VehicleControl(throttle=1.0)) for car in car_list]

            # tick
            world.tick()

        if passed:
            print("PASS at {}".format(step))
        else:
            print("FAIL, exceeded {} steps".format(step))
            # print(car_name_list)
            # print(COLLISION_STATE)
            print([car_name_list[i] for i in range(num_cars) if COLLISION_STATE[i] == False])

        # destroy cars and sensors
        [sensor.destroy() for sensor in sensor_list]
        [car.destroy() for car in car_list]


if __name__ == "__main__":
    test_collision_sensor()