from car.reward import CarReward
# from car.map import CarMap

import weakref

import carla
import numpy as np
import threading


class Car:
    def __init__(self,
                 options: dict,
                 reward_weights: dict,
                 actor: carla.Vehicle,
                 world: carla.World):
        # default options
        self.options = {
            "camera_x": 640,
            "camera_y": 320,
            "camera_postprocess": False
        }
        self.options.update(options)

        # actor
        self.actor = actor
        self.world = world

        # cameras
        self.camera_pos = []
        self.cameras = []
        self.camera_images = []
        self.camera_event = []

        self.set_camera_pos()
        self.create_cameras()

        # sensors
        self.sensors = []

        self.collision_events = []
        self.lane_invasion_events = []
        self.create_collision_sensor()
        self.create_lane_invasion_sensor()

        # map
        # self.map = CarMap(self, world, self.options.get("map", {}))

        # reward
        self.reward = CarReward(self, reward_weights, self.options.get("reward", {}))

    def destroy(self):
        [cam.destroy() for cam in self.cameras]
        [sensor.destroy() for sensor in self.sensors]

        self.actor.destroy()

    def set_camera_pos(self):
        self.camera_pos = []

        # front camera
        self.camera_pos.append(carla.Transform(
            carla.Location(
                x=np.random.uniform(1.6, 2),
                y=np.random.uniform(-0.025, 0.025),
                z=np.random.uniform(1.0, 2)
            ),
            carla.Rotation(
                yaw=np.random.uniform(-1.0, 1.0),
                pitch=np.random.uniform(-15.0, 15.0),
                roll=np.random.uniform(-1.0, 1.0),
            )
        ))

    def create_cameras(self):
        self.cameras = []
        self.camera_images = []
        self.camera_event = []

        # create blueprint
        cam_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_blueprint.set_attribute('image_size_x', str(self.options["camera_x"]))
        cam_blueprint.set_attribute('image_size_y', str(self.options["camera_y"]))
        cam_blueprint.set_attribute('enable_postprocess_effects', str(self.options["camera_postprocess"]))

        for cam_id, cam_pos in enumerate(self.camera_pos):
            cam = self.world.spawn_actor(cam_blueprint,
                                         cam_pos,
                                         attach_to=self.actor,
                                         attachment_type=carla.AttachmentType.Rigid)
            weak_self = weakref.ref(self)
            cam.listen(lambda img: Car._camera_callback(weak_self, cam_id, img))

            self.cameras.append(cam)
            self.camera_images.append(None)
            self.camera_event.append(threading.Event())

    @staticmethod
    def _camera_callback(weak_self, cam_id, carla_img):
        """
        WARNING: This Function is Executed in different thread!
        """
        self = weak_self()
        if not self:
            return

        # convert image
        # extract rgba image from carla_img
        img = np.frombuffer(carla_img.raw_data, dtype=np.dtype("uint8"))
        img = np.reshape(img, (carla_img.height, carla_img.width, 4))
        # extract rgb channel
        img = img[:, :, :3]
        # bgr to rgb
        img = img[:, :, ::-1]

        # save image
        self.camera_images[cam_id] = img

        # trigger semaphore
        self.camera_event[cam_id].set()

    def create_collision_sensor(self):
        # collision
        sensor_blueprint = self.world.get_blueprint_library().find('sensor.other.collision')
        sensor = self.world.spawn_actor(sensor_blueprint, carla.Transform(), attach_to=self.actor)

        weak_self = weakref.ref(self)
        sensor.listen(lambda event: Car._collision_callback(weak_self, event))

        self.sensors.append(sensor)

    @staticmethod
    def _collision_callback(weak_self, event):
        self = weak_self()
        if not self:
            return

        self.collision_events.append(event)

    def create_lane_invasion_sensor(self):
        # collision
        sensor_blueprint = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        sensor = self.world.spawn_actor(sensor_blueprint, carla.Transform(), attach_to=self.actor)

        weak_self = weakref.ref(self)
        sensor.listen(lambda event: Car.lane_invasion_callback(weak_self, event))

        self.sensors.append(sensor)

    @staticmethod
    def lane_invasion_callback(weak_self, event):
        self = weak_self()
        if not self:
            return

        self.lane_invasion_events.append(event)

    def sync_before_tick(self):
        # clear events
        [event.clear() for event in self.camera_event]

    def sync_after_tick(self):
        # acquire events
        [event.wait() for event in self.camera_event]

    def sync_clear_events(self):
        # clear events
        self.collision_events = []
        self.lane_invasion_events = []

    def apply_control(self, action):
        if isinstance(action, np.ndarray):
            accel = np.clip(action[0], -1, 1)
            steer = np.clip(action[1], -1, 1)

            action = carla.VehicleControl(
                throttle=max(0.0, accel),
                brake=-min(0.0, accel),

                steer=steer,
            )

        self.actor.apply_control(action)

    def get_observation(self):
        # return images
        return self.camera_images, self.actor.get_velocity()

    def get_reward_done(self, world_map):
        result = self.reward.get_reward_done(world_map)

        self.sync_clear_events()
        return result
