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
            # camera properties
            "camera_x": 640,
            "camera_y": 320,
            "camera_postprocess": False,
            # bev camera properties
            "bev_camera_x": 800,
            "bev_camera_y": 600,
            "bev_fov": 90,
            "bev_camera_height": 40.0,

            # vehicle physics properties
            "max_rpm": 5000.0,
            "center_of_mass": carla.Vector3D(0.0,0.0,0.0),
            "torque_curve": [[0, 500], [5000, 500]],
            "moi": 1.0,
            "damping_rate_full_throttle": 0.15,
            "damping_rate_zero_throttle_clutch_engaged": 2.0,
            "damping_rate_zero_throttle_clutch_disengaged": 0.35,
            "use_gear_autobox": True,
            "gear_switch_time": 0.5,
            "clutch_strength": 10.0,
            "final_ratio": 4.0,
            # gears: ratio=1.0, down_ratio=0.5, up_ratio=0.65
            "forward_gears": [carla.GearPhysicsControl(1.0, 0.5, 0.65)],
            "drag_coefficient": 0.3,
            "steering_curve": [[0.0, 1.0], [10.0, 0.5]], 
            # wheels: tire_friction=2.0, damping_rate=0.25, max_steer_angle=70.0, radius=30.0, max_brake_torque=1500.0, max_handbrake_torque=3000.0, position=(0.0,0.0,0.0)
            "wheels": [carla.WheelPhysicsControl(2.0, 0.25, 70.0, 30.0, 1500.0, 3000.0, 2.0, 17.0, 1000.0, carla.Vector3D(0.0,0.0,0.0))], # add args for 0.9.11->0.9.13
            "use_sweep_wheel_collision": False, 
            "mass": 1000.0
        }
        self.options.update(options)

        # actor
        self.actor = actor
        self.actor.apply_physics_control(
            carla.VehiclePhysicsControl(
                max_rpm=self.options["max_rpm"], 
                center_of_mass=self.options["center_of_mass"],
                torque_curve=self.options["torque_curve"],
                moi=self.options["moi"],
                damping_rate_full_throttle=self.options["damping_rate_full_throttle"],
                damping_rate_zero_throttle_clutch_engaged=self.options["damping_rate_zero_throttle_clutch_engaged"],
                damping_rate_zero_throttle_clutch_disengaged=self.options["damping_rate_zero_throttle_clutch_disengaged"],
                use_gear_autobox=self.options["use_gear_autobox"],
                gear_switch_time=self.options["gear_switch_time"],
                clutch_strength=self.options["clutch_strength"],
                final_ratio=self.options["final_ratio"],
                forward_gears=self.options["forward_gears"],
                drag_coefficient=self.options["drag_coefficient"],
                steering_curve=self.options["steering_curve"],
                wheels=self.options["wheels"],
                use_sweep_wheel_collision=self.options["use_sweep_wheel_collision"],
                mass=self.options["mass"]
                )
            )
        self.world = world

        # cameras
        self.camera_pos = []
        self.cameras = []
        self.camera_images = []
        self.camera_event = []
        # bev cameras
        self.bev_camera_pos = []
        self.bev_cameras = []
        self.bev_camera_images = []
        self.bev_camera_event = []
        # bev seg cameras
        self.seg_bev_camera_pos = []
        self.seg_bev_cameras = []
        self.seg_bev_camera_images = []
        self.seg_bev_camera_event = []

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
        # destroy cameras and other sensors
        [cam.destroy() for cam in self.cameras and self.bev_cameras and self.seg_bev_cameras]
        [sensor.destroy() for sensor in self.sensors]

        # destroy the vehicle
        self.actor.destroy()

    def set_camera_pos(self):
        self.camera_pos = []
        self.bev_camera_pos = []

        car_extent_x = self.actor.bounding_box.extent.x
        # front camera
        self.camera_pos.append(carla.Transform(
            carla.Location(
                x=car_extent_x + np.random.uniform(0, 1.0),
                y=np.random.uniform(-0.02, 0.02),
                z=np.random.uniform(1.0, 2)
            ),
            carla.Rotation(
                yaw=np.random.uniform(-1.0, 1.0),
                pitch=np.random.uniform(-15.0, 15.0),
                roll=np.random.uniform(-1.0, 1.0),
            )
        ))
        
        # BEV camera
        self.bev_camera_pos.append(carla.Transform(
            carla.Location(
                x=0.0,
                y=0.0,
                z=self.options["bev_camera_height"],
            ),
            carla.Rotation(
                yaw=0.0,
                pitch=-90.0,
                roll=0.0,
            )
        ))
        
        # BEV seg camera
        self.seg_bev_camera_pos.append(carla.Transform(
            carla.Location(
                x=0.0,
                y=0.0,
                z=self.options["bev_camera_height"],
            ),
            carla.Rotation(
                yaw=0.0,
                pitch=-90.0,
                roll=0.0,
            )
        ))

    def create_cameras(self):
        self.cameras = []
        self.camera_images = []
        self.camera_event = []
        
        self.bev_cameras = []
        self.bev_camera_images = []
        self.bev_camera_event = []
        
        self.seg_bev_cameras = []
        self.seg_bev_camera_images = []
        self.seg_bev_camera_event = []

        # create blueprint
        cam_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_blueprint.set_attribute('image_size_x', str(self.options["camera_x"]))
        cam_blueprint.set_attribute('image_size_y', str(self.options["camera_y"]))
        cam_blueprint.set_attribute('enable_postprocess_effects', str(self.options["camera_postprocess"]))
        
        bev_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bev_blueprint.set_attribute('image_size_x', str(self.options["bev_camera_x"]))
        bev_blueprint.set_attribute('image_size_y', str(self.options["bev_camera_y"]))
        bev_blueprint.set_attribute('enable_postprocess_effects', str(self.options["camera_postprocess"]))
        bev_blueprint.set_attribute('fov', str(self.options["bev_fov"]))
        
        seg_bev_blueprint = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        seg_bev_blueprint.set_attribute('image_size_x', str(self.options["bev_camera_x"]))
        seg_bev_blueprint.set_attribute('image_size_y', str(self.options["bev_camera_y"]))
        seg_bev_blueprint.set_attribute('fov', str(self.options["bev_fov"]))

        # attach the cameras to the vehicle rigidly
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
            
        # attach the bev cameras to the vehicle rigidly
        for bev_cam_id, bev_cam_pos in enumerate(self.bev_camera_pos):
            bev_cam = self.world.spawn_actor(bev_blueprint,
                                        bev_cam_pos,
                                        attach_to=self.actor,
                                        attachment_type=carla.AttachmentType.Rigid)
            weak_self = weakref.ref(self)
            bev_cam.listen(lambda bev_img: Car._bev_camera_callback(weak_self, bev_cam_id, bev_img))

            self.bev_cameras.append(bev_cam)
            self.bev_camera_images.append(None)
            self.bev_camera_event.append(threading.Event())
            
        # attach the bev cameras to the vehicle rigidly
        for seg_bev_cam_id, seg_bev_cam_pos in enumerate(self.seg_bev_camera_pos):
            seg_bev_cam = self.world.spawn_actor(seg_bev_blueprint,
                                        seg_bev_cam_pos,
                                        attach_to=self.actor,
                                        attachment_type=carla.AttachmentType.Rigid)
            weak_self = weakref.ref(self)
            seg_bev_cam.listen(lambda seg_bev_img: Car._seg_bev_camera_callback(weak_self, seg_bev_cam_id, seg_bev_img))

            self.seg_bev_cameras.append(seg_bev_cam)
            self.seg_bev_camera_images.append(None)
            self.seg_bev_camera_event.append(threading.Event())

    @staticmethod
    def _camera_callback(weak_self, cam_id, carla_img):
        """
        WARNING: This Function is Executed in different thread!
        """
        self = weak_self()
        if not self:
            return

        # convert image
        # extract rgba(4 channels) image from carla_img
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
        
    @staticmethod
    def _bev_camera_callback(weak_self, cam_id, carla_img):
        """
        WARNING: This Function is Executed in different thread!
        """
        self = weak_self()
        if not self:
            return

        # convert image
        # extract rgba(4 channels) image from carla_img
        img = np.frombuffer(carla_img.raw_data, dtype=np.dtype("uint8"))
        img = np.reshape(img, (carla_img.height, carla_img.width, 4))
        # extract rgb channel
        img = img[:, :, :3]
        # bgr to rgb
        img = img[:, :, ::-1]

        # save image
        self.bev_camera_images[cam_id] = img

        # trigger semaphore
        self.bev_camera_event[cam_id].set()
        
    @staticmethod
    def _seg_bev_camera_callback(weak_self, cam_id, carla_img):
        """
        WARNING: This Function is Executed in different thread!
        """
        self = weak_self()
        if not self:
            return

        # convert image
        # extract rgba(4 channels) image from carla_img
        img = np.frombuffer(carla_img.raw_data, dtype=np.dtype("uint8"))
        img = np.reshape(img, (carla_img.height, carla_img.width, 4))
        # extract B channel for segment label
        seg_label = img[:, :, 2]  # Extract B channel for segment label
        
        """
        Convert an image containing CARLA semantic segmentation labels to
        Cityscapes palette.
        """
        classes = {
            0: [0, 0, 0],         # None
            1: [70, 70, 70],      # Buildings
            2: [190, 153, 153],   # Fences
            3: [72, 0, 90],       # Other
            4: [220, 20, 60],     # Pedestrians
            5: [153, 153, 153],   # Poles
            6: [157, 234, 50],    # RoadLines
            7: [128, 64, 128],    # Roads
            8: [244, 35, 232],    # Sidewalks
            9: [107, 142, 35],    # Vegetation
            10: [0, 0, 255],      # Vehicles
            11: [102, 102, 156],  # Walls
            12: [220, 220, 0]     # TrafficSigns
        }
        seg_label_rgb = np.zeros((seg_label.shape[0], seg_label.shape[1], 3))
        for key, value in classes.items():
            seg_label_rgb[np.where(seg_label == key)] = value
            
        # save image
        self.seg_bev_camera_images[cam_id] = seg_label_rgb.astype(np.uint8)

        # trigger semaphore
        self.seg_bev_camera_event[cam_id].set()

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
        [event.clear() for event in self.camera_event and self.bev_camera_event and self.seg_bev_camera_event]

    def sync_after_tick(self):
        # acquire events
        [event.wait() for event in self.camera_event and self.bev_camera_event and self.seg_bev_camera_event]

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

        if action is not None:
            self.actor.apply_control(action)

    def get_observation(self):
        # return images
        return self.camera_images, self.bev_camera_images, self.seg_bev_camera_images, self.actor.get_velocity()

    def get_reward_done(self, world_map):
        result = self.reward.get_reward_done(world_map)

        self.sync_clear_events()
        return result
