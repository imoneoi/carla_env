"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""
import carla
from enum import Enum
import numpy as np

from .controller import PIDController
# import carla_gym.utils.transforms as trans_utils


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations
    when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class LocalPlanner(object):

    def __init__(self, target_speed=0.0,
                 longitudinal_pid_params=[0.5, 0.025, 0.1],
                 lateral_pid_params=[0.75, 0.05, 0.0],
                 threshold_before=7.5,
                 threshold_after=5.0):

        self._target_speed = target_speed
        self._speed_pid = PIDController(longitudinal_pid_params)
        self._turn_pid = PIDController(lateral_pid_params)
        self._threshold_before = threshold_before
        self._threshold_after = threshold_after
        self._max_skip = 20

        self._last_command = 4

    def run_step(self, route_plan, actor_transform, actor_speed):
        target_index = -1
        for i, (waypoint, road_option) in enumerate(route_plan[0:self._max_skip]):
            if self._last_command == 4 and road_option.value != 4:
                threshold = self._threshold_before
            else:
                threshold = self._threshold_after

            distance = waypoint.transform.location.distance(actor_transform.location)
            if distance < threshold:
                self._last_command = road_option.value
                target_index = i

        if target_index < len(route_plan)-1:
            target_index += 1
        target_command = route_plan[target_index][1]
        target_location_world_coord = route_plan[target_index][0].transform.location
        target_location_actor_coord = loc_global_to_ref(target_location_world_coord, actor_transform)

        # steer
        x = target_location_actor_coord.x
        y = target_location_actor_coord.y
        theta = np.arctan2(y, x)
        steer = self._turn_pid.step(theta)

        # throttle
        target_speed = self._target_speed
        if target_command not in [3, 4]:
            target_speed *= 0.75
        delta = target_speed - actor_speed
        throttle = self._speed_pid.step(delta)

        # brake
        brake = 0.0

        # clip
        steer = np.clip(steer, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)

        return throttle, steer, brake
    
def loc_global_to_ref(target_loc_in_global, ref_trans_in_global):
    """
    :param target_loc_in_global: carla.Location in global coordinate (world, actor)
    :param ref_trans_in_global: carla.Transform in global coordinate (world, actor)
    :return: carla.Location in ref coordinate
    """
    x = target_loc_in_global.x - ref_trans_in_global.location.x
    y = target_loc_in_global.y - ref_trans_in_global.location.y
    z = target_loc_in_global.z - ref_trans_in_global.location.z
    vec_in_global = carla.Vector3D(x=x, y=y, z=z)
    vec_in_ref = vec_global_to_ref(vec_in_global, ref_trans_in_global.rotation)

    target_loc_in_ref = carla.Location(x=vec_in_ref.x, y=vec_in_ref.y, z=vec_in_ref.z)
    return target_loc_in_ref


def vec_global_to_ref(target_vec_in_global, ref_rot_in_global):
    """
    :param target_vec_in_global: carla.Vector3D in global coordinate (world, actor)
    :param ref_rot_in_global: carla.Rotation in global coordinate (world, actor)
    :return: carla.Vector3D in ref coordinate
    """
    R = carla_rot_to_mat(ref_rot_in_global)
    np_vec_in_global = np.array([[target_vec_in_global.x],
                                 [target_vec_in_global.y],
                                 [target_vec_in_global.z]])
    np_vec_in_ref = R.T.dot(np_vec_in_global)
    target_vec_in_ref = carla.Vector3D(x=np_vec_in_ref[0, 0], y=np_vec_in_ref[1, 0], z=np_vec_in_ref[2, 0])
    return target_vec_in_ref


def rot_global_to_ref(target_rot_in_global, ref_rot_in_global):
    target_roll_in_ref = cast_angle(target_rot_in_global.roll - ref_rot_in_global.roll)
    target_pitch_in_ref = cast_angle(target_rot_in_global.pitch - ref_rot_in_global.pitch)
    target_yaw_in_ref = cast_angle(target_rot_in_global.yaw - ref_rot_in_global.yaw)

    target_rot_in_ref = carla.Rotation(roll=target_roll_in_ref, pitch=target_pitch_in_ref, yaw=target_yaw_in_ref)
    return target_rot_in_ref

def rot_ref_to_global(target_rot_in_ref, ref_rot_in_global):
    target_roll_in_global = cast_angle(target_rot_in_ref.roll + ref_rot_in_global.roll)
    target_pitch_in_global = cast_angle(target_rot_in_ref.pitch + ref_rot_in_global.pitch)
    target_yaw_in_global = cast_angle(target_rot_in_ref.yaw + ref_rot_in_global.yaw)

    target_rot_in_global = carla.Rotation(roll=target_roll_in_global, pitch=target_pitch_in_global, yaw=target_yaw_in_global)
    return target_rot_in_global


def carla_rot_to_mat(carla_rotation):
    """
    Transform rpy in carla.Rotation to rotation matrix in np.array

    :param carla_rotation: carla.Rotation 
    :return: np.array rotation matrix
    """
    roll = np.deg2rad(carla_rotation.roll)
    pitch = np.deg2rad(carla_rotation.pitch)
    yaw = np.deg2rad(carla_rotation.yaw)

    yaw_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    pitch_matrix = np.array([
        [np.cos(pitch), 0, -np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0, np.cos(pitch)]
    ])
    roll_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(roll), np.sin(roll)],
        [0, -np.sin(roll), np.cos(roll)]
    ])

    rotation_matrix = yaw_matrix.dot(pitch_matrix).dot(roll_matrix)
    return rotation_matrix

def get_loc_rot_vel_in_ev(actor_list, ev_transform):
    location, rotation, absolute_velocity = [], [], []
    for actor in actor_list:
        # location
        location_in_world = actor.get_transform().location
        location_in_ev = loc_global_to_ref(location_in_world, ev_transform)
        location.append([location_in_ev.x, location_in_ev.y, location_in_ev.z])
        # rotation
        rotation_in_world = actor.get_transform().rotation
        rotation_in_ev = rot_global_to_ref(rotation_in_world, ev_transform.rotation)
        rotation.append([rotation_in_ev.roll, rotation_in_ev.pitch, rotation_in_ev.yaw])
        # velocity
        vel_in_world = actor.get_velocity()
        vel_in_ev = vec_global_to_ref(vel_in_world, ev_transform.rotation)
        absolute_velocity.append([vel_in_ev.x, vel_in_ev.y, vel_in_ev.z])
    return location, rotation, absolute_velocity

def cast_angle(x):
    # cast angle to [-180, +180)
    return (x+180.0)%360.0-180.0
