# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import logging
import math
import sys
import time
import traceback
from copy import deepcopy

import numpy as np
import tqdm
import gymnasium as gym

from lerobot.common.robot_devices.motors.configs import SimulatedMotorsBusConfig
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc
from lerobot.common.robot_devices.motors.utils import dataBuffer

import threading as th

PROTOCOL_VERSION = 0


class SimulatedMotorsBus:

    def __init__(
        self,
        config: SimulatedMotorsBusConfig,
    ):  
        self.handle = config.handle
        self.params = config.params['env']['gym']
        self.thread = None
        self.stop_event = th.Event()
        self.sim_target_pose_buffer = dataBuffer()
        self.sim_current_pose_buffer = dataBuffer()
        self.is_connected = False
        self.group_readers = {}
        self.group_writers = {}
        self.logs = {}

        self.track_positions = {}
        
    
    @staticmethod
    def run_simulation(handle, stop_event, q_sim_target_pose_db: dataBuffer, q_sim_current_pose_db: dataBuffer, env_parms, teleop_time_s=None):
        from gymnasium.envs.registration import register

        register(
            id="LerobotCube-v0",
            entry_point="lerobot.sim.lerobot_env:LerobotEnv",
            max_episode_steps=50,
        )
        env = gym.make(handle, disable_env_checker=True, **env_parms)
        #DOF = env.num_dof
        DOF = env.unwrapped.num_dof
        env.reset()
        reset_time = None
        # reset_time = 120
        rc = 1
        start_teleop_t = time.perf_counter()
        curr_sim_pose = None
        curr_leader_pos = None
        i = 0
        try:
            while not stop_event.is_set():
                i += 1
                # Simulate the robot's behavior
                
                if q_sim_target_pose_db.check_new_data():
                    curr_leader_pos = q_sim_target_pose_db.read()
                    
                if curr_leader_pos is not None and curr_sim_pose is not None:
                    action = curr_leader_pos - curr_sim_pose
                else:
                    action = np.zeros(DOF, dtype=np.float32)
                
                observation, reward, terminted, truncated, info = env.step(action)
                curr_sim_pose = np.array(observation['arm_qpos'], dtype=np.float32)
                images = {"front": observation["image_front"], "top": observation["image_top"]}
                write_data = [curr_sim_pose, images]
                q_sim_current_pose_db.write(write_data)

                # time.sleep(0.01)
                
                if reset_time is not None and (time.perf_counter() - start_teleop_t*rc) >= reset_time:
                    print("resetting environment")
                    env.reset()
                    rc += 1
                
                if teleop_time_s is not None and time.perf_counter() - start_teleop_t > teleop_time_s:
                    print("Teleoperation processes finished.")
                    break
        except Exception:
            traceback.print_exc()
            raise
        finally:
            env.close()
            stop_event.set()
            print("Simulation stopped.")
            if teleop_time_s is not None:
                print("Teleoperation processes finished.")
                

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"Simulation is already running. Do not call `motors_bus.connect()` twice."
            )
        try:
            "Try to connect to the bus"
            self.thread = th.Thread(target=self.run_simulation, args=(self.handle, self.stop_event, self.sim_target_pose_buffer, self.sim_current_pose_buffer, self.params), daemon = True)
            self.thread.start()
            self.stop_event.clear()
        except Exception:
            traceback.print_exc()
            sys.exit()
            raise

        # Allow to read and write
        self.is_connected = True

    def reconnect(self):
        try:
            "Try to connect to the bus"
            if self.thread is None: self.thread = th.Thread(target=self.run_simulation, args=(self.handle, self.stop_event, self.sim_target_pose_buffer, self.sim_current_pose_buffer, self.params), daemon = True)
    
            self.thread.start()
            self.stop_event.clear()
        except Exception:
            traceback.print_exc()
            raise

        # Allow to read and write
        self.is_connected = True


    def set_calibration(self, calibration: dict[str, list]):
        self.calibration = calibration


    def real_positions_to_sim(self, values: np.ndarray):
        """Counts - starting position -> radians -> align axes -> offset"""
        # return axis_directions * (real_positions - start_pos) * 2.0 * np.pi / 4096 + offsets
        HALF_TURN_DEGREE = 180
        resolution = 4096

        drive_mode = self.calibration["drive_mode"]
        homing_offset = self.calibration["homing_offset"]
        
        values = values / HALF_TURN_DEGREE * (resolution // 2)
        values -= homing_offset
        values *= np.where(drive_mode==1, -1, 1)
        values[1] *= -1
        values = (values * HALF_TURN_DEGREE / (resolution // 2)) * np.pi / 180
        values[-1] -= 0.8
        return values
    
    def read(self):
        curr_sim_pose = self.sim_current_pose_buffer.read()
        return curr_sim_pose
    
    def apply_calibration(self):
        raise NotImplementedError("Apply calibration is not implemented for simulated motors.")
    
    def revert_calibration(self):
        raise NotImplementedError("Revert calibration is not implemented for simulated motors.")


    def write(self, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        values = np.array(values)
        values = self.real_positions_to_sim(values)
        values = values.tolist()
        self.sim_target_pose_buffer.write(values)

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"Simulation is not running. Try running `motors_bus.connect()` first."
            )
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
