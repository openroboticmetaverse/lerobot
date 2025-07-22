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

from typing import Protocol

from lerobot.common.robot_devices.motors.configs import (
    DynamixelMotorsBusConfig,
    FeetechMotorsBusConfig,
    SimulatedMotorsBusConfig,
    MotorsBusConfig,
)

import threading  as th

class MotorsBus(Protocol):
    def motor_names(self): ...
    def set_calibration(self): ...
    def apply_calibration(self): ...
    def revert_calibration(self): ...
    def read(self): ...
    def write(self): ...


def make_motors_buses_from_configs(motors_bus_configs: dict[str, MotorsBusConfig]) -> list[MotorsBus]:
    motors_buses = {}

    for key, cfg in motors_bus_configs.items():
        if cfg.type == "dynamixel":
            from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus

            motors_buses[key] = DynamixelMotorsBus(cfg)

        elif cfg.type == "feetech":
            from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

            motors_buses[key] = FeetechMotorsBus(cfg)
        
        elif cfg.type == "simulated":
            from lerobot.common.robot_devices.motors.simulated_motors import SimulatedMotorsBus

            motors_buses[key] = SimulatedMotorsBus(cfg)

        else:
            raise ValueError(f"The motor type '{cfg.type}' is not valid.")

    return motors_buses


def make_motors_bus(motor_type: str, **kwargs) -> MotorsBus:
    if motor_type == "dynamixel":
        from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus

        config = DynamixelMotorsBusConfig(**kwargs)
        return DynamixelMotorsBus(config)

    elif motor_type == "feetech":
        from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

        config = FeetechMotorsBusConfig(**kwargs)
        return FeetechMotorsBus(config)

    elif motor_type == "simulated":
        from lerobot.common.robot_devices.motors.simulated_motors import SimulatedMotorsBus

        config = SimulatedMotorsBusConfig(**kwargs)
        return SimulatedMotorsBus(config)


    else:
        raise ValueError(f"The motor type '{motor_type}' is not valid.")


class dataBuffer:
    def __init__(self):
        self._lock = th.Lock()
        self._condition = th.Condition(self._lock)
        self._value = None
        self._new_data = False
        self._closed = False

    def write(self, value):
        with self._lock:
            if self._closed:
                raise RuntimeError("Buffer is closed")
            self._value = value
            self._new_data = True
            self._condition.notify_all()
            return True
        return False

    def read(self):
        with self._lock:
            while not self._new_data and not self._closed:
                self._condition.wait(timeout=0.01)
            if self._closed:
                raise RuntimeError("Buffer is closed")
            value = self._value
            self._new_data = False
            return value

    def check_new_data(self):
        with self._lock:
            return self._new_data

    def close(self):
        with self._lock:
            self._closed = True
            self._condition.notify_all()