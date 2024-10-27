"""
This script reads values from motors connected to the specified port.

Example of usage:
```bash
python find_motors_id.py \
  --port /dev/ttyACM0 \
  --brand feetech \
  --model sts3215
```
"""

import argparse
import time

def read_motor_values(port, brand, model):
    # Import appropriate motor libraries based on brand
    if brand == "feetech":
        from lerobot.common.robot_devices.motors.feetech import MODEL_BAUDRATE_TABLE
        from lerobot.common.robot_devices.motors.feetech import (
            SCS_SERIES_BAUDRATE_TABLE as SERIES_BAUDRATE_TABLE,
        )
        from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus as MotorsBusClass
    elif brand == "dynamixel":
        from lerobot.common.robot_devices.motors.dynamixel import MODEL_BAUDRATE_TABLE
        from lerobot.common.robot_devices.motors.dynamixel import (
            X_SERIES_BAUDRATE_TABLE as SERIES_BAUDRATE_TABLE,
        )
        from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus as MotorsBusClass
    else:
        raise ValueError(
            f"Currently we do not support this motor brand: {brand}. We currently support feetech and dynamixel motors."
        )

    # Validate motor model
    if model not in MODEL_BAUDRATE_TABLE:
        raise ValueError(
            f"Invalid model '{model}' for brand '{brand}'. Supported models: {list(MODEL_BAUDRATE_TABLE.keys())}"
        )

    # Initialize motor bus with temporary motor configuration
    motor_bus = MotorsBusClass(port=port, motors={"temp_motor": (1, model)})

    try:
        # Connect to the motor bus
        motor_bus.connect()
        print(f"Connected on port {motor_bus.port}")

        # Scan all possible baudrates to find connected motors
        print("Scanning for connected motors...")
        all_baudrates = set(SERIES_BAUDRATE_TABLE.values())
        found_motors = []

        for baudrate in all_baudrates:
            motor_bus.set_bus_baudrate(baudrate)
            present_ids = motor_bus.find_motor_indices(list(range(1, 254)))  # Scan full ID range
            if present_ids:
                print(f"Found motors at baudrate {baudrate}: {present_ids}")
                for motor_id in present_ids:
                    found_motors.append((baudrate, motor_id))

        if not found_motors:
            print("No motors detected. Please check connections.")
            return

        # Read values from each found motor
        for baudrate, motor_id in found_motors:
            print(f"\nReading values from motor ID {motor_id} at baudrate {baudrate}")
            motor_bus.set_bus_baudrate(baudrate)
            
            try:
                # Read common parameters (adjust these based on your needs)
                present_position = motor_bus.read_with_motor_ids(
                    motor_bus.motor_models, motor_id, "Present_Position"
                )
                present_voltage = motor_bus.read_with_motor_ids(
                    motor_bus.motor_models, motor_id, "Present_Voltage"
                )
                present_temperature = motor_bus.read_with_motor_ids(
                    motor_bus.motor_models, motor_id, "Present_Temperature"
                )
                
                print(f"Motor ID: {motor_id}")
                print(f"Position: {present_position}")
                print(f"Voltage: {present_voltage}")
                print(f"Temperature: {present_temperature}°C")

            except Exception as e:
                print(f"Error reading from motor {motor_id}: {e}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        motor_bus.disconnect()
        print("\nDisconnected from motor bus.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, required=True, help="Motors bus port (e.g. /dev/ttyACM0)")
    parser.add_argument("--brand", type=str, required=True, help="Motor brand (e.g. dynamixel,feetech)")
    parser.add_argument("--model", type=str, required=True, help="Motor model (e.g. xl330-m077,sts3215)")
    args = parser.parse_args()

    read_motor_values(args.port, args.brand, args.model)