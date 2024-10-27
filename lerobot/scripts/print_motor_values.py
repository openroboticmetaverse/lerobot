"""
This script continuously monitors and displays the positions of all 6 robot axes.
Shows positions in format: shoulder_pan: value, shoulder_lift: value, etc.

Example of usage:
```bash
python monitor_positions.py --port /dev/ttyUSB0 --brand feetech
```
"""

import argparse
import time
import os

def monitor_positions(port, brand):
    # Define axis names and their corresponding motor IDs
    axes = {
        "shoulder_pan": 1,
        "shoulder_lift": 2,
        "elbow_flex": 3,
        "wrist_flex": 4,
        "wrist_roll": 5,
        "gripper": 6
    }

    # Import appropriate motor libraries
    if brand == "feetech":
        from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus as MotorsBusClass
    elif brand == "dynamixel":
        from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus as MotorsBusClass
    else:
        raise ValueError(
            f"Currently we do not support this motor brand: {brand}. We currently support feetech and dynamixel motors."
        )

    # Initialize motor bus with all motors configuration
    motors_config = {name: (motor_id, "sts3215") for name, motor_id in axes.items()}
    motor_bus = MotorsBusClass(port=port, motors=motors_config)

    try:
        # Connect to the motor bus at 1M baudrate
        motor_bus.connect()
        motor_bus.set_bus_baudrate(1000000)
        print(f"Connected on port {motor_bus.port} at 1M baudrate")
        print("Monitoring axis positions. Press Ctrl+C to stop.\n")

        while True:
            # Clear the terminal (works on both Windows and Unix-like systems)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Build the position string
            positions = []
            for axis_name, motor_id in axes.items():
                try:
                    position = motor_bus.read_with_motor_ids(
                        motor_bus.motor_models, motor_id, "Present_Position"
                    )
                    positions.append(f"{axis_name}: {position}")
                except Exception as e:
                    positions.append(f"{axis_name}: ERROR")

            # Print all positions on one line
            print(", ".join(positions))
            
            # Small delay to prevent overwhelming the bus
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        motor_bus.disconnect()
        print("Disconnected from motor bus.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, required=True, help="Motors bus port (e.g. /dev/ttyUSB0)")
    parser.add_argument("--brand", type=str, required=True, help="Motor brand (e.g. dynamixel,feetech)")
    args = parser.parse_args()

    monitor_positions(args.port, args.brand)