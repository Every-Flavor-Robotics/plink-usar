import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from gpiozero import Device, Servo
from gpiozero.pins.pigpio import PiGPIOFactory
from motorgo_msgs.msg import MotorGoEncoderData, MotorGoMotorCommand
from nav_msgs.msg import Odometry
from rclpy.node import Node

Device.pin_factory = PiGPIOFactory()


class RoverDriver(Node):
    """ROS node to control the rover rover. Maps from cmd_vel to motor commands.

    Args:
        Node (_type_): _description_
    """

    def __init__(self):
        """Initializes the RoverDriver class."""
        super().__init__("rover_driver")

        #### Constants

        # PWM constants for steering servo
        self.min_calibration_pulse_width = 0.87
        self.max_calibration_pulse_width = 1.88
        self.min_calibration_angle = -90
        self.max_calibration_angle = 90
        self.pulse_width_center = (
            self.min_calibration_pulse_width + self.max_calibration_pulse_width
        ) / 2
        self.pulse_width_per_degree = (
            self.max_calibration_pulse_width - self.min_calibration_pulse_width
        ) / (self.max_calibration_angle - self.min_calibration_angle)

        # Target steering range
        self.steering_range = [-60, 60]
        self.steering_angle = 0

        # Wheel base constants
        self.wheel_base_length = 0.2
        self.wheel_base_width = 0.2

        # Set up servo
        self.steering_servo = Servo(
            13,
            min_pulse_width=self._angle_to_pulse_width(self.steering_range[0]),
            max_pulse_width=self._angle_to_pulse_width(self.steering_range[1]),
        )

        self.steering_servo.value = self._angle_to_value(-60)
        time.sleep(0.3)
        self.steering_servo.value = self._angle_to_value(60)
        time.sleep(0.6)
        self.steering_servo.value = self._angle_to_value(0)

        # Subscribe to the encoder data
        self.encoder_sub = self.create_subscription(
            MotorGoEncoderData, "motorgo/encoder", self.encoder_callback, 10
        )

        # Publish the motor commands
        self.motor_command_pub = self.create_publisher(
            MotorGoMotorCommand, "motorgo/command", 10
        )

        # Subscribe to cmd_vel
        self.cmd_vel_sub = self.create_subscription(
            Twist, "cmd_vel", self.cmd_vel_callback, 10
        )

        self.odometry_pub = self.create_publisher(Odometry, "odom", 10)

        self.odometry_data = Odometry()
        self.last_odom_time = self.get_clock().now()
        self.odometry_data.header.stamp = self.get_clock().now().to_msg()

        # Log the initialization
        self.get_logger().info("Rover driver initialized")

    def _angle_to_pulse_width(self, angle: float) -> float:
        """Converts the angle to pulse width.

        Args:
            angle (float): The angle to convert. 0 is center, -90 is left, 90 is right.

        Returns:
            float: The pulse width.
        """

        return (self.pulse_width_center + angle * self.pulse_width_per_degree) / 1000

    def _angle_to_value(self, angle: float) -> float:
        """Converts the angle to a servo value.

        Args:
            angle (float): The angle to convert. 0 is center, -90 is left, 90 is right.

        Returns:
            float: The servo value from -1 to 1.
        """

        return (angle - self.steering_range[0]) / (
            self.steering_range[1] - self.steering_range[0]
        ) * 2 - 1

    def set_steering_angle(self, angle: float):
        """Sets the steering angle of the rover.

        Args:
            angle (float): The angle to set the steering to.
        """

        # clamp the angle to the range
        self.steering_angle = max(
            self.steering_range[0], min(self.steering_range[1], angle)
        )

        # set the angle
        # Compute the pulse width from the angle
        self.steering_servo.value = self._angle_to_value(self.steering_angle)

    def encoder_callback(self, msg: MotorGoEncoderData):
        """Callback function for the encoder data.

        Args:
            msg (MotorGoEncoderData): The encoder data message.
        """

        # Compute and publish the odometry
        time = self.get_clock().now()
        dt = (time - self.last_odom_time).nanoseconds * 1e-9
        self.last_odom_time = time

        # Stamp
        self.odometry_data.header.stamp = self.get_clock().now().to_msg()
        # Frame ID
        self.odometry_data.header.frame_id = "odom"
        # Child frame ID
        self.odometry_data.child_frame_id = "base_link"

        # Velocity
        linear_velocity = (msg.motor_1_velocity + msg.motor_2_velocity) / 2

        self.odometry_data.twist.twist.linear.x = linear_velocity
        self.odometry_data.twist.twist.angular.z = (
            linear_velocity / self.wheel_base_length
        ) * np.tan(np.deg2rad(self.steering_angle))

        # Compute the position
        self.odometry_data.pose.pose.position.x += (
            linear_velocity * np.cos(self.odometry_data.pose.pose.orientation.z) * dt
        )
        self.odometry_data.pose.pose.position.y += (
            linear_velocity * np.sin(self.odometry_data.pose.pose.orientation.z) * dt
        )
        self.odometry_data.pose.pose.orientation.z += (
            self.odometry_data.twist.twist.angular.z
        ) * dt

    def cmd_vel_callback(self, msg: Twist):
        """Callback function for the cmd_vel message.

        Args:
            msg (Twist): The cmd_vel message.
        """

        # Handle edge cases: no linear velocity
        if msg.linear.x == 0 and msg.angular.z == 0:
            self.set_steering_angle(0)

        # Handle edge cases: no angular velocity
        elif msg.angular.z == 0:
            self.set_steering_angle(0)
        else:
            # Compute the steering angle from the twist message
            steering_angle = np.arctan2(
                self.wheel_base_length * msg.angular.z, msg.linear.x
            )

            # Set the steering angle
            self.set_steering_angle(np.rad2deg(steering_angle))

        # Compute the wheel speeds
        # Target velocity of the center of the robot is the linear velocity
        # Need to handle differential wheel speeds, so the outer wheel has to go faster

        left_speed = msg.linear.x - msg.angular.z * self.wheel_base_length / 2
        right_speed = msg.linear.x + msg.angular.z * self.wheel_base_length / 2

        # Publish the motor commands
        motor_command = MotorGoMotorCommand()
        # Stamp
        motor_command.header.stamp = self.get_clock().now().to_msg()
        # Left motor
        motor_command.motor_1_power = left_speed
        # Right motor
        motor_command.motor_2_power = -right_speed

        self.motor_command_pub.publish(motor_command)


def main(args=None):
    """Main function to run the RoverDriver node."""

    rclpy.init(args=args)

    # Create the node
    node = RoverDriver()

    # Spin forever
    rclpy.spin(node)


if __name__ == "__main__":
    main()
    main()
