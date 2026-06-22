"""ROS2 publishers for robot motion and TTS / lip-sync control."""

import time

from std_msgs.msg import Bool, Empty as EmptyMsg

from constants import MAX_EYE_X, MAX_EYE_Y, MOTOR_IDS
from haru2_core_msgs.msg import (
    EyesPose,
    LCDCommand,
    LedsArray,
    MotorCommand,
    TTSCommand,
)

_DEFAULT_MOVE_DURATION_S = 0.1
_RESET_WAIT_S = 2.0


def _make_motor_cmd(motor_id: int, position: float, duration_s: float) -> MotorCommand:
    """Builds a non-relative absolute motor command."""
    return MotorCommand(
        motor=motor_id,
        position=float(position),
        play_time=int(duration_s * 1000),
        relative=False,
        disable_eyes_roll_sync=False,
    )


class RobotController:
    """Publishes motor / eye / LED commands to the Haru 2 robot."""

    def __init__(self, node) -> None:
        self.node = node
        self._setup_publishers()

    def _setup_publishers(self) -> None:
        self._pub_base = self.node.create_publisher(
            MotorCommand, '/haru2/cmd_base_pos', 10)
        self._pub_neck_pitch = self.node.create_publisher(
            MotorCommand, '/haru2/cmd_neck_pitch_pos', 10)
        self._pub_neck_roll = self.node.create_publisher(
            MotorCommand, '/haru2/cmd_neck_roll_pos', 10)
        self._pub_mouth_leds = self.node.create_publisher(
            LedsArray, '/haru2/cmd_mouth_leds', 10)
        self._pub_lcd = self.node.create_publisher(
            LCDCommand, '/haru2/cmd_lcd', 10)
        self._pub_eyes = self.node.create_publisher(
            EyesPose, '/haru2/cmd_eyes_pose', 10)
        self._pub_home = self.node.create_publisher(
            EmptyMsg, '/haru2/cmd_move_home', 10)

    # ---------------------------------------------------------------------
    # Motion commands
    # ---------------------------------------------------------------------

    def move_base(self, position: float,
                  duration: float = _DEFAULT_MOVE_DURATION_S) -> None:
        self._pub_base.publish(
            _make_motor_cmd(MOTOR_IDS['base_yaw'], position, duration))

    def move_neck_pitch(self, pitch: float,
                        duration: float = _DEFAULT_MOVE_DURATION_S) -> None:
        self._pub_neck_pitch.publish(
            _make_motor_cmd(MOTOR_IDS['neck_pitch'], pitch, duration))

    def move_neck_roll(self, roll: float,
                       duration: float = _DEFAULT_MOVE_DURATION_S) -> None:
        self._pub_neck_roll.publish(
            _make_motor_cmd(MOTOR_IDS['neck_roll'], roll, duration))

    def move_eyes(self, eye_num: int, pos_x: float, pos_y: float) -> None:
        """Publishes a clamped EyesPose for the given eye index."""
        pose = EyesPose()
        pose.eye = eye_num
        pose.pos_x = float(max(0.0, min(MAX_EYE_X, pos_x)))
        pose.pos_y = float(max(0.0, min(MAX_EYE_Y, pos_y)))
        self._pub_eyes.publish(pose)

    def reset_motors(self) -> None:
        """Returns all motors to home position and waits for completion."""
        self._pub_home.publish(EmptyMsg())
        self.node.get_logger().info('Resetting all motors to home position...')
        time.sleep(_RESET_WAIT_S)


class TTSPublisher:
    """Publisher for text-to-speech commands and lip-sync toggling."""

    def __init__(self, node) -> None:
        self.node = node
        self._tts_pub = self.node.create_publisher(
            TTSCommand, '/haru2/cmd_tts', 10)
        self._lip_sync_pub = self.node.create_publisher(
            Bool, '/haru2/enable_lip_sync', 10)
        self.enable_lip_sync(True)

    def publish_tts(self, text: str, language_code: str = 'en') -> None:
        msg = TTSCommand()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.message = text
        msg.disable_lipsync = False
        msg.language_code = language_code
        self._tts_pub.publish(msg)
        print(f'TTS: {text}')

    def enable_lip_sync(self, enable: bool) -> None:
        msg = Bool()
        msg.data = enable
        self._lip_sync_pub.publish(msg)
