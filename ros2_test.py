import time
from threading import Thread, Event

from absl import app
from absl import flags
from absl import logging
import envlogger
from envlogger.backends import tfds_backend_writer

from envlogger.testing import catch_env
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float32  # Replace with the appropriate message type

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
import dm_env
from dm_env import specs
import numpy as np
import random
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import JointState
# from tf2_msgs.msg import TFMessage
from moveit_msgs.srv import GetPositionFK
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point

from data_collection.online.openvla_env import OpenVlaEnv
from data_collection.online.openvla_env import CHECK_ORDER

FLAGS = flags.FLAGS


flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes to log.')
flags.DEFINE_string('trajectories_dir', '/home/koyo/openvla/data_collection/datasets/test',
                    'Path in a filesystem to record trajectories.')

class Gen3LiteNode(Node):
    def __init__(self):
        super().__init__("record_data_node") #node name

        cb_group_1 = MutuallyExclusiveCallbackGroup()
        cb_group_2 = MutuallyExclusiveCallbackGroup()
        timer_cb_group = MutuallyExclusiveCallbackGroup()


        # Create subscription
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_callback, 10, callback_group=cb_group_1
        )
        # self.tf_sub = self.node.create_subscription(
        #     TFMessage, "/tf", self.tf_callback, 10
        # )

        self.fk_client = self.create_client(GetPositionFK, '/compute_fk', callback_group=cb_group_2)
        while not self.fk_client.wait_for_service(timeout_sec=1.0):
            print('Service not available, waiting again...')
        if self.fk_client.service_is_ready():
           print("service is ready")
        #    breakpoint()

        self.timer = self.create_timer(0.001, self._timer_cb, callback_group=cb_group_1)
        
        # self.cap = cv2.VideoCapture(6)  # Open the default camera  #v4l2-ctl --list-devices
        # self.bridge = CvBridge()
        resized_dummy_image = np.full((256, 256), 150, dtype=np.uint8) 
        self.image = resized_dummy_image

        self.instructions = ["pick up a banana", "get a banana"] # Use self.done to judge if it should be changed

        self.joint_positions = None
        self.ee_position = Point()

        # Initialize environment state
        self.current_observation = None
        self.current_reward = 0.0
        self.done = False

        self.start_joint = False
        self.start_action = False
        self.joint_received = False

    def start_obs(self):
        if CHECK_ORDER:
          if self.start_action:
            print("8")
          else:
            print("1")

        self.start_joint = True

    def joint_callback(self, msg):
        print("joint cb")
        if not self.start_joint:
            return

        
        self.start_joint = False

        if CHECK_ORDER:
          if self.start_action:
            print("9")
          else:
            print("2")

        joint_names = msg.name
        joint_positions = np.array(msg.position)
        # d: double-precision floating-point numbers (8 bytes per element), matching the float64 data type.
        
        # #print(f"Received joint names: {joint_names}")
        #print(f"Received joint positions: {joint_positions}") 
        
        # Store in observation
        self.current_observation = {
            "joint_positions": joint_positions,
        }

        # print("Joint names:", " ".join(joint_names))
        # print("Joint positions:", " ".join(map(str, joint_positions)))

        # # Desired order of joints without "right_finger_bottom_joint"
        # desired_order = [
        #     "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
        # ]
        # Desired order of joints with "right_finger_bottom_joint"
        desired_order = [
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"
        ]

        # Create a map to store positions indexed by joint names
        name_position_map = dict(zip(joint_names, joint_positions))

        # Reorder joint_names and joint_positions based on desired_order
        reordered_positions = []
        reordered_names = []
        for name in desired_order:
            if name in name_position_map:
                reordered_positions.append(name_position_map[name])
                reordered_names.append(name)

        # Update joint_names and joint_positions
        self.joint_names = reordered_names
        self.joint_positions = reordered_positions

        # print("Reordered joint names:", " ".join(self.joint_names))
        # print("Reordered joint positions:", " ".join(map(str, self.joint_positions)))

        self.joint_received = True

        # self.get_end_effecter_position(joint_positions, joint_names)

    def _timer_cb(self):
        print("timer_cb")
        if not self.joint_received:
          return
        

        self.joint_received = False
    
        self.get_end_effecter_position(self.joint_positions, self.joint_names)

    def get_end_effecter_position(self, joint_positions, joint_names):
        if CHECK_ORDER:
          if self.start_action:
            print("10")
          else:
            print("3")
        #print("Checking if /compute_fk service is available...")


        #print("Preparing /compute_fk service request...")
        fk_request = GetPositionFK.Request()

        fk_request.header.frame_id = 'world'
        # fk_request.fk_link_names = ['right_finger_prox_link']
        fk_request.fk_link_names = ['end_effector_link']
        # fk_request.robot_state.joint_state = joint_states
        fk_request.robot_state.joint_state.name = ["joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7"]
        fk_request.robot_state.joint_state.position = [0.1, -0.3, 0.12, 0.1, 0.5, 0.2, 0.1]

        #print("Sending request to /compute_fk service...")
        fk_future = self.fk_client.call_async(fk_request)
        rclpy.spin_until_future_complete(self, fk_future)
        #print("Waiting for /compute_fk response...")

        # while not self.fk_client.wait_for_service(timeout_sec=3.0):
        #     print('Service not available, waiting again...')

        try:
            # time.sleep(0.1)
            response = fk_future.result()
            # logging.info(f'FK response: {response}')
            #print("response ", response)
            # ee_pos_x = response.pose_stamped[0].pose.position.x
            # ee_pos_y = response.pose_stamped[0].pose.position.y
            # ee_pos_x = response.pose_stamped[0].pose.position.z
            # ee_ori_x = response.pose_stamped[0].pose.orientation.x
            #print(f"End effector position: x={ee_position.x}, y={ee_position.y}, z={ee_position.z}")
        except Exception as e:
            # print('Service call failed')
            pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w = random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)
            ee_position_dummy = Point()
            ee_position_dummy.x = pos_x
            ee_position_dummy.y = pos_y
            ee_position_dummy.z = pos_z
            # ee_position_dummy.orientation.x = rot_x
            # ee_position_dummy.orientation.y = rot_y
            # ee_position_dummy.orientation.z = rot_z
            # ee_position_dummy.orientation.w = rot_w
            #print("ef position dummy ", ee_position_dummy)
            self.ee_position = ee_position_dummy



def main(args=None):

    rclpy.init()
    gen3_lite_node =Gen3LiteNode()
    executor = MultiThreadedExecutor()
    executor.add_node(gen3_lite_node)       # Keeps the node running, processing incoming messages
    executor.spin()
    rclpy.shutdown()


if __name__ == '__main__':
    main()