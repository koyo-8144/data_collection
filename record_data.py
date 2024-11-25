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

from openvla_env import OpenVlaEnv
from openvla_env import CHECK_ORDER

FLAGS = flags.FLAGS


flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes to log.')
flags.DEFINE_string('trajectories_dir', '/home/koyo/openvla/data_collection/datasets/test',
                    'Path in a filesystem to record trajectories.')

class Gen3LiteNode(Node):
    def __init__(self):
        super().__init__("record_data_node") #node name

        cb_group_1 = MutuallyExclusiveCallbackGroup()
        cb_group_2 = MutuallyExclusiveCallbackGroup()


        # Create subscription
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_callback, 10, callback_group=cb_group_1
        )
        # self.tf_sub = self.node.create_subscription(
        #     TFMessage, "/tf", self.tf_callback, 10
        # )

        self.fk_client = self.create_client(GetPositionFK, '/compute_fk', callback_group=cb_group_2)
        # while not self.fk_client.wait_for_service(timeout_sec=1.0):
        #     #print('Service not available, waiting again...')
        
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

    def start_obs(self):
        if CHECK_ORDER:
          if self.start_action:
            print("8")
          else:
            print("1")

        self.start_joint = True

    def joint_callback(self, msg):
        if not self.start_joint:
            return
        
        self.start_joint = False

        if CHECK_ORDER:
          if self.start_action:
            print("9")
          else:
            print("2")

        joint_states = msg
        joint_names = msg.name
        joint_positions = np.array(msg.position) 
        self.joint_positions = joint_positions
        # d: double-precision floating-point numbers (8 bytes per element), matching the float64 data type.
        
        # #print(f"Received joint names: {joint_names}")
        #print(f"Received joint positions: {joint_positions}") 
        
        # Store in observation
        self.current_observation = {
            "joint_positions": joint_positions,
        }

        joint_states_dummy = JointState()
        joint_states_dummy.position = [0.1, -0.3, 0.12, 0.1, 0.5, 0.2]

        self.get_end_effecter_position(joint_states)

    def get_end_effecter_position(self, joint_states):
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
        # rclpy.spin_until_future_complete(self, fk_future)
        #print("Waiting for /compute_fk response...")

        # while not self.fk_client.wait_for_service(timeout_sec=3.0):
        #     print('Service not available, waiting again...')

        try:
            time.sleep(0.1)
            response = fk_future.result()
            # logging.info(f'FK response: {response}')
            #print("response ", response)
            ee_position_x = response.pose_stamped[0].pose.position.x
            ee_position_y = response.pose_stamped[0].pose.position.y
            ee_position_x = response.pose_stamped[0].pose.position.z
            #print(f"End effector position: x={ee_position.x}, y={ee_position.y}, z={ee_position.z}")
        except Exception as e:
            print('Service call failed')
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

        # # Wait for response
        # response = fk_future.result()
        # if response.error_code.val == 1:  # SUCCESS
        #     self.ee_position = [
        #         response.pose_stamped[0].pose.position.x,
        #         response.pose_stamped[0].pose.position.y,
        #         response.pose_stamped[0].pose.position.z,
        #         response.pose_stamped[0].pose.orientation.x,
        #         response.pose_stamped[0].pose.orientation.y,
        #         response.pose_stamped[0].pose.orientation.z,
        #         response.pose_stamped[0].pose.orientation.w
        #     ]
        # else:
        #     print('Service call failed')
        #     pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w = random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)
        #     ee_position_dummy = Point()
        #     ee_position_dummy.pose_stamped[0].pose.position.x = pos_x
        #     ee_position_dummy.pose_stamped[0].pose.position.y = pos_y
        #     ee_position_dummy.pose_stamped[0].pose.position.z = pos_z
        #     ee_position_dummy.pose_stamped[0].pose.orientation.x = rot_x
        #     ee_position_dummy.pose_stamped[0].pose.orientation.y = rot_y
        #     ee_position_dummy.pose_stamped[0].pose.orientation.z = rot_z
        #     ee_position_dummy.pose_stamped[0].pose.orientation.w = rot_w
        #     #print("ef position dummy ", ee_position_dummy)
        #     self.ee_position = ee_position_dummy

        
        self.get_camera_image()


    # Constantly get image data, same as inference
    def get_camera_image(self):
        if CHECK_ORDER:
          if self.start_action:
            print("11")
          else:
            print("4")

        # # Capture a frame from the camera
        # ret, frame = self.cap.read() #frame, (1080, 1920)
        # if ret:
        #     # Convert the frame to the expected format if necessary
        #     cropped_image = frame[0:1080,400:1920]
        #     resized_image = cv2.resize(cropped_image, (256, 256))
        #     self.image = resized_image
        #     # #print("self.image ", self.image.shape[:2])
        #     # Render the image in a window
        #     # cv2.imshow("Cropped Image", cropped_image)
        #     cv2.imshow("Cropped Image", resized_image)
        #     # WaitKey allows image rendering and checks if 'q' was pressed to quit
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         rclpy.shutdown()
        # else:
        #     print("Failed to capture image")


        resized_dummy_image = np.full((256, 256), 150, dtype=np.uint8) 
        self.image = resized_dummy_image
        

    
    def get_obs(self):
        if CHECK_ORDER:
          print("0")
        self.start_obs()
        time.sleep(0.01)
        if CHECK_ORDER:
          print("5")

        joint_pos = self.joint_positions
        self.ef_pos_before = self.ee_position
        image = self.image

        # Combine the observations into one structure
        obs = {
            "joint_positions": joint_pos,
            "end_effector_position": {
                "x": self.ef_pos_before.x,
                "y": self.ef_pos_before.y,
                "z": self.ef_pos_before.z,
            },
            "image": image,  # Image as a NumPy array
        }

        return obs
    
    def get_action(self):
        if CHECK_ORDER:
          print("7")
        self.start_action = True
        self.start_obs()
        time.sleep(0.01)
        self.start_action = False
        if CHECK_ORDER:
          print("12")

        joint_pos = self.joint_positions
        ef_pos_after = self.ee_position
        image = self.image

        # delta_pos = ef_pos_after - self.ef_pos_before
        delta_pos = joint_pos

        return delta_pos
        
    

def record_data(unused_argv):
    logging.info('Creating Catch environment...')
    env = OpenVlaEnv()
    logging.info('Done creating Catch environment.')

    # Initialize ROS 2
    rclpy.init()
    gen3_lite_node = Gen3LiteNode()
    executor = MultiThreadedExecutor()
    executor.add_node(gen3_lite_node)

    # Run the ROS 2 executor in a separate thread
    ros_thread = Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    try:
        def step_fn(unused_timestep, unused_action, unused_env):
            return {'timestamp': time.time()}
        

 
        action_info = tfds.features.FeaturesDict({
            'translation_delta': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
            'rotation_delta': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
            'gripper_delta': tfds.features.Tensor(shape=(1,), dtype=tf.float32),
        })

        observation_info = tfds.features.FeaturesDict({
            'image': tfds.features.Tensor(shape=(256, 256, 3), dtype=tf.uint8, encoding=tfds.features.Encoding.ZLIB),
            'natural_language_instruction': tfds.features.Tensor(shape=(), dtype=tf.string),
            'end_effector_position': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
            'joint_position': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
            # 'natural_language_embedding': tfds.features.Tensor(shape=(512,), dtype=tf.float32),
        })


        dataset_config = tfds.rlds.rlds_base.DatasetConfig(
            name='test',
            observation_info=observation_info,
            action_info=action_info,
            reward_info=tfds.features.Tensor(shape=(), dtype=tf.float32),
            discount_info=tf.float64,
            step_metadata_info={'timestamp': tf.float32})

        logging.info('Wrapping environment with EnvironmentLogger...')
        with envlogger.EnvLogger(
            env,
            step_fn=step_fn,
            backend=tfds_backend_writer.TFDSBackendWriter(
                data_directory=FLAGS.trajectories_dir,
                split_name='train',
                max_episodes_per_file=500,
                ds_config=dataset_config),
        ) as env:
            logging.info('Done wrapping environment with EnvironmentLogger.')

            logging.info('Training a random agent for %r episodes...',
                         FLAGS.num_episodes)

            for i in range(FLAGS.num_episodes):  # Episode iteration
                # logging.info('episode %r', i)
                timestep = env.reset()
                j = 0

                while not timestep.last():  # Step iteration
                    j += 1
                    # logging.info('step %r', j)

                    # Access the most recent value from the ROS 2 topic
                    obs = gen3_lite_node.get_obs()
                    env.update_observation(obs)

                    # if gen3_lite_node.joint_positions is not None:
                    #     # action = int(gen3_lite_node.current_value)
                    #     action = np.random.randint(low=0, high=3)
                    #     print("Goood ", gen3_lite_node.joint_positions)
                    # else:
                    #     action = np.random.randint(low=0, high=3)
                    #     print("Nooo ", gen3_lite_node.joint_positions)

                    # action = np.random.randint(low=0, high=3)
                    action = gen3_lite_node.get_action()
                    timestep = env.step(action)

            logging.info('Done training a random agent for %r episodes.',
                         FLAGS.num_episodes)
    finally:
        # Shutdown ROS 2
        gen3_lite_node.destroy_node()
        rclpy.shutdown()
        ros_thread.join()

if __name__ == '__main__':
    app.run(record_data)
