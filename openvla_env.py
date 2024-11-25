import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
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

CHECK_ORDER = False


class OpenVlaEnv(dm_env.Environment):
    def __init__(self):
        
        # self.cap = cv2.VideoCapture(6)  # Open the default camera  #v4l2-ctl --list-devices
        # self.bridge = CvBridge()
        resized_dummy_image = np.full((256, 256), 150, dtype=np.uint8) 
        self.image = resized_dummy_image

        self.instructions = ["pick up a banana", "get a banana"]
        self.instruction = random.choice(self.instructions)

        self.ed_position = None

        self.joint_position = None


    ### We need the following required abstract methods in dm_env.Environment


    def reset(self):
        # Reset the environment state
        #print("Environment reset")
        self.current_observation = None
        self.current_reward = 0.0
        self.done = False
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=None,
            discount=None,
            observation=self.current_observation,
        )

    def step(self, action): # delta motion
        if CHECK_ORDER:
            print("13")

        # Apply the action and update the environment state
        #print(f"Action taken: {action}")
        self.current_reward = 1.0 # or 0.0
        self.done = self.current_reward > 10  # We need to think about it, like a switch

        # self.current_observation = 

        # Change self.instruction at the end of episode using a LLM
        # Make a list of instructions and pick one of them
        if self.done:
            self.instruction = random.choice(self.instructions)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.LAST if self.done else dm_env.StepType.MID,
            reward=self.current_reward,
            discount=1.0,
            observation=self.current_observation,
        )

    def action_spec(self):
        return {
            "translation_delta": specs.BoundedArray(
                shape=(3,), dtype=np.float32, minimum=-np.inf, maximum=np.inf, name="translation_delta",
            ),
            "rotation_delta": specs.BoundedArray(
                shape=(3,), dtype=np.float32, minimum=-np.inf, maximum=np.inf, name="rotation_delta",
            ),
            "gripper_delta": specs.BoundedArray(
                shape=(1,), dtype=np.float32, minimum=-np.inf, maximum=np.inf, name="gripper_delta",
            ),
        }
    

    def observation_spec(self):
        return {
            "image": specs.BoundedArray(
                shape=(256, 256, 3),
                dtype=np.uint8,
                minimum=0,
                maximum=255,
                name="image",
            ),
            "natural_language_instruction": specs.Array(
                shape=(), dtype=np.str_, name="natural_language_instruction"
            ),
            "end_effector_position": specs.BoundedArray(
                shape=(7,),
                dtype=np.float32,
                minimum=-np.inf,  # Replace with specific bounds if known
                maximum=np.inf,
                name="end_effector_position",
            ),
            "joint_position": specs.BoundedArray(
                shape=(7,),
                dtype=np.float32,
                minimum=-np.inf,  # Replace with specific bounds if known
                maximum=np.inf,
                name="joint_position",
            ),
        }

    
    # Get self.image, self.instruction and self.ef_position at the same frequency?
    def _observation(self):
        pass

    # def get_action(self):

    def update_observation(self, obs):
        if CHECK_ORDER:
            print("6")
        self.current_observation = obs
        

