import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float32  # Replace with the appropriate message type
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
import numpy as np
import random
import logging

from sensor_msgs.msg import JointState
# from tf2_msgs.msg import TFMessage
from moveit_msgs.srv import GetPositionFK
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point

CHECK_ORDER = False


class Gen3LiteClientNode(Node):
    def __init__(self):
        super().__init__("record_ros2_data_node") #node name

        cb_group = MutuallyExclusiveCallbackGroup()
        client_cb_group = MutuallyExclusiveCallbackGroup()


        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_callback, 10, callback_group=cb_group
        )

        self.ee_states_pub = self.create_publisher(
            Pose, "/ee_states", 10
        )

        self.fk_client = self.create_client(GetPositionFK, '/compute_fk', callback_group=client_cb_group)
        while not self.fk_client.wait_for_service(timeout_sec=1.0):
            print('Service not available, waiting again...')
        if self.fk_client.service_is_ready():
           print("service is ready")
        #    breakpoint()

        self._timer = self.create_timer(0.01, self._timer_callback, callback_group=cb_group)
        
        resized_dummy_image = np.full((256, 256), 150, dtype=np.uint8) 
        self.image = resized_dummy_image

        self.instructions = ["pick up a banana", "get a banana"] # Use self.done to judge if it should be changed

        self.joint_positions = None
        self.joint_names = None
        self.ee_states = Pose()

        self.joint_received = False

        # # Desired order of joints without "right_finger_bottom_joint"
        # self.desired_order = [
        #     "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
        # ]
        self.desired_order = [
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"
        ]

    # There is no problem with executing jonit_callback in a row as long as _timer_callback is executed after joint_callback
    def joint_callback(self, msg):
        
        print("joint_callback")
        self.joint_received = True # gate for fk_timer_callback

        joint_names = msg.name
        joint_positions = np.array(msg.position)
        
        # #print(f"Received joint names: {joint_names}")
        #print(f"Received joint positions: {joint_positions}") 

        # Create a map to store positions indexed by joint names
        name_position_map = dict(zip(joint_names, joint_positions))

        # Reorder joint_names and joint_positions based on self.desired_order
        reordered_positions = []
        reordered_names = []
        for name in self.desired_order:
            if name in name_position_map:
                reordered_positions.append(name_position_map[name])
                reordered_names.append(name)

        # Update joint_names and joint_positions
        self.joint_names = reordered_names
        self.joint_positions = reordered_positions

        # print("Reordered joint names:", " ".join(self.joint_names))
        # print("Reordered joint positions:", " ".join(map(str, self.joint_positions)))

        ##---> _timer_callback



    def _timer_callback(self):
        if not self.joint_received:
            return

        print("_timer_callback")
    
        self.get_pub_ee_states(self.joint_positions, self.joint_names)
        


    def get_pub_ee_states(self, joint_positions, joint_names):
        #print("Preparing /compute_fk service request...")
        fk_request = GetPositionFK.Request()

        fk_request.header.frame_id = 'world'
        # fk_request.fk_link_names = ['right_finger_prox_link']
        fk_request.fk_link_names = ['end_effector_link']
        # fk_request.robot_state.joint_state = joint_states
        fk_request.robot_state.joint_state.name = joint_names
        fk_request.robot_state.joint_state.position = joint_positions
        #print("Sending request to /compute_fk service...")
        fk_future = self.fk_client.call(fk_request)
        # fk_future = self.fk_client.call_async(fk_request)
        # rclpy.spin_until_future_complete(self, fk_future)
        # #print("Waiting for /compute_fk response...")

        while not self.fk_client.wait_for_service(timeout_sec=5.0):
            print('Service not available, waiting again...')

        try:
            # time.sleep(0.1)
            response = fk_future.pose_stamped[0].pose
            print(f'FK response: {response}')
            #print("response ", response)
            self.ee_states = response
            #print(f"End effector position: x={ee_position.x}, y={ee_position.y}, z={ee_position.z}")
        except Exception as e:
            logging.error(str(e))
            pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w = random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)
            ee_states_dummy = Pose()
            ee_states_dummy.position.x = pos_x
            ee_states_dummy.position.y = pos_y
            ee_states_dummy.position.z = pos_z
            ee_states_dummy.orientation.x = rot_x
            ee_states_dummy.orientation.y = rot_y
            ee_states_dummy.orientation.z = rot_z
            ee_states_dummy.orientation.w = rot_w
            self.ee_states = ee_states_dummy
        
    
        # Publish the end-effector position
        self.ee_states_pub.publish(self.ee_states)
        # self.get_logger().info(f"Published EE states: {self.ee_states}")
        


def main(args=None):

    rclpy.init()
    gen3_lite_client_node =Gen3LiteClientNode()
    executor = MultiThreadedExecutor()
    executor.add_node(gen3_lite_client_node)       # Keeps the node running, processing incoming messages
    executor.spin()
    rclpy.shutdown()


if __name__ == '__main__':
    main()




"""
callback group selection:

For the interaction of an individual callback with "itself":

1. if "it" should be executed in parallel to itself. -> Register "it" to a Reentrant Callback Group
   An example case could be an action/service "server" that needs to be able to process several action calls in parallel to each other.

2. if "it" should never be executed in parallel to itself. -> Register "it" to a Mutually Exclusive Callback Group
   An example case could be a timer callback that runs a control loop that publishes control commands.


In my case,
    joint_callback -> 2
    self.client, fk_timer_callback -> 2
    publish_ee_states -> 2
    


For the interaction of different callbacks with "each other":

1. if "they" should never be executed in parallel. -> Register them to the same Mutually Exclusive Callback Group
   An example case could be that the callbacks are accessing shared critical and non-thread-safe resources.

2. if "they" should be executed in parallel.
    1. Register them to different Mutually Exclusive Callback Groups (no overlap of the individual callbacks)
    2. Register them to a Reentrant Callback Group (overlap of the individual callbacks)

    An example case of running different callbacks in parallel is a Node that has a synchronous service client and a timer calling this service.

In my case,
    I get joint states in joint_callback, use the joint states to compute ee states in fk_timer_callback and publish the ee states -> they should never be executed in parallel
"""