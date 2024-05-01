import math
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

from robot_components import Link, Joint
import general_utility as general
from motor import Motor
from foc import FOC

def trapazoid_calc(prev_value: float, prev_der: float, new_der: float,
                   timestep: float) -> float:
  """Calculate a forward propogated step using the trapazoidal rule 

  Args:
      prev_value (float): previous value (vel or position)
      prev_der (float): Previous derivative (accel or vel)
      new_der (float): New derivative (accel or vel)
      timestep (float): timestep to calculate over

  Returns:
      float: new value (vel or position)
  """
  return prev_value + 0.5 * (prev_der + new_der) * timestep


class Workspace(object):
  def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float,
               z_min: float, z_max: float):
    """Initialize the workspace 

    Args:
        x_min (float): min x of the worksapce in mm
        x_max (float): max x of the workspace in mm 
        y_min (float): min y of the workspace in mm
        y_max (float): max y of the workspace in mm
        z_min (float): min z of the workspace in mm
        z_max (float): max z of the workspace in mm
    """
    self.x_min = x_min
    self.x_max = x_max
    self.y_min = y_min
    self.y_max = y_max
    self.z_min = z_min
    self.z_max = z_max


class JointState(object):
  """Class to hold the current joint state if you want to use it."""
  @property
  def pos(self):
    """Return the joint position (angular)

    Returns:
        float: angular position of the joint
    """
    return self._pos

  @pos.setter
  def pos(self, value):
    """Set the position of the joint

    Args:
        value (float): angular position to set 
    """
    self._pos = value

  @property
  def vel(self):
    """Velocity of the joint

    Returns:
        float: angular velocity of the joint 
    """
    return self._vel

  @vel.setter
  def vel(self, value):
    """Set the angular velocity of the joint

    Args:
        value (float): Angular velocity of the joint to set
    """
    self._vel = value

  @property
  def accel(self):
    """Get the angular acceleration of the joint

    Returns:
        float: angular acceleration of the joint. 
    """
    return self._accel

  @accel.setter
  def accel(self, value):
    """Set the angular acceleration of the joint

    Args:
        value (float): angular acceleration to set 
    """
    self._accel = value

  def __init__(self,
               position: float = 0,
               velocity: float = 0,
               acceleration: float = 0) -> None:
    """Create the joint class 

    Args:
        position (float, optional): angular position of the joint. Defaults to 0.
        velocity (float, optional): angular velocity of the joint. Defaults to 0.
        acceleration (float, optional): angular acceleration of the joint. Defaults to 0.
    """
    self._pos = position
    self._vel = velocity
    self._accel = acceleration

  def __repr__(self):
    """Allows easier printing of the state of the joint."""
    return ("pos = {}\nvel={}\naccel = {}".format(self.pos, self.vel,
                                                  self.accel))


class RRBot(object):
  """RR Bot class! """
  @property
  def joints(self) -> List[Joint]:
    """Return the list of joints 1, 2, and end effector

    Returns:
        List[Joint]: list of joints 
    """
    return [self._joint_1, self._joint_2, self._joint_ee]

  @property
  def ee_frame(self) -> np.ndarray:
    """Return the end effector frame 

    Returns:
        np.ndarray: 4x4 Transformation matrix of the end effector 
    """
    ee_frame = self._joint_1.dh_transform @ self._joint_2.dh_transform
    return ee_frame

  @property
  def tool_tip(self) -> np.ndarray:
    """Return the tool tip location

    Returns:
        np.ndarray: 4x4 transformation of the tool tip 
    """
    tool_tip = self.ee_frame @ self.tool_frame
    return tool_tip

  def __init__(self):
    """Initialize the class"""

    # Base info for class function
    self._drawn_once = False
    self._frame_size = 0.1
    self.colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    self._m_1 = 1  #kg
    self._m_2 = 5  #kg
    self._m_r1 = 2.3  #kg
    self._m_r2 = 2.3  #kg
    self._l_1 = 1.0  #m
    self._l_2 = 1.41  #m
    self.g = 9.81  #m/s^2
    self.tool_frame = np.eye(4)
    self.tool_frame[3, 0] = self._l_2
    self._kp1 = 10
    self._kv1 = 20
    self._kp2 = 1000
    self._kv2 = 100

    self._dt = 0.01
    self._duration = 10.0

    self.workspace = Workspace(-(self._l_1 + self._l_2),
                               (self._l_1 + self._l_2),
                               -(self._l_1 + self._l_2),
                               (self._l_1 + self._l_2), -self._l_2, self._l_2)

    self._create_plot()

    self._joints = []
    self._joint_1 = None
    self._joint_2 = None
    self._joint_ee = None
    self._setup_joints()
    self._joint_1_des = 0.0
    self._joint_2_des = 0.0

    self._links = [Link(self.ax, self.colors[index]) for index in range(3)]

    self._base_frame = Joint(self.ax)
    self._base_frame.set_final_transform(np.eye(4))
    self._base_frame.set_dh_transform()

  def set_initial_conditions(self, joint_1_pos: float, joint_1_vel: float,
                             joint_2_pos: float, joint_2_vel: float) -> None:
    """Set the initial conditions of the robot

    Args:
        joint_1_pos (float): joint 1 starting position
        joint_1_vel (float): joint 1 starting velocity
        joint_2_pos (float): joint 2 starting position
        joint_2_vel (float): joint 2 starting velocity 
    """
    self._joint_1.pos = joint_1_pos
    self._joint_1.vel = joint_1_vel
    self._joint_2.pos = joint_2_pos
    self._joint_2.vel = joint_2_vel

  def set_desired_positions(self, joint_1_pos: float,
                            joint_2_pos: float) -> None:
    """Set the desired position of the robot

    Args:
        joint_1_pos (float): joint 1 desired position 
        joint_2_pos (float): joint 2 desired position 
    """
    self._joint_1_des = joint_1_pos
    self._joint_2_des = joint_2_pos

  def calculate_fk(self, joint_angles: np.ndarray) -> None:
    """Calculate the forward kinematics
    This is taken care of for you under the hood.

    Args:
        joint_angles (np.ndarray): 1x2 of the joint angles 
    """
    general.check_proper_numpy_format(joint_angles, (2, ))

    self._joint_1.pos = joint_angles[0]
    self._joint_2.pos = joint_angles[1]

  def _create_plot(self) -> None:
    """Initialize the plot to use throughout"""
    self.fig = plt.figure(figsize=(8, 8), facecolor='w')
    self.ax = self.fig.add_subplot(111, projection='3d')
    plt.xlim([self.workspace.x_min, self.workspace.x_max])
    plt.ylim([self.workspace.y_min, self.workspace.y_max])
    self.ax.set_zlim([self.workspace.z_min, self.workspace.z_max])
    self.ax.set_xlabel('X (mm)', fontsize=16)
    self.ax.set_ylabel('Y (mm)', fontsize=16)
    self.ax.set_zlabel('Z (mm)', fontsize=16)
    # self.ax.view_init(elev=22.8, azim=147.3)
    plt.grid(True)

  def _setup_joints(self) -> None:
    """Setup the joints -- this is done for you"""
    self._joint_1 = Joint(self.ax, frame_size=self._frame_size)
    self._joint_1.set_joint_limits(-math.inf, math.inf)
    self._joint_1.set_dh_parameters(0, 0, 0, 0)

    self._joint_2 = Joint(self.ax, frame_size=self._frame_size)
    self._joint_2.set_joint_limits(-math.inf, math.inf)
    self._joint_2.set_dh_parameters(self._l_1, math.pi / 2, 0, 0)

    self._joint_ee = Joint(self.ax, frame_size=self._frame_size)
    self._joint_ee.set_joint_limits(-math.inf, math.inf)
    self._joint_ee.set_dh_parameters(self._l_2, 0, 0, 0)

    for joint in self.joints:
      joint.draw()

  def draw_rr(self, joint_angles: np.ndarray) -> None:
    """Draw the robot given a set of joint angles 

    Args:
        joint_angles (np.ndarray): 1x2 array of joint angles
    """
    general.check_proper_numpy_format(joint_angles, (2, ))

    self.calculate_fk(joint_angles)
    if not self._drawn_once:
      plt.show(block=False)
      plt.pause(0.5)
      self._drawn_once = True

    self._base_frame.draw()
    current_transform = self._base_frame.dh_transform

    for index, joint in enumerate(self.joints):
      prev_transform = current_transform
      current_transform = current_transform @ joint.dh_transform
      joint.set_final_transform(current_transform)
      joint.draw()
      self._links[index].update_frames(prev_transform, current_transform)
      self._links[index].draw()

    plt.pause(0.001)

  def calculate_energy(self) -> float:
    """
    Calculates the kinetic and potential energy of the robot at a given point in time
    """
    # calculate the kinetic energy first
    pass

  def simulate_rr(self) -> None:
    """Simulate the RR robot

    TODO 
    This is the function to simulate the RR robot. Using the physical parameters above 
    You should calculate the dynamics of the robot and then simulate the robot over the 
    course of 10 seconds. 
    In the first pass, you will implement it with just the dynamics, and no tau. In the 
    second pass, you will add a simple controller to the system. 

    The general process it to calculate your M, C, and G arrays and with your tau, calculate
    your theta_accel and use that to do trapezoidal calculation to propogte states 

    You can simply call draw_rr([joint1, joint2]) to simulate and calculating the total energy
    is a good way to test. 
    Note: when simulating, due to rounding, the energy may not be perfectly zero, but it should be 
    pretty close, the smaller the increment you simulate, the closer to zero it should be. 
    """
    # Setup variables for reference
    grav = 9.81
    m1 = self._m_1 + self._m_r1
    m2 = self._m_2 + self._m_r2
    lc1 = ((self._m_1 + (0.5 * self._m_r1)) * self._l_1) / m1
    lc2 = ((self._m_2 + (0.5 * self._m_r2)) * self._l_2) / m2
    i1 = ((1/12) * self._m_r1 * (self._l_1**2)) + (self._m_1 * ((self._l_1/2)**2))
    i2 = ((1/12) * self._m_r2 * (self._l_2**2)) + (self._m_2 * ((self._l_2/2)**2))

    self.set_initial_conditions(0, 0, -np.pi/2, 0)
    self.set_desired_positions(0, -np.pi/4) # do I need to set these?

    index = 0
    curr_joint_states = [
        JointState(self._joint_1.pos, self._joint_1.vel, 0),
        JointState(self._joint_2.pos, self._joint_2.vel, 0)
    ]
    timesteps = np.arange(0, self._duration, self._dt)

    time_series_theta_1 = []
    time_series_theta_2 = []

    for t in timesteps:
      prev_joint_states = deepcopy(curr_joint_states)

      # collect the variables we will need
      theta_1 = prev_joint_states[0].pos
      theta_2 = prev_joint_states[1].pos
      theta_1_dot = prev_joint_states[0].vel
      theta_2_dot = prev_joint_states[1].vel

      #yapf: disable
      M = np.array([[(m1 * (lc1**2)) + (m2 * ((self._l_1 + (lc2 * np.cos(theta_2)))**2)) + i1 + i2, 0],
                    [0, (m2 * (lc2**2)) + i2]])

      C = np.array([[-2 * m2 * lc2 * np.sin(theta_2) * (self._l_1 + (lc2 * np.cos(theta_2))) * theta_1_dot * theta_2_dot],
                    [m2 * lc2 * np.sin(theta_2) * (self._l_1 + (lc2 * np.cos(theta_2))) * (theta_1_dot**2)]])

      G = np.array([[0],
                    [m2 * grav * lc2 * np.cos(theta_2)]])
      #yapf: enable

      # find the torque vector to apply to the robot. gravity is included on the second joint
      target_torque = np.array([[-self._kp1 * (theta_1 - self._joint_1_des) - self._kv1 * theta_1_dot],
                      [-self._kp2 * (theta_2 - self._joint_2_des) - self._kv2 * theta_2_dot - G[1, 0]]])
      
      # use the motor and FOC objects to return the desired torque
      # TODO turn target_speed into target_torque ()
      # speed1, pos1, new_i_q1, new_i_d1 = foc1.control_loop(motor1, target_torque[0, 0], i_q_prev1, i_d_prev1, pos1, speed1)
      # actual_torque1 = motor1.output_torque()

      # speed2, pos2, new_i_q2, new_i_d2 = foc2.control_loop(motor2, target_torque[1, 0], i_q_prev2, i_d_prev2, pos2, speed2)
      # actual_torque2 = motor2.output_torque()

      # actual_torque = np.array([[actual_torque1],
      #                           [actual_torque2]])
      
      # calculate theta 1 double dot and theta 2 double dot using the equation tau = M(THETA..) + C + G
      t_double_dot = np.linalg.inv(M) @ (target_torque - C - G)

      # Trapezoidal calc 
      if index >= 0:
        
        # Note, I provided the function trapazoid_calc() to help you out. 

        self._joint_1.vel = trapazoid_calc(prev_joint_states[0].vel, prev_joint_states[0].accel, t_double_dot[0, 0], self._dt)
        self._joint_2.vel = trapazoid_calc(prev_joint_states[1].vel, prev_joint_states[1].accel, t_double_dot[1, 0], self._dt) 
        self._joint_1.pos = trapazoid_calc(prev_joint_states[0].pos, prev_joint_states[0].vel, self._joint_1.vel, self._dt)
        self._joint_2.pos = trapazoid_calc(prev_joint_states[1].pos, prev_joint_states[1].vel, self._joint_2.vel, self._dt)

        self.draw_rr(np.array([self._joint_1.pos, self._joint_2.pos]))

        time_series_theta_1.append(self._joint_1.pos)
        time_series_theta_2.append(self._joint_2.pos)

        curr_joint_states = [
            JointState(self._joint_1.pos, self._joint_1.vel, t_double_dot[0, 0]), 
            JointState(self._joint_2.pos, self._joint_2.vel, t_double_dot[1, 0])]

      index += 1

    # plot the values of theta 1 and theta 2 over time
    fig, ax = plt.subplots()
    ax.plot(timesteps, time_series_theta_1, label='Theta 1')
    ax.plot(timesteps, time_series_theta_2, label='Theta 2')
    plt.legend()
    plt.show()
