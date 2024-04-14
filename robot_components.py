import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D

import general_utility as general
from drawing_helper import FrameDrawing, LinkDrawing, PointDrawing


def dh_tf(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
  """Create a DH transform using alpha, a, d, and theta
  Args:
      alpha (float): alpha angle in radians
      a (float): a in mm
      d (float): d in mm
      theta (float): theta in radians

  Returns:
      np.ndarray: (4,4) numpy array of the transformation matrix
  """
  c_theta = math.cos(theta)
  s_theta = math.sin(theta)
  c_alpha = math.cos(alpha)
  s_alpha = math.sin(alpha)

  T = np.array([[c_theta, -s_theta, 0, a],
                [s_theta * c_alpha, c_theta * c_alpha, -s_alpha, -s_alpha * d],
                [s_theta * s_alpha, c_theta * s_alpha, c_alpha, c_alpha * d],
                [0, 0, 0, 1]])

  return T


class Link(object):
  """Class to hold information about and for a link"""
  def __init__(self, ax: Axes3D, color=None):
    """Create a Link 
    Links contain LinkDrawings and act a pass through and information holders

    Args:
        ax (Axes3D): Axes3D to use to draw
        color (_type_, optional): Color to draw if rgb frame is not desired. Defaults to None.
    """
    self._frame_1 = np.eye(4)
    self._frame_2 = np.eye(4)
    self._link_drawings = LinkDrawing(ax, color)
    self._drawn_once = False

  def update_frames(self, frame_1: np.ndarray, frame_2: np.ndarray):
    """Update the start and end frame of the link

    Args:
        frame_1 (np.ndarray): (4,4) Transformation matrix to the start point
        frame_2 (np.ndarray): (4,4) Transformation matrix to the end point. 
    """
    general.check_proper_numpy_format(frame_1, (4, 4))
    general.check_proper_numpy_format(frame_2, (4, 4))

    self._link_drawings.update_frames(frame_1, frame_2)

  def draw(self):
    """Draw the link"""
    if not self._drawn_once:
      self._link_drawings.draw()
      self._drawn_once = True
    else:
      self._link_drawings.redraw()


class Joint(object):
  """Hold the definition and information for each joint"""
  @property
  def low_limit(self):
    """Returns the joints low joint limit

    Returns:
        float: low joint limit
    """
    return self._joint_limit[0]

  @property
  def high_limit(self):
    """Return the joints high limit

    Returns:
        float: joints high limit
    """
    return self._joint_limit[1]

  @property
  def dh_transform(self):
    """Return the joints dh_transform

    Returns:
        np.ndarray: (4,4) array of the DH transform of the joint
    """
    return self._dh_transform

  @property
  def pos(self):
    return self._theta

  @pos.setter
  def pos(self, value):
    self._theta = value
    self.set_dh_transform()

  @property
  def vel(self):
    return self._theta_dot

  @vel.setter
  def vel(self, value):
    self._theta_dot = value

  @property
  def accel(self):
    return self._accel

  @accel.setter
  def accel(self, value):
    self._accel = value

  @property
  def final_transform(self):
    """Returns the stored final transform of the joint

    Returns:
        np.ndarray: (4,4) final transform of the joint 
    """
    return self._final_transform

  def __init__(self, ax: Axes3D, color=None, frame_size: float = 0.1):
    """Initialize the joint

    Args:
        ax (Axes3D): Axes3D used to draw the joint
        color (array, optional): Color if black isn't desired. Defaults to None.
    """
    self._joint_limit = [0, 0]
    self._drawn_once = False
    self._dh_transform = np.eye(4)
    self._final_transform = np.eye(4)
    self._color = color
    self._frame_drawing = FrameDrawing(ax, color, frame_size=frame_size)

    self._a = 0
    self._alpha = 0
    self._d = 0
    self._theta = 0
    self._theta_dot = 0
    self._accel = 0

  def set_joint_limits(self, low_limit: float, high_limit: float):
    """Set the low and high joint limits

    Args:
        low_limit (float): Low limit of the joint
        high_limit (float): High limit of the joint
    """
    self._joint_limit = [low_limit, high_limit]

  def set_dh_transform(self):
    self._dh_transform = dh_tf(self._alpha, self._a, self._d, self._theta)
    self._update_drawing()

  def set_dh_parameters(self, a: float, alpha: float, d: float,
                        theta: float) -> None:
    self._a = a
    self._alpha = alpha
    self._d = d
    self._theta = theta

    self.set_dh_transform()

  def set_final_transform(self, transform: np.ndarray):
    """Set the final transform in space (where it will get drawn)

    Args:
        transform (np.ndarray): (4,4) transform for location to draw in space
    """
    general.check_proper_numpy_format(transform, (4, 4))
    self._final_transform = transform

  def _update_drawing(self):
    """Update the artist with most recent data"""
    self._frame_drawing.update_frame(self._final_transform)

  def draw(self):
    """Draw the frame"""

    self._update_drawing()
    if not self._drawn_once:
      self._frame_drawing.draw()
      self._drawn_once = True
    else:
      self._frame_drawing.redraw()

  def is_inside_joint_limit(self, input_angle: float) -> bool:
    """Check if the angle is insdie the joint limits

    Args:
        input_angle (float): angle to check

    Returns:
        bool: True if inside the limits, false otherwise 
    """
    if not self.low_limit <= input_angle <= self.high_limit:
      return False
    else:
      return True
    

class RobotPayload(object):

  def __init__(self, ax: Axes3D, color=None):
    self._ax = ax 
    self._color = color 
    self._frame = None 
    self._drawn_once = False 
    self._enabled = False 

    self._point_drawing = PointDrawing(self._ax, self._color)

  def set_position(self, pos: np.ndarray):
    self._frame = np.eye(4)
    self._frame[0, 3] = pos[0]
    self._frame[1, 3] = pos[1]
    self._frame[2, 3] = pos[2]

  def enable(self):
    self._enabled = True 

  def disable(self):
    self._enabled = False 


  def _update_drawing(self):
    """Update the artist with most recent data"""
    self._point_drawing.update_frame(self._frame)
    if self._enabled:
      self._point_drawing.set_visible()
    else:
      self._point_drawing.set_invisible()

  def draw(self):
    """Draw the frame"""

    self._update_drawing()
    if not self._drawn_once:
      self._point_drawing.draw()
      self._drawn_once = True
    else:
      self._point_drawing.redraw()