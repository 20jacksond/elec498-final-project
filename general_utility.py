import numpy as np
import math
import yaml


def rollr(roll: float) -> np.ndarray:
  """Create an X Rotation matrix

  Args:
      roll (float): radian value to rotate around x

  Returns:
      np.ndarray: 3x3 numpy array of rotation matrix
  """
  return np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)],
                  [0, math.sin(roll), math.cos(roll)]])

def rollT(roll: float) -> np.ndarray:
  """Create a transformation matrix with a roll about X

  Args:
      roll (float): radian value to rotate around x

  Returns:
      np.ndarray: 4x4 numpy array of transformation matrix with X roll
  """
  result = np.eye(4)
  result[:3, :3] = rollr(roll)
  return result


def pitchr(pitch: float) -> np.ndarray:
  """Create a Y rotation matrix

  Args:
      pitch (float): radian value to rotate around Y 

  Returns:
      np.ndarray: 3x3 numpy array of rotation matrix about Y 
  """
  return np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0],
                  [-math.sin(pitch), 0, math.cos(pitch)]])

def pitchT(pitch: float) -> np.ndarray:
  """Create a transformation matrix with a pitch about Y 

  Args:
      pitch (float): radian value to pitch around Y 

  Returns:
      np.ndarray: 4x4 numpy array of transformation matrix with Y pitch 
  """
  result = np.eye(4) 
  result[:3, :3] = pitchr(pitch)
  return result


def yawr(yaw: float) -> np.ndarray:
  """Create a rotation matrix with a yaw about Z

  Args:
      yaw (float): radian value to rotate around Z

  Returns:
      np.ndarray: 3x3 numpy array of rotation matrix with Z yaw 
  """
  return np.array([[math.cos(yaw), -math.sin(yaw), 0],
                  [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])

def yawT(yaw: float) -> np.ndarray:
  """Create a transformation matrix with a yaw about Z

  Args:
      yaw (float): radian value to rotate around the Z axis 

  Returns:
      np.ndarray: 4x4 numpy array of transformation matrix with a Z yaw 
  """
  result = np.eye(4)
  result[:3, :3] = yawr(yaw) 
  return result 


def check_proper_numpy_format(value: np.ndarray, shape: tuple) -> bool:
  """Send in a numpy array to make sure it is the right shape 

  Args:
      value (np.ndarray): Input value to test
      shape (tuple): desired shape as a tuple ex (4,4)

  Returns:
      bool: If True, the array is well formed 
  """
  if not isinstance(value, np.ndarray):
    return False

  if value.shape != shape:
    return False

  return True



def get_data_from_yaml(filepath: str) -> dict:
  """Get the data out of a yaml file in dictionary form

  Args:
      filepath (str): full filepath to the location on your computer
                      eg: "/home/lcfarrell/ME_498/Lab2/pahts/path_name.yaml"

  Returns:
      dict: dictionary in the format {"color": int, "x": float, "y": float, "z": float}
  """

  with open(filepath, "r") as yaml_file:
    data = yaml.safe_load(yaml_file)
  return data
