import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import general_utility as general


class FrameDrawing(object):
  """Class to draw and hold information for a frame"""
  def __init__(self, ax: Axes3D, color=None, frame_size: float = 0.1):
    """Initialize the fraem drawing, set a color if desired

    Args:
        ax (Axes3D): the subplot that this frame will draw on 
        color (str or array, optional): string or array description of a color. Defaults to None.
    """

    self._frame_size = frame_size
    self._x_unit = np.array([
        1, 0, 0
    ]) * self._frame_size  # unit vector in X of length LENGTH for calcs
    self._y_unit = np.array([
        0, 1, 0
    ]) * self._frame_size  # unit vector in Y of length LENGTH for calcs
    self._z_unit = np.array([
        0, 0, 1
    ]) * self._frame_size  # unit vector in Z of length LENGTH for calcs
    self._ax = ax
    self.color = color

    self._x_axis_artist = None
    self._y_axis_artist = None
    self._z_axis_artist = None

    self._x_vect = self._x_unit
    self._y_vect = self._y_unit
    self._z_vect = self._z_unit

    self._x_pos = 0.0
    self._y_pos = 0.0
    self._z_pos = 0.0

    self._rotation = np.eye(3)

  def update_location(self, x: float, y: float, z: float):
    """Update the internal location of the frame drawing

    Args:
        x (float): x location 
        y (float): y_location
        z (float): z_location
    """
    self._x_pos = x
    self._y_pos = y
    self._z_pos = z

    self._pos_vect = np.array([self._x_pos, self._y_pos, self._z_pos])

  def update_location_vector(self, location_vector: np.ndarray):
    """Update the internal location of the frame using a vector

    Args:
        location_vector (np.ndarray): location of the frame as a (3,) np vector
    """
    general.check_proper_numpy_format(location_vector, (3, ))
    self._pos_vect = location_vector
    self._x_pos = location_vector[0]
    self._y_pos = location_vector[1]
    self._z_pos = location_vector[2]

  def update_rotation(self, rotation_matrix: np.ndarray):
    """Update the rotation matrix of the frame

    Args:
        rotation_matrix (np.ndarray): (3,3) numpy rotation matrix 
    """
    general.check_proper_numpy_format(rotation_matrix, (3, 3))
    self._rotation = rotation_matrix

    self._x_vect = rotation_matrix @ self._x_unit
    self._y_vect = rotation_matrix @ self._y_unit
    self._z_vect = rotation_matrix @ self._z_unit

  def update_frame(self, transform_matrix: np.ndarray):
    """update the rotation and translation using a frame

    Args:
        transform_matrix (np.ndarray): (4,4) transformation matrix
    """
    general.check_proper_numpy_format(transform_matrix, (4, 4))

    self.update_location_vector(transform_matrix[:3, 3])
    self.update_rotation(transform_matrix[:3, :3])

  def draw(self):
    """Draw the frame in space"""
    # Draw X axis
    (self._x_axis_artist, ) = self._ax.plot(
        [self._x_pos, self._x_pos + self._x_vect[0]],
        [self._y_pos, self._y_pos + self._x_vect[1]],
        [self._z_pos, self._z_pos + self._x_vect[2]],
        color="r" if self.color is None else self.color,
    )

    # Draw Y Axis
    (self._y_axis_artist, ) = self._ax.plot(
        [self._x_pos, self._x_pos + self._y_vect[0]],
        [self._y_pos, self._y_pos + self._y_vect[1]],
        [self._z_pos, self._z_pos + self._y_vect[2]],
        color="g" if self.color is None else self.color,
    )

    # Draw Z Axis
    (self._z_axis_artist, ) = self._ax.plot(
        [self._x_pos, self._x_pos + self._z_vect[0]],
        [self._y_pos, self._y_pos + self._z_vect[1]],
        [self._z_pos, self._z_pos + self._z_vect[2]],
        color="b" if self.color is None else self.color,
    )

  def _update_artist(self, artist, x_pos: float, y_pos: float, z_pos: float,
                     vector: np.ndarray):
    """Update the internal values of the artis to render in new way next draw

    Args:
        artist (_type_): the artist, line3D to update
        x_pos (float): x position of the artist
        y_pos (float): y position of the artist
        z_pos (float): z position of the artist
        vector (np.ndarray): (3,3) rotation matrix of the artist
    """
    artist.set_xdata([x_pos, x_pos + vector[0]])
    artist.set_ydata([y_pos, y_pos + vector[1]])
    artist.set_3d_properties([z_pos, z_pos + vector[2]])

  def _update_drawing(self):
    """Update the internal values of the drawing artists"""
    self._update_artist(self._x_axis_artist, self._x_pos, self._y_pos,
                        self._z_pos, self._x_vect)
    self._update_artist(self._y_axis_artist, self._x_pos, self._y_pos,
                        self._z_pos, self._y_vect)
    self._update_artist(self._z_axis_artist, self._x_pos, self._y_pos,
                        self._z_pos, self._z_vect)

  def redraw(self):
    """Redraw the artists using temporary animations"""
    self._update_drawing()

    self._x_axis_artist.set_animated(True)
    self._ax.draw_artist(self._x_axis_artist)
    self._x_axis_artist.set_animated(False)

    self._y_axis_artist.set_animated(True)
    self._ax.draw_artist(self._y_axis_artist)
    self._y_axis_artist.set_animated(False)

    self._z_axis_artist.set_animated(True)
    self._ax.draw_artist(self._z_axis_artist)
    self._z_axis_artist.set_animated(False)


class LinkDrawing(object):
  """Objects holding the information to draw/redraw a link"""
  def __init__(self, ax: Axes3D, color=None):
    """Initialize the Link Drawing

    Args:
        ax (Axes3D): The Axes3D to use for drawing
        color (_type_, optional): The color if other than rgb desired. Defaults to None.
    """
    self._ax = ax
    self._color = color
    self._line_artist = None

    self._frame_1 = np.eye(4)
    self._frame_2 = np.eye(4)

  def update_frames(self, frame_1: np.ndarray, frame_2: np.ndarray):
    """Update the starting and ending frame of the link

    Args:
        frame_1 (np.ndarray): (4,4) transformation for start point
        frame_2 (np.ndarray): (4,4) transformation for end point
    """
    general.check_proper_numpy_format(frame_1, (4, 4))
    general.check_proper_numpy_format(frame_2, (4, 4))

    self._frame_1 = frame_1
    self._frame_2 = frame_2

  def draw(self):
    """Draw the line from start to end """
    (self._line_artist, ) = self._ax.plot(
        [self._frame_1[0,3],
         self._frame_2[0,3]],
        [self._frame_1[1,3],
         self._frame_2[1,3]],
        [self._frame_1[2,3],
         self._frame_2[2,3]],
        color='b' if self._color is None else self._color)

  def _update_drawing(self):
    """Update the artist with the internal data"""
    self._line_artist.set_xdata(
        [self._frame_1[0,3],
         self._frame_2[0,3]])
    self._line_artist.set_ydata(
        [self._frame_1[1,3],
         self._frame_2[1,3]])
    self._line_artist.set_3d_properties(
        [self._frame_1[2,3],
         self._frame_2[2,3]])

  def redraw(self):
    """Redraw the existing artist """
    self._update_drawing()
    self._line_artist.set_animated(True)
    self._ax.draw_artist(self._line_artist)
    self._line_artist.set_animated(False)


class PointDrawing(object):
  """ Object to draw a point that changes in Space """
  def __init__(self, ax: Axes3D, color=None):
    """Initialize the Link Drawing

    Args:
        ax (Axes3D): The Axes3D to use for drawing
        color (_type_, optional): The color if other than rgb desired. Defaults to None.
    """
    self._ax = ax
    self._color = color
    self._point_artist = None
    self._frame = None

  def update_frame(self, frame: np.ndarray):

    general.check_proper_numpy_format(frame, (4, 4))

    self._frame = frame 

  def draw(self):
    """ Draw the point """
    (self._point_artist, ) = self._ax.plot(
      [self._frame[0, 3]],
      [self._frame[1, 3]],
      [self._frame[2, 3]],
      color = 'black' if self._color is None else self._color,
       marker='o', markersize=10
    )

  def _update_drawing(self):
    self._point_artist.set_xdata([self._frame[0,3]])
    self._point_artist.set_ydata([self._frame[1,3]])
    self._point_artist.set_3d_properties([self._frame[2,3]])

  def redraw(self):
    """Redraw the existing artist """
    self._update_drawing()
    self._point_artist.set_animated(True) 
    self._ax.draw_artist(self._point_artist)
    self._point_artist.set_animated(False)

  def set_visible(self):
    if self._point_artist is not None:
      self._point_artist.set_visible(True)
  
  def set_invisible(self):
    if self._point_artist is not None:
      self._point_artist.set_visible(False)