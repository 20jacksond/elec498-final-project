"""
Describing the motor physics in detail.
"""

import numpy as np

# defining a few constants
TIME_STEP = 0.001

class Motor(object):

    @property
    def resistance(self):
        return self._R
    
    @property
    def position(self):
        return self.Theta_r
    
    @property
    def prev_pos(self):
        return self.Theta_r_prev
    
    @property
    def speed(self):
        return self.speed_m

    def __init__(self, r: float):
        """
        Sets up the BLDC motor using the input parameters
        """

        # define the resistance
        self._R = r

        # define the power rating
        self._W = 1 # HP

        # physical characteristics
        self._P = 3 # number of poles
        self._N = 100 # number of coils per phase
        self._B = 0.1 # magnetic field strength in T for the PM rotor
        self._A = 0.00785 # area of each phase coil
        self._I = 10 # rotational inertia of rotor

        # create more variables to hold statistics about the motor
        self.speed_m = 0 # in radians per second
        self.Theta_r = 0 # rotor position initialized at zero
        self.Theta_r_prev = 0 # rotor speed

    def update_position(self, bemf: np.ndarray):
        """
        Updates the position of the rotor with respect to the u phase coil.
        Basically the reverse of the work done to get the Back-EMF
        Input:
            bemf - 3x0 np.ndarray representing the back emf on each phase
        """
        self.Theta_r_prev = self.Theta_r

        # use the bemf and magnetic flux of the u phase coil (absolute 0 for the rotor)
        sin = bemf[0] / (self._N * self._B * self._A * self.speed_m)
        cos = np.sqrt(1 - (sin**2))

        # use atan2 to make sure we get the correct angle
        option1 = np.arctan2(sin, cos)
        option2 = np.arctan2(sin, -cos)

        # choose the resulting angle closest to the previous angle
        if abs(option1 - self.Theta_r_prev) <= abs(option2 - self.Theta_r_prev):
            self.Theta_r = option1
        else:
            self.Theta_r = option2

    def calculate_speed_m(self):
        """
        Calculate the mechanical speed of the motor.
        """
        self.speed_m = (self.Theta_r_prev - self.Theta_r) / (self._P * TIME_STEP)

    def calculate_bemf(self) -> np.ndarray:
        """
        Calculates the Back EMF on each phase. Assumes the north pole of the rotor magnet
        is oriented facing the u phase coil when Theta_r=0
        Output:
            bemf - 3x0 np.ndarray representing the back emf on each phase
        """
        bemf = np.zeros((3,))
        
        # calculate the Back-EMF on the u coil
        bemf[0] = self._N * self._B * self._A * np.sin(self.Theta_r) * self.speed_m

        # calculate the Back-EMF on the v coil (+120 degrees)
        bemf[1] = self._N * self._B * self._A * np.sin(self.Theta_r + (2*np.pi/3)) * self.speed_m

        # calculate the Back-EMF on the w coil (+240 degrees)
        bemf[2] = self._N * self._B * self._A * np.sin(self.Theta_r + (4*np.pi/3)) * self.speed_m

        return bemf

    def output_torque(self) -> float:
        """
        Calculates the output torque of the motor. Uses the internal mechanical speed tracker var
        Output:
            torque - float representing the output torque of the motor
        """
        torque = (60 * self._W) / (2 * np.pi * self.speed_m)

        return torque