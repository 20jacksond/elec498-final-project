"""
Describing the motor physics in detail.
"""

import numpy as np

# defining a few constants
TIME_STEP = 0.001

class Motor(object):

    def __init__(self, l: float, m: float, r: float):
        """
        Sets up the BLDC motor using the input parameters
        """
        # define the inductances
        self._L = l
        self._M = m

        # define the resistance
        self._R = r

        # physical characteristics
        self._P = 3 # number of poles
        self._F = 0.2 # mechanical friction coefficient
        self._lambda = 0 # TODO flux linkage
        self._I = 10 # rotational inertia of rotor

        # create more variables to hold statistics about the motor
        self.speed_m = 0 # in radians per second
        self.flux = None
        self.Theta_r = 0 # rotor position initialized at zero
        self.Theta_r_prev = 0 # rotor speed

    def calculate_phase_voltage(self, current: np.ndarray, prev_current: np.ndarray,
                                bemf: np.ndarray) -> np.ndarray:
        """
        Calculates the voltage for each phase using the motor params and BEMF.
        Input:
            current - 3x1 np.ndarray representing the current on each phase winding
            prev_current - 3x1 np.ndarray representing the previous current values
            bemf: 3x1 np.ndarray representing the back EMF on each phase winding
        Output:
            voltage - 3x1 np.ndarray representing the voltage on each phase winding
        """
        # define the resistance and inductance matrices
        R = self._R * np.eye(3)
        L = (self._L - self._M) * np.eye(3)

        # find the derivative of the phase currents
        # this is simply the current minus the previous
        derivative = (current - prev_current) / TIME_STEP

        # add the different voltages together
        voltage = (R @ current) + (L @ derivative) + bemf

        return voltage
    
    def calculate_speed_m(self):
        """
        Calculate the mechanical speed of the motor.
        """
        self.speed_m = (self.Theta_r_prev - self.Theta_r) / (self._P * TIME_STEP)

    def fas(self, theta: float) -> float:
        """
        Calculates the back emf for various magnitude instants
        """
        bemf_a = 0# P * B * L * n * W * R
        return bemf_a

    def fbs(self, theta: float) -> float:
        """
        Calculates the back emf for various magnitude instants
        """
        bemf_b = 0
        return bemf_b
    
    def fcs(self, theta: float) -> float:
        """
        Calculates the back emf for various magnitude instants
        """
        bemf_c = 0
        return bemf_c

    def calculate_bemf(self) -> np.ndarray:
        """
        Calculates the Back EMF on each phase.
        Output:
            bemf - 3x1 np.ndarray representing the back emf on each phase
        """
        bemf = self.speed_m * self._lambda * np.array([[self.fas(self.Theta_r)],
                                                       [self.fbs(self.Theta_r)],
                                                       [self.fcs(self.Theta_r)]])
        
        return bemf

    def output_torque(self, current: np.ndarray, bemf:np.ndarray) -> float:
        """
        Calculates the output torque of the motor.
        Input:
            bemf: 3x1 np.ndarray representing the back EMF on each phase winding
        Output:
            torque - float representing the output torque of the motor
        """
        # define values
        ea = bemf[0, 0]
        eb = bemf[1, 0]
        ec = bemf[2, 0]
        ia = current[0, 0]
        ib = current[1, 0]
        ic = current[2, 0]

        torque_e = (1 / self.speed_m) * ((ea*ia) + (eb*ib) + (ec*ic))

        torque_m = torque_e - (self._I * (self.Theta_r - self.Theta_r_prev) / TIME_STEP) - (self._F * self.Theta_r)

        return torque_m