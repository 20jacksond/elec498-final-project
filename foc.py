"""
This file contains my implementation of Field Oriented Control
The algorithm is written as methods inside of the FOC class
Author: Davis Jackson
"""

import numpy as np
import matplotlib as plt
import math

class FOC(object):

    def __init__(self, motor_params) -> None:
        pass

    def park(self) -> None:
        pass

    def clarke(self, vec_abc) -> np.ndarray:
        Kc = np.sqrt(2/3) * np.array([[1, -1/2, -1/2],
                                      [0, np.sqrt(3)/2, -np.sqrt(3)/2],
                                      [1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2)]])
        
        return Kc @ vec_abc
    
    def inv_clarke(self, vec_xyz) -> np.ndarray:
        Kc = np.sqrt(2/3) * np.array([[1, -1/2, -1/2],
                                      [0, np.sqrt(3)/2, -np.sqrt(3)/2],
                                      [1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2)]])
        
        return np.linalg.inv(Kc) @ vec_xyz

    def pi_control(self) -> None:
        pass

    def inverter(self) -> None:
        pass

    def sensorless_sense(self) -> None:
        pass