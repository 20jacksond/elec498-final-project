"""
This file contains my implementation of Field Oriented Control
The algorithm is written as methods inside of the FOC class
Author: Davis Jackson

Sources:
Clarke and Park Transforms: https://en.wikipedia.org/wiki/Direct-quadrature-zero_transformation#Park's_transformation
"""

import numpy as np
import matplotlib as plt
import math

class FOC(object):

    def __init__(self, motor_params) -> None:
        pass

    def park(self, theta: float, alpha_beta: np.ndarray) -> np.ndarray:
        """
        Performs the Park Transform on the (alpha, beta) vector to determine the (d, q) vector.
        Input:
            theta - float representing the rotor flux position
            alpha_beta - np.ndarray representing the (alpha, beta) vector
            NOTE alpha_beta is a vector with three elements, but the third = 0
        Output:
            d_q - np.ndarray representing the (d, q) vector
        """
        P = np.array([[np.cos(theta), np.sin(theta), 0],
                      [-np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
        
        d_q = P @ alpha_beta

        return d_q

    def clarke(self, vec_abc: np.ndarray) -> np.ndarray:
        """
        Performs the Clarke Transform on the (a, b, c) vector to determine the (alpha, beta) vector.
        Input:
            vec_abc - np.ndarry representing the stator current phases (a, b, c)
        Output:
            alpha_beta - np.ndarray representing the (alpha, beta) vector
            NOTE alpha_beta is a vector with three elements, but the third = 0
        """
        C = np.sqrt(2/3) * np.array([[1, -1/2, -1/2],
                                     [0, np.sqrt(3)/2, -np.sqrt(3)/2],
                                     [1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2)]])
        
        alpha_beta = C @ vec_abc
        
        return alpha_beta
    
    def inv_clarke(self, alpha_beta: np.ndarray) -> np.ndarray:
        """
        Performs the inverse Clarke Transform on the (alpha, beta) vector to determine the 
        (a, b, c) phase vector.
        Input:
            alpha_beta - np.ndarray representing the (alpha, beta) vector
            NOTE alpha_beta is a vector with three elements, but the third = 0
        Output:
            vec_abc - np.ndarray representing the (a, b, c) phase vector
        """
        C = np.sqrt(2/3) * np.array([[1, -1/2, -1/2],
                                     [0, np.sqrt(3)/2, -np.sqrt(3)/2],
                                     [1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2)]])
        
        vec_abc = np.linalg.inv(C) @ alpha_beta
        
        return vec_abc
    
    def field_weakening(self, target_velocity: float) -> np.ndarray:
        i_d = 0
        return i_d
    
    def velocity_control(self, target_velocity: float, omega: float) -> np.ndarray:
        i_q = 0
        return i_q

    def flux_control(self, i_d: np.ndarray) -> np.ndarray:
        v_d = 0
        return v_d

    def torque_control(self, i_q: np.ndarray) -> np.ndarray:
        v_q = 0
        return v_q

    def inverter(self, v_uvw: np.ndarray) -> np.ndarray:
        i_uvw = 0
        return i_uvw

    def sensorless_sense(self, i_alpha: np.ndarray, i_beta: np.ndarray) -> tuple:
        omega = 0
        theta = 0
        return omega, theta