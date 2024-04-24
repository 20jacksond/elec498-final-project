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
import motor

class FOC(object):

    def __init__(self, motor_params) -> None:
        self.VDC = 12
        self.C = np.sqrt(2/3) * np.array([[1, -1/2, -1/2],
                                     [0, np.sqrt(3)/2, -np.sqrt(3)/2],
                                     [1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2)]])
        self.REF = 2 # this needs to be changed
        self.PWM = np.array([[12*np.cos(0), 12*np.sin(0)],
                             [12*np.cos(120), 12*np.sin(120)],
                             [12*np.cos(240), 12*np.sin(240)]])
        self.T = motor.TIME_STEP
        
        self.error_series = []

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
    
    def inv_park(self, theta: float, d_q: np.ndarray) -> np.ndarray:
        """
        Performs the Park Transform on the (alpha, beta) vector to determine the (d, q) vector.
        Input:
            theta - float representing the rotor flux position
            d_q - np.ndarray representing the (d, q) vector
        Output:
            alpha_beta - np.ndarray representing the (alpha, beta) vector
            NOTE alpha_beta is a vector with three elements, but the third = 0
        """
        P = np.array([[np.cos(theta), np.sin(theta), 0],
                      [-np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
        
        alpha_beta = np.linalg.inv(P) @ d_q

        return alpha_beta

    def clarke(self, vec_abc: np.ndarray) -> np.ndarray:
        """
        Performs the Clarke Transform on the (a, b, c) vector to determine the (alpha, beta) vector.
        Input:
            vec_abc - np.ndarry representing the stator current phases (a, b, c)
        Output:
            alpha_beta - np.ndarray representing the (alpha, beta) vector
            NOTE alpha_beta is a vector with three elements, but the third = 0
        """
        alpha_beta = self.C @ vec_abc
        
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
        vec_abc = np.linalg.inv(self.C) @ alpha_beta
        
        return vec_abc
    
    def pi_regulator(self, y_ref: float, y_fbk: float) -> float:
        """
        Performs a PI control loop regulation of signals.
        Inputs:
            y_ref - float representing the reference signal at time t
            y_fbk - float representing the feedback signal at time t
            error - numpy array representing the error over time
        Output:
            u_k - float representing the control output at time t
        """
        # define control gains
        K_P = 10
        K_I = 0.4

        # compute the error between reference and feedback
        error = y_ref - y_fbk

        # sum the previous errors
        sum = self.error_series.sum()

        # calculate the output
        u_k = (K_P * error) + (K_I * error) + sum

        # add the error to the time series
        self.error_series.append(error)

        return u_k

    def svpwm(self, v_sa_ref: float, v_sB_ref: float) -> np.ndarray:
        """
        Determines the PWM control signal applied to a DC inverter. The inverter
        output is returned so that it can be applied to the motor windings.
        Inputs:
            v_sa_ref - float representing the reference voltage for the alpha vector
            v_sB_ref - float representing the reference voltage for the beta vector
        Output:
            i_uvw - np.ndarray representing the currents for each winding at the given timestep
        """
        vector = np.array([[v_sa_ref],
                           [v_sB_ref]])
        
        # determine the coords of the ref voltage according to the state-vector diagram
        B1 = self.PWM[:2, :].T
        B2 = self.PWM[1:3, :].T
        
        coords1 = np.linalg.inv(B1) * vector
        coords2 = np.linalg.inv(B2) * vector

        # normalized to a 12V reference
        u_coords = coords1[0] / 12
        v_coords = coords1[1] / 12
        w_coords = coords2[1] / 12

        # determine the pwm duty cycles using a triangle wave trigger
        u_duty = self.T * (1 - u_coords)
        v_duty = self.T * (1 - v_coords)
        w_duty = self.T * (1 - w_coords)

        return np.array([u_duty, v_duty, w_duty])
    
    def inverter(self, duty_cycles: np.ndarray) -> np.ndarray:
        """
        Produces a 2D array representing the voltage waveforms applied to each motor phase.
        Input:
            duty_cycles - np.ndarray representing the duty cycles for each motor phase
        Output:
            motor_voltages - np.ndarray representign the voltage outputs for each motor phase
        """
        u_duty = duty_cycles[0, 0]
        v_duty = duty_cycles[0, 1]
        w_duty = duty_cycles[0, 2]

        motor_voltages = np.zeros((3, 100))

        # create a time serires to stack every period
        time = np.linspace(0, self.T, num=100)

        # set the values to be on or off
        for i in time:
            if (i >= (self.T/2 - u_duty)) or (i <= (self.T/2 + u_duty)):
                motor_voltages[0, i] = self.VDC
            if (i >= (self.T/2 - v_duty)) or (i <= (self.T/2 + v_duty)):
                motor_voltages[1, i] = self.VDC
            if (i >= (self.T/2 - w_duty)) or (i <= (self.T/2 + w_duty)):
                motor_voltages[2, i] = self.VDC

        return motor_voltages

    def sensorless_sense(self, i_alpha: np.ndarray, i_beta: np.ndarray) -> tuple:
        omega = 0
        theta = 0
        return omega, theta
    
    def control_loop(self, 
                     motor: motor.Motor, 
                     target_speed: float,
                     i_q_prev: float,
                     i_d_prev: float,
                     rotor_pos: float,
                     rotor_speed: float) -> tuple[float]:
        """
        This function completes the full FOC control loop using the Motor object.
        Input:
            motor - the Motor object for simulation
            target_speed - float representing the desired speed to control the motor to
            i_q_prev - float representing a feedback signal
            i_d_prev - float representing a feedback signal
            rotor_pos - float representing the angular position of the motor, relative to a stationary frame
            rotor_speed - the current rotor position minus the previous divided by the time step
        Output:
            a list containing the new motor speed, new position, new i_q_prev, and new i_d_prev
            
        NOTE This structure follow the diagram shown in https://www.ti.com/lit/ml/slyp711/slyp711.pdf?ts=1713049008776&ref_url=https%253A%252F%252Fwww.ecosia.org%252F
        with edits from https://www.ti.com/lit/an/bpra073/bpra073.pdf 
        """

        # run the control loop
        # get the q current
        i_q = self.pi_regulator(target_speed, rotor_speed)

        # get the d and q voltages
        v_d = self.pi_regulator(0, i_d_prev)
        v_q = self.pi_regulator(i_q, i_q_prev)

        # perform the inverse Park transform
        v_a_B = self.inv_park(rotor_pos, np.array([v_d, v_q]).T)

        # space vector PWM
        duty_cycles = self.svpwm(v_a_B[0, 0], v_a_B[1, 0])

        # calculate phase currents
        v_uvw = self.inverter(duty_cycles)
        bemf = motor.calculate_bemf()

        v_diff = v_uvw - bemf

        i_uvw = v_diff / motor.resistance

        # TODO get information from the motor

        # calculate the new speed of the motor
        new_speed, new_pos = self.sensorless_sense(0, 0) 

        # use the current in the feedback portion of the loop
        i_a_B = self.clarke(i_uvw)

        # calculate the new torque control signals
        new_torque_control = self.park(new_pos, i_a_B)

        # return all the important values
        # return array is [speed, position, i_q_prev, i_d_prev]
        return [new_speed, new_pos, new_torque_control[0, 0], new_torque_control[1, 0]]