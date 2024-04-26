"""
File for testing the FOC control loop and motor physics.
Does not rely on torque input, just speed input per the diagram on the poster.
Will report torque.
Used to debug the foc and motor classes.
"""

from foc import FOC
import motor as mot
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    
    # create the motor and foc classes
    motor = mot.Motor(10) # initializes a 3 pole motor with 10 Ohm resistance on each coil

    # create the foc class
    foc = FOC()

    # speed to control to
    DESIRED_SPEED = 100.0 # Hz

    # initialize the variables
    pos = 0.0
    speed = 0.0
    i_q_prev = 0.0
    i_d_prev = 0.0

    # intialize arrays for handling output data
    torque = np.zeros(int(10/mot.TIME_STEP))
    speed= np.zeros(int(10/mot.TIME_STEP))
    time = np.arange(0, 10, mot.TIME_STEP)

    current_speed = 0.0
    current_pos = 0.0

    for t in range(int(10/mot.TIME_STEP)):
        # print(f"Time: {time[t]}")

        speed[t], pos, i_q_prev, i_d_prev = foc.control_loop(motor, DESIRED_SPEED, i_q_prev, i_d_prev, current_pos, current_speed)

        current_speed = speed[t]
        current_pos = pos

        torque[t] = motor.output_torque()

    fig, ax = plt.subplots()
    ax.plot(time, torque, label='Torque')
    ax.plot(time, speed, label='Speed')
    ax.legend()
    plt.title("Torque and speed over time")
    plt.show()