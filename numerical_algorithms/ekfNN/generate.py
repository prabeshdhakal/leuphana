import os
import random
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def first_order_signal(N=100, noise=0.15, Kp=3, Tp=2, random_state=0, plot=False):
    """
    Simulate a first order system (Tp * dy/dt = -y + K*u)
    with an accompanying plot.

        State Space Model:
            X_dot = Ax + Bu
                y = Cx + Du

            Where:
            A = 1 / Tp, B = Kp / Tp
            C = 1,      D = 0

        Differential Equation: dy/dt = -(1/Tp)y + (Kp/Tp)u

    Based on the implementation presented here:
    https://apmonitor.com/pdc/index.php/Main/ModelSimulation

    Parameters
    ----------
        N : int
            The no. of data points to generate
        noise : float
            The amount of measurement noise to apply
        random_state : int
            The seed value applied for the introduced noise.
    Outputs
    -------
        t_true : array
            Time index
        y_true : array
            True values of the signal
        y_meas : array
            The measurement values
    """
    np.random.seed(random_state)

    # State Space Model
    A = -1.0 / Tp
    B = Kp / Tp
    C = 1.0
    D = 0.0
    sys = signal.StateSpace(A, B, C, D)
    t_true, y_true = signal.step(sys, N=N)

    # "take" measurements
    y_meas = np.add(y_true, noise * np.random.randn(y_true.shape[0]))

    if plot == True:
        plt.plot(t_true, y_true, "-", label="signal")
        plt.plot(t_true, y_meas, ".", label="measurement")
        plt.legend()
        plt.show()

    # Reshape the values
    t_true = t_true.reshape(-1, 1)
    y_true = y_true.reshape(-1, 1)
    y_meas = y_meas.reshape(-1, 1)

    return t_true, y_true, y_meas


def second_order_signal(N=100, noise=0.15, Kp=3, Tp=1, zeta=0.15, random_state=0, plot=False):
    """
    Simulate a second order system with an accompanying plot.

        State-Space Model:
            x1_dot = 0*x1 + 1*x2 + 0*u(t - theta_p)
            x2_dot = -(1/Tp^2)x1 + (-2*zeta/Tp)x2 + (Kp/Tp^2)*u(t-theta_p) 

        Second Order Differential Equation:
            Tp^2 * (d^2y/dt^2) + 2*zeta*Tp*(dy/dt) + y = Kp*u(t-theta_p)

    Based on the implementation presented here:
    https://apmonitor.com/pdc/index.php/Main/ModelSimulation

    Parameters
    ----------
        N : int
            The no. of data points to generate
        noise : float
            The amount of measurement noise to apply
        random_state : int
            The seed value applied for
    Outputs
    -------
        t_true : array
            Time index
        y_true : array
            True values of the signal
        y_meas : array
            The measurement values
    """
    np.random.seed(random_state)

    # State Space Model
    A = [[0.0, 1.0], [-1.0 / Tp ** 2, -2.0 * zeta / Tp]]
    B = [[0.0], [Kp / Tp ** 2]]
    C = [1.0, 0.0]
    D = 0.0
    sys = signal.StateSpace(A, B, C, D)
    t_true, y_true = signal.step(sys, N=N) # the signal


    # "take" measurements (add measurement signal)
    y_meas = np.add(y_true, noise * np.random.randn(y_true.shape[0]))

    if plot == True:
        plt.plot(t_true, y_true, "-", label="signal")
        plt.plot(t_true, y_meas, ".", label="measurement")
        plt.legend()
        plt.show()

    # Reshape the arrays
    t_true = t_true.reshape(-1, 1)
    y_true = y_true.reshape(-1, 1)
    y_meas = y_meas.reshape(-1, 1)

    return t_true, y_true, y_meas

def sin_signal(N=100, noise=0.15, random_state=0, plot=False):
    """
    Generates sinusoidal signal with accompanying plot.

    Parameters
    ----------
        N : int 
            The no. of data points to generate
        noise : float
            The amount of measurement noise to apply
        random_state : int
            The seed value applied for 
    Outputs
    -------
        t_true : array
            Time index
        y_true : array
            True values of the signal
        y_meas : array
            The measurement values
    """
    np.random.seed(random_state)
    t_true = np.sort(np.random.randn(N, 1), axis=0).reshape(-1)
    y_true = np.sin(t_true * 2 * np.pi / 3)
    y_meas = y_true + 0.25 * np.random.randn(*y_true.shape)

    if plot == True:
        fig, ax = plt.subplots(1)
        ax.plot(t_true, y_true, "-", label="signal")
        ax.plot(t_true, y_meas, ".", label="measurement")
        plt.legend()
        plt.show()

    # Reshape
    t_true = t_true.reshape(-1, 1)
    y_true = y_true.reshape(-1, 1)
    y_meas = y_meas.reshape(-1, 1)

    return t_true, y_true, y_meas
