"""Clohessy-Wiltshire relative motion.

Hill (LVLH) frame centered on a target in a circular orbit:
    x radial (outward), y along-track, z orbit normal.
State s = [x, y, z, vx, vy, vz] (m, m/s). Input a = chaser thrust acceleration (m/s^2).

    x'' - 2n y' - 3n^2 x = a_x
    y'' + 2n x'          = a_y
    z'' + n^2 z          = a_z
"""

import numpy as np
from scipy.linalg import expm

MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6378.137e3  # m


def mean_motion(r0, mu=MU_EARTH):
    """n = sqrt(mu / r0^3), rad/s."""
    return np.sqrt(mu / r0**3)


def cw_matrices(n):
    """Continuous-time A (6x6), B (6x3) for s' = A s + B a."""
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3, 0] = 3 * n**2
    A[3, 4] = 2 * n
    A[4, 3] = -2 * n
    A[5, 2] = -(n**2)
    B = np.zeros((6, 3))
    B[3:6, :] = np.eye(3)
    return A, B


def discretize(n, dt):
    """Exact zero-order-hold discretization: s_{k+1} = Ad s_k + Bd a_k.

    Van Loan: expm([[A, B], [0, 0]] dt) = [[Ad, Bd], [0, I]].
    """
    A, B = cw_matrices(n)
    M = np.zeros((9, 9))
    M[:6, :6] = A
    M[:6, 6:] = B
    E = expm(M * dt)
    return E[:6, :6], E[:6, 6:]


def stm(n, t):
    """Closed-form state transition matrix Phi(t) for coasting motion."""
    c, s, nt = np.cos(n * t), np.sin(n * t), n * t
    return np.array([
        [4 - 3 * c,      0, 0,  s / n,             2 * (1 - c) / n,     0],
        [6 * (s - nt),   1, 0, -2 * (1 - c) / n,   (4 * s - 3 * nt) / n, 0],
        [0,              0, c,  0,                 0,                   s / n],
        [3 * n * s,      0, 0,  c,                 2 * s,               0],
        [-6 * n * (1 - c), 0, 0, -2 * s,           4 * c - 3,           0],
        [0,              0, -n * s, 0,             0,                   c],
    ])


def rollout(s0, actions, n, dt):
    """Propagate s0 under a (T, 3) sequence of piecewise-constant thrusts. Returns (T+1, 6)."""
    Ad, Bd = discretize(n, dt)
    states = np.empty((len(actions) + 1, 6))
    states[0] = s0
    for k, a in enumerate(actions):
        states[k + 1] = Ad @ states[k] + Bd @ a
    return states
