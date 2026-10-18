import numpy as np
from scipy.integrate import solve_ivp

from dreamdock.cw import MU_EARTH, R_EARTH, cw_matrices, discretize, mean_motion, rollout, stm

R0 = R_EARTH + 500e3
N = mean_motion(R0)
PERIOD = 2 * np.pi / N


def test_discrete_step_matches_closed_form_stm():
    for dt in [1.0, 10.0, 600.0, PERIOD]:
        Ad, _ = discretize(N, dt)
        np.testing.assert_allclose(Ad, stm(N, dt), rtol=1e-10, atol=1e-10)


def test_thrust_input_matches_ode_integration():
    A, B = cw_matrices(N)
    s0 = np.array([100.0, -50.0, 20.0, 0.1, -0.05, 0.02])
    a = np.array([1e-4, -2e-4, 5e-5])
    dt = 120.0
    Ad, Bd = discretize(N, dt)
    sol = solve_ivp(lambda t, s: A @ s + B @ a, (0, dt), s0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Ad @ s0 + Bd @ a, sol.y[:, -1], rtol=1e-9, atol=1e-9)


def test_bounded_orbit_has_no_drift():
    x0 = 100.0
    s0 = np.array([x0, 0.0, 0.0, 0.0, -2 * N * x0, 0.0])
    steps_per_orbit = 100
    states = rollout(s0, np.zeros((10 * steps_per_orbit, 3)), N, PERIOD / steps_per_orbit)
    np.testing.assert_allclose(states[-1], s0, atol=1e-6)
    assert np.abs(states[:, 1]).max() < 5 * x0  # 2:1 ellipse, |y| <= 2*x0


def test_radial_offset_drifts_behind():
    # Higher orbit is slower: x0 > 0 at rest falls behind at -6 n x0 per unit time (averaged).
    x0 = 100.0
    s0 = np.array([x0, 0, 0, 0, 0, 0])
    s = stm(N, PERIOD) @ s0
    np.testing.assert_allclose(s[1], -6 * N * x0 * PERIOD, rtol=1e-10)


def test_out_of_plane_is_harmonic():
    z0, vz0 = 50.0, 0.03
    t = np.linspace(0, 3 * PERIOD, 50)
    z = [(stm(N, ti) @ [0, 0, z0, 0, 0, vz0])[2] for ti in t]
    np.testing.assert_allclose(z, z0 * np.cos(N * t) + vz0 / N * np.sin(N * t), atol=1e-9)


def _two_body_lvlh(rho0, rhodot0, t):
    """Propagate chaser with full two-body gravity, return LVLH state at t."""
    w = np.array([0.0, 0.0, N])

    def frame(tt):
        th = N * tt
        C = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        return C, C @ [R0, 0, 0], C @ [0, N * R0, 0]

    C, rt, vt = frame(0)
    rc = rt + C @ rho0
    vc = vt + C @ (rhodot0 + np.cross(w, rho0))

    def f(_, y):
        r = y[:3]
        return np.concatenate([y[3:], -MU_EARTH * r / np.linalg.norm(r) ** 3])

    y = solve_ivp(f, (0, t), np.concatenate([rc, vc]), rtol=1e-13, atol=1e-6).y[:, -1]
    C, rt, vt = frame(t)
    rho = C.T @ (y[:3] - rt)
    return np.concatenate([rho, C.T @ (y[3:] - vt) - np.cross(w, rho)])


def test_linearization_error_is_second_order():
    # CW drops O(rho^2/r0) terms: halving the offset should quarter the error.
    t = PERIOD / 2
    errs = []
    for scale in [1000.0, 500.0]:
        rho0 = scale * np.array([1.0, 0.5, 0.3])
        rhodot0 = np.array([0.0, -2 * N * rho0[0], 0.0])
        cw = stm(N, t) @ np.concatenate([rho0, rhodot0])
        errs.append(np.linalg.norm(_two_body_lvlh(rho0, rhodot0, t)[:3] - cw[:3]))
    assert errs[0] / 1000.0 < 1e-2  # < 1% of offset at 1 km
    assert 3.5 < errs[0] / errs[1] < 4.5
