"""Check of the stability and rotation columns of Table 1 for the certified orbits (certificates/orbits_12.json).
Along each orbit (integrated in the plane with RK4, two step sizes) the normal Jacobi equation J'' + K J = 0 of the
closed geodesic on the shape sphere is integrated in g-arclength (ds = 2U dt at E = 0), K from the exact curvature of
g = 2U g0 (verify_converse.geom). The trace of the return map J1(L) + J2'(L) (|trace| > 2: hyperbolic), its
determinant (must be 1), and the rotation of the configuration per period (the least-squares angle taking q(0) to
q(T)) are compared with the paper's values; the rotation also against the geometric phase (half the solid angle the
shape curve encloses about C12, Montgomery 1996). Uses only this folder's code.
    python verify/verify_stability.py"""
import cmath
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # python -I does not add the script's folder
from verify_converse import geom  # noqa: E402
from verify_orbits import deriv  # noqa: E402
PAIRS = ((0, 1), (1, 2), (2, 0))
# the paper's Table 1: trace, rotation in degrees
CLAIMED = {(1, 1, 0.4): (53.50, 146.0), (1, 1, 0.5): (8.102, -116.9), (1, 1, 0.52): (4.988, -89.0),
           (1, 1, 0.53): (3.872, -73.3), (1, 1, 0.54): (3.017, -56.1), (1, 1, 0.55): (2.413, -37.0),
           (1, 1, 0.56): (2.068, -15.5), (1, 0.5, 0.3): (22.01, -171.1), (1, 0.2, 0.15): (19.94, -169.0)}


def hopf(q, m):
    m1, m2, m3 = m
    M12, M = m1 + m2, m1 + m2 + m3
    mu12, mu3 = m1 * m2 / M12, m3 * M12 / M
    Q = q[:, 0] + 1j * q[:, 1]
    z1 = math.sqrt(mu12) * (Q[1] - Q[0])
    z2 = math.sqrt(mu3) * (Q[2] - (m1 * Q[0] + m2 * Q[1]) / M12)
    c = 2 * z1 * z2.conjugate()
    w = np.array([abs(z1) ** 2 - abs(z2) ** 2, c.real, c.imag])
    return w / np.linalg.norm(w)


def run(orb, N):
    m = np.array(orb["masses"], float)
    q0 = np.array([float(x) for x in orb["positions"]]).reshape(3, 2)
    v0 = np.array([float(x) for x in orb["velocities"]]).reshape(3, 2)
    T = float(orb["period"])
    h = T / N

    def f(Y):
        q = Y[0:6].reshape(3, 2)
        U = sum(m[i] * m[j] / float(np.sum((q[i] - q[j]) ** 2)) for i, j in PAIRS)
        K = float(geom(hopf(q, m).reshape(3, 1), tuple(m))[2][0])
        J = Y[13:17].reshape(2, 2)  # rows: J and dJ/ds; columns: the two solutions
        return np.concatenate([deriv(Y[:13], m), np.array([2 * U * J[1], -2 * U * K * J[0]]).ravel()])

    Y = np.concatenate([q0.ravel(), v0.ravel(), [0.0], [1.0, 0.0, 0.0, 1.0]])
    c12 = np.array([-1.0, 0.0, 0.0])
    w_prev, omega = hopf(q0, m), 0.0
    for _ in range(N):
        k1 = f(Y)
        k2 = f(Y + 0.5 * h * k1)
        k3 = f(Y + 0.5 * h * k2)
        k4 = f(Y + h * k3)
        Y = Y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        w = hopf(Y[0:6].reshape(3, 2), m)  # solid angle swept about C12 (fan of spherical triangles)
        omega += 2 * math.atan2(c12 @ np.cross(w_prev, w), 1 + w_prev @ w + w @ c12 + c12 @ w_prev)
        w_prev = w
    J = Y[13:17].reshape(2, 2)
    qT = Y[0:6].reshape(3, 2)
    A0, AT = q0[:, 0] + 1j * q0[:, 1], qT[:, 0] + 1j * qT[:, 1]
    rot = math.degrees(cmath.phase(np.sum(m * A0.conjugate() * AT)))
    phase = (math.degrees(omega / 2) + 180) % 360 - 180  # the geometric phase: half the enclosed solid angle
    return J[0, 0] + J[1, 1], J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0], rot, phase


def main():
    orbs = json.load(open(os.path.join(HERE, "certificates", "orbits_12.json"), encoding="utf-8"))["orbits"]
    allok = True
    for orb in orbs:
        key = tuple(orb["masses"])
        t1, _, _, _ = run(orb, 2 ** 13)
        t2, det, rot, phase = run(orb, 2 ** 14)
        ct, cr = CLAIMED[key]
        ok = (abs(t2 - t1) < 1e-3 * abs(t2) and abs(det - 1) < 1e-6 and abs(t2 - ct) <= 0.006 * abs(ct)
              and abs(rot - cr) <= 0.1 and abs(t2) > 2 and abs((rot - phase + 180) % 360 - 180) < 0.05)
        allok &= ok
        print(f"masses {str(key):15s} trace {t2:9.4f} (2^13 steps: {t1:9.4f}; paper {ct}), det {det:.8f}, "
              f"rotation {rot:+8.2f} deg (paper {cr:+.1f}; geometric phase {phase:+8.2f}): {'hyperbolic, ' if abs(t2) > 2 else ''}"
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    print("ALL PASS" if allok else "SOME FAILED")


if __name__ == "__main__":
    main()
