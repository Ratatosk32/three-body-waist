"""Stability and rotation of the certified class-(12) orbits (verify/certificates/orbits_12.json).
Along each orbit (integrated in the plane with RK4), the normal Jacobi equation J'' + K J = 0 of the closed geodesic on
the shape sphere is integrated in g-arclength s (ds = 2U dt at E = 0), K from the exact curvature of g = 2U g0. The trace
of the monodromy (J1(L) + J2'(L)) decides: |trace| > 2 hyperbolic. Also the rotation of the configuration per period
(the least-squares angle taking q(0) to q(T)), and the turning of the binary axis 1-2 and of the line from the binary's
centre of mass to body 3. Two step counts as a convergence check.
    python tools/words_orbit_stability.py"""
import cmath
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify"))
from verify_converse import geom  # noqa: E402  (exact K of g = 2U g0 on the unit shape sphere)
from verify_orbits import deriv  # noqa: E402

PAIRS = ((0, 1), (1, 2), (2, 0))


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


def U_pos(q, m):
    return sum(m[i] * m[j] / float(np.sum((q[i] - q[j]) ** 2)) for i, j in PAIRS)


def run(orb, N):
    m = np.array(orb["masses"], float)
    q0 = np.array([float(x) for x in orb["positions"]]).reshape(3, 2)
    v0 = np.array([float(x) for x in orb["velocities"]]).reshape(3, 2)
    T = float(orb["period"])
    h = T / N

    def f(Y):
        y = Y[:13]
        q = y[0:6].reshape(3, 2)
        U = U_pos(q, m)
        K = float(geom(hopf(q, m).reshape(3, 1), tuple(m))[2][0])
        J = Y[13:17].reshape(2, 2)  # columns: two solutions; rows: J, dJ/ds
        dJ = np.array([2 * U * J[1], -2 * U * K * J[0]])
        return np.concatenate([deriv(y, m), dJ.ravel()]), U, K

    Y = np.concatenate([q0.ravel(), v0.ravel(), [0.0], [1.0, 0.0, 0.0, 1.0]])
    Kmax, worst_U = -math.inf, 0.0
    for _ in range(N):
        k1, U, K = f(Y)
        Kmax = max(Kmax, K)
        w = hopf(Y[0:6].reshape(3, 2), m)
        k2 = f(Y + 0.5 * h * k1)[0]
        k3 = f(Y + 0.5 * h * k2)[0]
        k4 = f(Y + h * k3)[0]
        Y = Y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    worst_U = abs(float(geom(w.reshape(3, 1), tuple(m))[0][0]) / U - 1)
    J = Y[13:17].reshape(2, 2)
    trace = J[0, 0] + J[1, 1]
    det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    qT = Y[0:6].reshape(3, 2)
    A0, AT = q0[:, 0] + 1j * q0[:, 1], qT[:, 0] + 1j * qT[:, 1]
    rot = math.degrees(cmath.phase(np.sum(m * A0.conjugate() * AT)))
    return trace, det, rot, Kmax, worst_U


def turning(orb, N):
    """Total turning (degrees, continuous) of the binary axis 1-2 and of the line from the binary's centre to body 3."""
    m = np.array(orb["masses"], float)
    y = np.concatenate([np.array([float(x) for x in orb["positions"]]), np.array([float(x) for x in orb["velocities"]]),
                        [0.0]])
    T = float(orb["period"])
    h = T / N
    ang = []
    for _ in range(N + 1):
        q = y[0:6].reshape(3, 2)
        a = q[1] - q[0]
        cm = (m[0] * q[0] + m[1] * q[1]) / (m[0] + m[1])
        b = q[2] - cm
        ang.append((math.atan2(a[1], a[0]), math.atan2(b[1], b[0]), math.hypot(*a), math.hypot(*b)))
        k1 = deriv(y, m)
        k2 = deriv(y + 0.5 * h * k1, m)
        k3 = deriv(y + 0.5 * h * k2, m)
        k4 = deriv(y + h * k3, m)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    A = np.array(ang)
    t12 = math.degrees(np.unwrap(A[:, 0])[-1] - A[0, 0])
    t3 = math.degrees(np.unwrap(A[:, 1])[-1] - A[0, 1])
    return t12, t3, (A[:, 2].min(), A[:, 2].max()), (A[:, 3].min(), A[:, 3].max())


def main():
    orbs = json.load(open(os.path.join(ROOT, "verify", "certificates", "orbits_12.json"), encoding="utf-8"))["orbits"]
    print("masses          trace(2^14)  trace(2^15)  det      rotation/period  max K on orbit  "
          "binary axis turn  line to 3 turn  r12 range       |q3 - c12| range")
    for orb in orbs:
        t1, _, _, _, _ = run(orb, 2 ** 14)
        t2, det, rot, Kmax, wU = run(orb, 2 ** 15)
        a12, a3, r12, r3 = turning(orb, 2 ** 14)
        print(f"{str(tuple(orb['masses'])):15s} {t1:11.4f}  {t2:11.4f}  {det:.6f}  {rot:+9.2f} deg     {Kmax:+.3e}      "
              f"{a12:+8.1f} deg      {a3:+8.1f} deg    {r12[0]:.3f}-{r12[1]:.3f}     {r3[0]:.3f}-{r3[1]:.3f}"
              f"   (U check {wU:.0e})", flush=True)


if __name__ == "__main__":
    main()
