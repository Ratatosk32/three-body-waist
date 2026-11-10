"""Independent check of the two-letter orbit certificates (verify/certificates/orbits_12.json).
Shares no code with the research tools: numpy only, its own fixed-step integrator (classical Runge-Kutta 4 at N and
2N steps over the claimed period, Richardson-extrapolated), its own syzygy detection.
For each certificate it checks:
  1. the start: total momentum, angular momentum and energy (potential sum m_i m_j / r_ij^2) vanish, and I = 1;
  2. closure on the shape sphere after the claimed period: the three mutual distances, their time derivatives and the
     rate of the signed area
     return to their start values (a relative periodic orbit may rotate as a whole; L = 0 allows that);
  3. the action int sum m |v|^2 dt over the period equals the claimed one;
  4. the syzygy sequence over one period is exactly [1, 2] (body 1 in the middle at t = 0, then body 2), no other.
    python verify/verify_orbits.py [N]   (N = RK4 steps per period, default 2**17)"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = ((0, 1), (1, 2), (2, 0))


def deriv(y, m):
    q = y[0:6].reshape(3, 2)
    v = y[6:12].reshape(3, 2)
    a = np.zeros((3, 2))
    for i, j in PAIRS:
        d = q[j] - q[i]
        r2 = d @ d
        f = 2.0 * d / (r2 * r2)  # force of U = m_i m_j / r^2, divided by the masses below
        a[i] += m[j] * f
        a[j] -= m[i] * f
    return np.concatenate([y[6:12], a.ravel(), [float(np.sum(m * np.sum(v * v, 1)))]])


def area(y):
    q = y[0:6].reshape(3, 2)
    return (q[1, 0] - q[0, 0]) * (q[2, 1] - q[0, 1]) - (q[1, 1] - q[0, 1]) * (q[2, 0] - q[0, 0])


def middle(y):
    q = y[0:6].reshape(3, 2)
    r = [np.linalg.norm(q[1] - q[2]), np.linalg.norm(q[2] - q[0]), np.linalg.norm(q[0] - q[1])]
    return int(np.argmax(r)) + 1  # the body opposite the longest side is in the middle


def rk4(y0, m, T, N, track=False):
    h = T / N
    y = y0.copy()
    word, prev = [], area(y)
    rmin = math.inf
    for k in range(N):
        k1 = deriv(y, m)
        k2 = deriv(y + 0.5 * h * k1, m)
        k3 = deriv(y + 0.5 * h * k2, m)
        k4 = deriv(y + h * k3, m)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if track:
            a = area(y)
            if prev * a < 0 and k < N - 1:
                word.append(middle(y))
            prev = a
            q = y[0:6].reshape(3, 2)
            rmin = min(rmin, *(np.linalg.norm(q[i] - q[j]) for i, j in PAIRS))
    return y, word, rmin


def shape_state(y):
    q = y[0:6].reshape(3, 2)
    v = y[6:12].reshape(3, 2)
    out = []
    for i, j in PAIRS:
        d, w = q[j] - q[i], v[j] - v[i]
        r = math.sqrt(d @ d)
        out += [r, float(d @ w) / r]
    return np.array(out)


def area_rate(y):
    """Time derivative of the signed area: with the distances and their rates it fixes the shape state completely
    (the distances alone do not see on which side of the collinear shapes the triangle moves)."""
    q = y[0:6].reshape(3, 2)
    v = y[6:12].reshape(3, 2)
    a, b, da, db = q[1] - q[0], q[2] - q[0], v[1] - v[0], v[2] - v[0]
    return float(da[0] * b[1] - da[1] * b[0] + a[0] * db[1] - a[1] * db[0])


def invariants(y, m):
    q = y[0:6].reshape(3, 2)
    v = y[6:12].reshape(3, 2)
    P = np.sum(m[:, None] * v, 0)
    L = float(np.sum(m * (q[:, 0] * v[:, 1] - q[:, 1] * v[:, 0])))
    K = 0.5 * float(np.sum(m * np.sum(v * v, 1)))
    U = sum(m[i] * m[j] / float((q[i] - q[j]) @ (q[i] - q[j])) for i, j in PAIRS)
    return float(np.max(np.abs(P))), abs(L), abs(K - U) / U, abs(float(np.sum(m * np.sum(q * q, 1))) - 1)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2 ** 17
    cert = json.load(open(os.path.join(HERE, "certificates", "orbits_12.json"), encoding="utf-8"))
    ok_all = True
    print(f"{'masses':18s} {'S claimed':>14s} {'S checked':>14s} {'closure':>9s} {'word':>8s} {'P, L, E':>22s}  verdict")
    for c in cert["orbits"]:
        m = np.array(c["masses"], float)
        y0 = np.array([float(v) for v in c["positions"]] + [float(v) for v in c["velocities"]] + [0.0])
        T, S = float(c["period"]), float(c["action"])
        P, L, E, dI = invariants(y0, m)
        y1, word, rmin = rk4(y0, m, T, N, track=True)
        y2, _, _ = rk4(y0, m, T, 2 * N)
        y = y2 + (y2 - y1) / 15.0  # Richardson for a fourth-order method
        err = float(np.max(np.abs(y2 - y1)))
        close = float(np.max(np.abs(shape_state(y) - shape_state(y0))))
        close = max(close, abs(area_rate(y) - area_rate(y0)))  # the signed area rate too
        dS = abs(y[12] - S)
        seq = [1] + word
        ok = (P < 1e-12 and L < 1e-12 and E < 1e-12 and dI < 1e-12 and close < 1e-7 and dS < 1e-7 * S
              and seq == [1, 2])
        ok_all &= ok
        print(f"{str(tuple(c['masses'])):18s} {S:14.10f} {y[12]:14.10f} {close:9.1e} {''.join(map(str, seq)):>8s} "
              f"{P:7.0e},{L:7.0e},{E:7.0e}  {'PASS' if ok else 'FAIL'}  (I - 1 {dI:.0e}, RK4 step-halving difference {err:.0e}, "
              f"min pair distance {rmin:.3f})", flush=True)
    print("ALL PASS" if ok_all else "SOME FAILED")


if __name__ == "__main__":
    main()
