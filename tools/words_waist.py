"""A closed orbit in the class (12) at a = 2, E = 0, L = 0, masses (1, 1, M).
Start: a syzygy with body 1 between 2 and 3 on the x-axis (x2 = 0, x1 = s, x3 = 1, then centred and scaled to I = 1),
velocities along y only (P = 0, L = 0 leave one direction, E = 0 fixes its size). Integrate to the first later syzygy;
if body 2 is in the middle there, the residual is the along-line shape velocity d/dt[(p2 - p1) / (p3 - p1)]; its
roots in s give reversible orbits of period 2 t. Kept: full word 12, closure checked over the period.
    python -I tools/words_waist.py M [M ...]  ->  words/out/a2/waist.txt (appended)"""
import math
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "words", "tools"))
import topology as topo  # noqa: E402
import words_alpha as W  # noqa: E402

PAIRS = ((0, 1), (1, 2), (2, 0))


def rhs_m(mass):
    def rhs(t, y):
        q = y[:6].reshape(3, 2)
        acc = np.zeros((3, 2))
        for i, j in PAIRS:
            d = q[j] - q[i]
            f = 2 * d / (d @ d) ** 2
            acc[i] += mass[j] * f
            acc[j] -= mass[i] * f
        v = y[6:12].reshape(3, 2)
        return np.concatenate([y[6:12], acc.ravel(), [float(np.sum(mass * np.sum(v * v, 1)))]])
    return rhs


def start(s, mass):
    x = np.array([s, 0.0, 1.0])  # bodies 1, 2, 3
    x -= np.sum(mass * x) / np.sum(mass)
    x /= math.sqrt(np.sum(mass * x * x))
    n = np.cross(mass, mass * x)  # P_y = 0 and L = 0 (was [1, 1, m3]: the same for masses (1, 1, m3), wrong otherwise)
    U = sum(mass[i] * mass[j] / (x[i] - x[j]) ** 2 for i, j in PAIRS)
    n *= math.sqrt(2 * U / np.sum(mass * n * n))
    y = np.zeros(13)
    y[0:6:2] = x
    y[7:12:2] = n
    return y


def coll(t, y):
    q = y[:6].reshape(3, 2)
    return min(np.linalg.norm(q[i] - q[j]) for i, j in PAIRS) - 1e-5
coll.terminal = True


def shape_state(y):
    """Mutual distances and their time derivatives: the state modulo rotation (change 1)."""
    q = y[:6].reshape(3, 2)
    v = y[6:12].reshape(3, 2)
    out = []
    for i, j in PAIRS:
        d, w = q[j] - q[i], v[j] - v[i]
        r = float(np.linalg.norm(d))
        out += [r, float(d @ w) / r]
    return np.array(out)


def first_syzygy(s, mass, tmax=60.0):
    sol = solve_ivp(rhs_m(mass), (0, tmax), start(s, mass), method="DOP853", rtol=1e-11, atol=1e-12,
                    events=[W.area, coll])
    for t, y in zip(sol.t_events[0], sol.y_events[0]):
        if t > 1e-6:
            return t, y
    return None


def resid(s, mass):
    e = first_syzygy(s, mass)
    if e is None or W.middle(e[1]) != 1:
        return np.nan
    q = e[1][:6].reshape(3, 2)
    v = e[1][6:12].reshape(3, 2)
    u = (q[2] - q[0]) / np.linalg.norm(q[2] - q[0])
    p, w = q @ u, v @ u
    return ((w[1] - w[0]) * (p[2] - p[0]) - (p[1] - p[0]) * (w[2] - w[0])) / (p[2] - p[0]) ** 2


def full(s, mass):
    t, _ = first_syzygy(s, mass)
    y0 = start(s, mass)
    sol = solve_ivp(rhs_m(mass), (0, 2 * t), y0, method="DOP853", rtol=1e-12, atol=1e-13, events=W.area)
    eps = 1e-6 * t
    seq = [W.middle(y) + 1 for tt, y in zip(sol.t_events[0], sol.y_events[0]) if eps < tt < 2 * t - eps]
    seq = [1] + seq
    word = "".join(map(str, topo.reduce_cyclic(seq)))
    yT = sol.y[:, -1]
    close = float(np.max(np.abs(shape_state(yT) - shape_state(y0))))  # change 1: on the shape
    rmin = min(float(np.min(np.hypot(*(sol.y[2 * i:2 * i + 2] - sol.y[2 * j:2 * j + 2])))) for i, j in PAIRS)
    return 2 * t, yT[12], word, close, rmin


def main():
    out = []
    for arg in sys.argv[1:]:  # m3 (masses 1, 1, m3) or m1,m2,m3 (16.5)
        mass = np.array([float(x) for x in arg.split(",")]) if "," in arg else np.array([1.0, 1.0, float(arg)])
        M = arg if "," in arg else float(arg)  # label as before for the (1, 1, m3) slice
        ss = (np.arange(2000) + 0.5) / 2000
        r = np.array([resid(s, mass) for s in ss])
        found = []
        for i in range(len(ss) - 1):
            if np.isfinite(r[i]) and np.isfinite(r[i + 1]) and r[i] * r[i + 1] < 0:
                s0 = brentq(lambda s: resid(s, mass), ss[i], ss[i + 1], xtol=1e-14)
                T, S, word, close, rmin = full(s0, mass)
                found.append((s0, T, S, word, close, rmin))
        valid = np.isfinite(r).sum()
        out.append(f"m3 = {M}: starts reaching a body-2 syzygy {valid} of {len(ss)}; roots {len(found)}")
        for s0, T, S, word, close, rmin in found:
            out.append(f"  s {s0:.10f}  T {T:.6f}  S {S:.6f}  word {word}  closure {close:.1e}  min pair distance {rmin:.3e}")
        print(out[-1 - len(found)], *out[len(out) - len(found):], sep="\n", flush=True)
    open(os.path.join(ROOT, "words/out/a2/waist.txt"), "a", encoding="utf-8").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
