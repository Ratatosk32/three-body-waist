"""Data files for the paper's figures (paper/fig/*.dat), from the certified orbits and closed forms.
    python tools/words_figures.py"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from verify_orbits import deriv  # noqa: E402
from words_orbit_stability import hopf  # noqa: E402

OUT = os.path.join(ROOT, "paper", "fig")
MSTAR = (1 + math.sqrt(65)) / 16


def orbit(orb, N=4096, periods=1):
    m = np.array(orb["masses"], float)
    y = np.concatenate([[float(x) for x in orb["positions"]], [float(x) for x in orb["velocities"]], [0.0]])
    T = float(orb["period"])
    h = T / N
    Q = [y[0:6].copy()]
    for _ in range(N * periods):
        k1 = deriv(y, m)
        k2 = deriv(y + 0.5 * h * k1, m)
        k3 = deriv(y + 0.5 * h * k2, m)
        k4 = deriv(y + h * k3, m)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Q.append(y[0:6].copy())
    return m, np.array(Q)


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(" ".join(f"{x:.6g}" for x in r) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    orbs = json.load(open(os.path.join(ROOT, "verify", "certificates", "orbits_12.json"), encoding="utf-8"))["orbits"]
    by = {tuple(o["masses"]): o for o in orbs}
    # (a) the orbits on the shape sphere, chart z = (w2, w3) / (1 - w1) about C12 = (-1, 0, 0) at z = 0
    for key in ((1, 1, 0.4), (1, 1, 0.5), (1, 1, 0.53), (1, 1, 0.56)):
        m, Q = orbit(by[key], 2048)
        rows = []
        for q in Q:
            w = hopf(q.reshape(3, 2), m)
            rows.append(((w[1]) / (1 - w[0]), w[2] / (1 - w[0])))
        write(f"shape_{key[2]:.2f}.dat", "x y", rows)
    # (b) the bodies in the plane over three periods
    for key, tag in (((1, 1, 0.4), "040"), ((1, 1, 0.56), "056"), ((1, 0.2, 0.15), "asym")):
        m, Q = orbit(by[key], 4096, 3)
        write(f"plane_{tag}.dat", "x1 y1 x2 y2 x3 y3", [tuple(q) for q in Q[::8]])
    # (c) the birth: 2 pi - S against m, and the asymptote 35.12 (m* - m)^2
    rows = [(o["masses"][2], 2 * math.pi - float(o["action"])) for o in orbs if o["masses"][:2] == [1, 1]]
    write("birth_orbits.dat", "m gap", sorted(rows))
    write("birth_asym.dat", "m gap", [(m, 35.120765 * (MSTAR - m) ** 2) for m in np.linspace(0.38, MSTAR, 60)])
    # (d) the mass triangle (barycentric, M = 1): the waist region of the end ij is m_k^2 (1 - m_k)^3 < m_i^2 m_j^2,
    # i.e. m_i m_j > m_k (1 - m_k)^(3/2): between the two roots of x^2 - (1 - m_k) x + m_k (1 - m_k)^1.5 = 0
    A, B, C = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, math.sqrt(3) / 2])
    corners = {0: A, 1: B, 2: C}
    for (i, j, k) in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        lo, hi = [], []
        mk_max = (-1 + math.sqrt(1 + 64)) / 32  # sqrt(1 - mk) = 4 mk
        for mk in np.linspace(0, mk_max, 200):
            s = 1 - mk
            disc = max(s * s - 4 * mk * s ** 1.5, 0.0)
            x1, x2 = (s - math.sqrt(disc)) / 2, (s + math.sqrt(disc)) / 2
            for x, lst in ((x1, lo), (x2, hi)):
                bary = {i: x, j: s - x, k: mk}
                lst.append(sum(bary[t] * corners[t] for t in range(3)))
        poly = lo + hi[::-1] + [lo[0]]
        write(f"waist_{i + 1}{j + 1}.dat", "x y", [tuple(p) for p in poly])
    write("triangle.dat", "x y", [tuple(A), tuple(B), tuple(C), tuple(A)])
    # the slice (1, 1, m) of the paper's examples: m from 0 to 2
    write("slice.dat", "x y", [tuple((1 * A + 1 * B + mm * C) / (2 + mm)) for mm in np.linspace(0, 3, 50)])
    star = (A + B + MSTAR * C) / (2 + MSTAR)
    write("mstar.dat", "x y", [tuple(star)])
    print("written", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()


def birth_ratios():
    """(2 pi - S)/(m* - m)^2 of the orbits and of the shortest round loops, against m* - m (for the birth figure)."""
    import mpmath as mp
    from verify_birth import min_loop
    orbs = json.load(open(os.path.join(ROOT, "verify", "certificates", "orbits_12.json"), encoding="utf-8"))["orbits"]
    rows = [(MSTAR - o["masses"][2], (2 * math.pi - float(o["action"])) / (MSTAR - o["masses"][2]) ** 2)
            for o in orbs if o["masses"][:2] == [1, 1] and o["masses"][2] >= 0.5]
    write("birth_ratio_orbits.dat", "eps ratio", sorted(rows))
    ms = (1 + mp.sqrt(65)) / 16
    loops = []
    for eps in (0.08, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125):
        L, _ = min_loop((1, 1, ms - mp.mpf(eps)))
        loops.append((eps, float((2 * mp.pi - L) / mp.mpf(eps) ** 2)))
    write("birth_ratio_loops.dat", "eps ratio", loops)
    print("birth ratios written")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "ratios":
    birth_ratios()
