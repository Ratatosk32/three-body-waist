"""Replacing step (II) of theorem G (proofs/waist_theorem.md): inward parallels of the convex loop sigma_0.
sigma_0 = the first everywhere-convex curve-shortening leaf (words/out/csf/sigma0_<m>.npy, chart of words_csf.py:
stereographic from the antipode of C_12, so C_12 is z = 0, |z| = tan(rho/2) with rho the unit distance to C_12).
From each point of sigma_0 a geodesic leaves along the inward unit normal; along it the Jacobi field of the parallel
family G'' = -K G, G(0) = 1, G'(0) = -k0 (k0 > 0 the geodesic curvature of sigma_0 towards C_12).
Condition (II'): every normal reaches the cap {rho < rho_1 - margin}, where the round circles about C_12 satisfy
condition C (words_leaf_cond.py), with G > 0 and G' < 0 all the way. Then the parallels (convex towards C_12) foliate
the ring between sigma_0 and the cap, overlapping the circles' foliation inside the cap, and lemma F excludes a closed
geodesic of the class (12) inside sigma_0.
    python -I tools/words_sigma0_normals.py m1,m2,m3 rho_1"""
import math
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import words_csf as F  # noqa: E402
import words_curv_exact as C  # noqa: E402

MARGIN = 0.02


def K_at(z, m):
    w = F.inv_stereo(z.reshape(2, 1))
    _, K = C.U_K(w, m)
    return float(K[0])


def main():
    m = tuple(float(v) for v in sys.argv[1].split(","))
    rho1 = float(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "sigma0"
    rcap = math.tan((rho1 - MARGIN) / 2)  # chart radius of the cap
    z = np.load(os.path.join(ROOT, "words/out/csf", tag + "_" + ",".join(f"{x:g}" for x in m) + ".npy"))
    # geodesic curvature of sigma_0 towards C_12 (the loop runs counterclockwise round 0: the left normal points in)
    psi, gpsi = F.psi_and_grad(z, m)
    nrm, kflat, hm = F.step_geom(z)
    kg = np.exp(-psi) * (kflat - np.sum(gpsi * nrm, 0))
    print(f"m {m}: sigma_0 has {z.shape[1]} points, k_g min {kg.min():+.4f} max {kg.max():+.4f}; cap rho < {rho1 - MARGIN:.3f}"
          f" (chart radius {rcap:.4f})", flush=True)

    def rhs(t, y):
        zz, v, G, Gp = y[0:2], y[2:4], y[4], y[5]
        ps, gp = F.psi_and_grad(zz.reshape(2, 1), m)
        gp = gp[:, 0]
        acc = -2 * (gp @ v) * v + (v @ v) * gp
        return np.concatenate([v, acc, [Gp, -K_at(zz, m) * G]])

    def incap(t, y):
        return np.hypot(y[0], y[1]) - rcap
    incap.terminal, incap.direction = True, -1

    def gp_zero(t, y):
        return y[5]
    gp_zero.terminal, gp_zero.direction = True, 1

    worst, fails = None, 0
    for i in range(z.shape[1]):
        v0 = nrm[:, i] * math.exp(-psi[i])  # unit g-speed along the inward normal
        y0 = np.concatenate([z[:, i], v0, [1.0, -kg[i]]])
        sol = solve_ivp(rhs, (0, 30.0), y0, method="DOP853", rtol=1e-9, atol=1e-11, events=[incap, gp_zero])
        reached = sol.t_events[0].size > 0
        G, Gp = sol.y[4], sol.y[5]
        ok = reached and G.min() > 0 and Gp.max() < 0
        fails += not ok
        score = -float(Gp.max())
        if worst is None or score < worst[0]:
            worst = (score, i, float(G.min()), float(Gp.max()), reached, float(sol.t[-1]))
    print(f"  normals failing (II'): {fails} of {z.shape[1]}; tightest: point {worst[1]}, max G' {worst[3]:+.3e}, "
          f"min G {worst[2]:.4f}, reached the cap {worst[4]}, length {worst[5]:.3f}", flush=True)
    print("  (II') " + ("HOLDS" if fails == 0 else "FAILS"), flush=True)


if __name__ == "__main__":
    main()
