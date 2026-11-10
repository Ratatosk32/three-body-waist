"""Step (II'') of theorem G (proofs/waist_theorem.md): a chain of nested convex leaves sigma_0 > sigma_1 > ... >
sigma_N (curve-shortening leaves, words/out/csf/chain_<m>.npy; the last inside the cap where round circles satisfy
condition C). For each k, the inward normals of sigma_k with the Jacobi field of its parallels (G'' = -K G, G(0) = 1,
G'(0) = -k0) must cross sigma_{k+1} while G > 0 and G' < 0. Then lemma F applies piecewise and excludes a closed
geodesic of the class (12) inside sigma_0.
    python -I tools/words_chain_check.py m1,m2,m3 [k0 k1]"""
import math
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import words_csf as F  # noqa: E402
import words_sigma0_normals as S  # noqa: E402


def inside_many(P, poly):
    """Even-odd ray casting for many points P (2, n) at once."""
    x, y = P
    res = np.zeros(P.shape[1], bool)
    xs, ys = poly[0], poly[1]
    for i in range(poly.shape[1]):
        x1, y1, x2, y2 = xs[i - 1], ys[i - 1], xs[i], ys[i]
        if y1 == y2:
            continue
        res ^= ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
    return res


R_END = 0.05  # unit-sphere radius of the collision caps (K < 0 inside them, lemma K)


def main():
    m = tuple(float(v) for v in sys.argv[1].split(","))
    chain = np.load(os.path.join(ROOT, "words/out/csf", "chain_" + ",".join(f"{x:g}" for x in m) + ".npy"))
    print(f"m {m}: chain of {len(chain)} leaves", flush=True)

    def rhs(t, y):
        zz, v, G, Gp = y[0:2], y[2:4], y[4], y[5]
        ps, gp = F.psi_and_grad(zz.reshape(2, 1), m)
        gp = gp[:, 0]
        acc = -2 * (gp @ v) * v + (v @ v) * gp
        return np.concatenate([v, acc, [Gp, -S.K_at(zz, m) * G]])

    def gp_zero(t, y):
        return y[5]
    gp_zero.terminal, gp_zero.direction = True, 1
    import words_curv_exact as C
    cols = C.collisions(m)

    def cap(t, y):  # stop on entering a collision cap (the 12-cap lies inside every leaf: counts as passing)
        w = F.inv_stereo(y[0:2].reshape(2, 1))[:, 0]
        return min(math.acos(min(1.0, float(w @ c))) for c in cols) - R_END
    cap.terminal, cap.direction = True, -1

    bad_pairs = 0
    k0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # optional pair range [k0, k1) for parallel runs
    k1 = int(sys.argv[3]) if len(sys.argv) > 3 else len(chain) - 1
    for k in range(k0, min(k1, len(chain) - 1)):
        z, nxt = chain[k], chain[k + 1]
        psi, gpsi = F.psi_and_grad(z, m)
        nrm, kflat, hm = F.step_geom(z)
        kg = np.exp(-psi) * (kflat - np.sum(gpsi * nrm, 0))
        fails, margin = 0, np.inf
        reasons = {}
        for i in range(0, z.shape[1], 2):
            v0 = nrm[:, i] * math.exp(-psi[i])
            y0 = np.concatenate([z[:, i], v0, [1.0, -kg[i]]])
            sol = solve_ivp(rhs, (0, 5.0), y0, method="DOP853", rtol=1e-9, atol=1e-11, events=[gp_zero, cap],
                            dense_output=True)
            ts = np.linspace(0, sol.t[-1], 400)
            Y = sol.sol(ts)
            ins = inside_many(Y[0:2], nxt)
            if sol.t_events[1].size:  # entered a collision cap: inside the next leaf only if it is the 12-cap
                w = F.inv_stereo(Y[0:2, -1:])[:, 0]
                if math.acos(min(1.0, float(w @ cols[0]))) < 2 * R_END:
                    ins[-1] = True
            hit = int(np.argmax(ins)) if ins.any() else None
            if hit is None or Y[4, :hit + 1].min() <= 0 or Y[5, :hit + 1].max() >= 0:
                fails += 1
                why = ("other cap" if sol.t_events[1].size and hit is None else
                       "G' >= 0 first" if sol.t_events[0].size else "did not reach the next leaf")
                reasons[why] = reasons.get(why, 0) + 1
            else:
                margin = min(margin, -float(Y[5, :hit + 1].max()))
        bad_pairs += fails > 0
        print(f"  leaf {k} -> {k + 1}: k_g min {kg.min():+.4f}; normals failing {fails} of {z.shape[1] // 2}; "
              f"min |G'| before the next leaf {margin:.3e}" + (f"; failures {reasons}" if reasons else ""), flush=True)
    print(f"(II'') {'HOLDS' if bad_pairs == 0 else f'FAILS on {bad_pairs} pairs'}", flush=True)


if __name__ == "__main__":
    main()
