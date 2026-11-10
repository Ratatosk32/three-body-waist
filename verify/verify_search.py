"""Independent rerun of the reversible search behind Section 5: which masses have a symmetric class-(12) orbit.
Start: a syzygy with body 1 on the segment between bodies 2 and 3 (relative position s in (0, 1)), all velocities
perpendicular to the line, total momentum 0, angular momentum 0, energy 0, I = 1. Integrate to the first syzygy with
body 2 in the middle; the residual is the rate of change of the collinear ratio (p2 - p1)/(p3 - p1) there. A root
closes the orbit after twice that time (reversibility). All 2000 starts are integrated at once with fixed-step RK4 at
two step sizes in a regularised time (dt = s dtau, s = 1 / sum 1 / r_ij^2), so the step shrinks near collisions;
starts that come within 1e-5 of a binary collision are dropped. Sign changes of the residual between
neighbouring valid starts are refined and reported for both step sizes, at nineteen mass triples; each root is
compared with the start parameter of the certified orbit. Shares no code with the research tools.
    python verify/verify_search.py"""
import json
import math
import os

import numpy as np

PAIRS = ((0, 1), (1, 2), (2, 0))
CASES = [((1, 1, 0.40), True), ((1, 1, 0.50), True), ((1, 1, 0.52), True), ((1, 1, 0.53), True),
         ((1, 1, 0.54), True), ((1, 1, 0.55), True), ((1, 1, 0.56), True), ((1, 0.5, 0.30), True),
         ((1, 0.2, 0.15), True),
         ((1, 1, 0.57), False), ((1, 1, 0.58), False), ((1, 1, 0.59), False), ((1, 1, 0.60), False), ((1, 1, 0.65), False),
         ((1, 1, 0.75), False), ((1, 0.5, 0.45), False), ((1, 0.2, 0.19), False), ((1, 0.3, 0.5), False),
         ((0.5, 0.3, 1), False)]


def starts(ss, m):
    n = len(ss)
    x = np.vstack([ss, np.zeros(n), np.ones(n)])  # bodies 1, 2, 3 on a line
    x -= (m @ x) / m.sum()
    x /= np.sqrt(m @ (x * x))
    mx = m[:, None] * x
    v = np.cross(np.broadcast_to(m[:, None], x.shape).T, mx.T).T  # orthogonal to m (P = 0) and to m x (L = 0)
    U = sum(m[i] * m[j] / (x[i] - x[j]) ** 2 for i, j in PAIRS)
    v *= np.sqrt(2 * U / (m @ (v * v)))
    Y = np.zeros((12, n))
    Y[0:6:2] = x
    Y[7:12:2] = v
    return Y


def deriv(Y, m):
    q = Y[0:6].reshape(3, 2, -1)
    a = np.zeros_like(q)
    for i, j in PAIRS:
        d = q[j] - q[i]
        r2 = np.sum(d * d, 0)
        f = 2.0 * d / r2 ** 2
        a[i] += m[j] * f
        a[j] -= m[i] * f
    return np.vstack([Y[6:12], a.reshape(6, -1)])


def area(Y):
    q = Y[0:6].reshape(3, 2, -1)
    return (q[1, 0] - q[0, 0]) * (q[2, 1] - q[0, 1]) - (q[1, 1] - q[0, 1]) * (q[2, 0] - q[0, 0])


def residuals(ss, m, h, tmax=20.0):
    n = len(ss)
    m = np.asarray(m, float)
    Y = starts(ss, m)
    live = np.ones(n, bool)
    res = np.full(n, np.nan)
    prev = area(Y)
    Y = np.vstack([Y, np.zeros(n)])  # row 12: the physical time

    def f(Z):
        # regularised time: dt = s dtau with s = 1 / sum(1 / r_ij^2), about the smallest r_ij^2
        q = Z[0:6].reshape(3, 2, -1)
        s = 1.0 / sum(1.0 / np.sum((q[i] - q[j]) ** 2, 0) for i, j in PAIRS)
        return s * np.vstack([deriv(Z[0:12], m), np.ones(Z.shape[1])])

    steps = 0
    while live.any() and steps < 10 ** 6:
        Yl = Y[:, live]
        k1 = f(Yl)
        k2 = f(Yl + 0.5 * h * k1)
        k3 = f(Yl + 0.5 * h * k2)
        k4 = f(Yl + h * k3)
        Ynew = Yl + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        steps += 1
        idx = np.flatnonzero(live)
        q = Ynew[0:6].reshape(3, 2, -1)
        rmin = np.min([np.sqrt(np.sum((q[i] - q[j]) ** 2, 0)) for i, j in PAIRS], 0)
        a_new = area(Ynew)
        cross = (prev[idx] * a_new < 0) & (Ynew[12] > 1e-6)
        for kk in np.flatnonzero(cross):
            # linear interpolation to the syzygy
            lam = prev[idx[kk]] / (prev[idx[kk]] - a_new[kk])
            y = Yl[:, kk] + lam * (Ynew[:, kk] - Yl[:, kk])
            qq = y[0:6].reshape(3, 2)
            vv = y[6:12].reshape(3, 2)
            r = [np.linalg.norm(qq[1] - qq[2]), np.linalg.norm(qq[2] - qq[0]), np.linalg.norm(qq[0] - qq[1])]
            if int(np.argmax(r)) == 1:  # body 2 in the middle
                u = (qq[2] - qq[0]) / np.linalg.norm(qq[2] - qq[0])
                p, w = qq @ u, vv @ u
                res[idx[kk]] = ((w[1] - w[0]) * (p[2] - p[0]) - (p[1] - p[0]) * (w[2] - w[0])) / (p[2] - p[0]) ** 2
            live[idx[kk]] = False
        bad = (rmin < 1e-5) | (Ynew[12] > tmax)
        live[idx[bad & ~cross]] = False
        Y[:, idx] = Ynew
        prev[idx] = a_new
    return res


def is_root(a, b, ra, rb, m, h, levels=3, k=32):
    """Refine a sign change on [a, b]: a root of a continuous residual makes |residual| shrink with the bracket
    (by about k per level); a jump between two branches does not."""
    scale = min(abs(ra), abs(rb))
    for _ in range(levels):
        ss = np.linspace(a, b, k + 1)
        r = residuals(ss, m, h)
        ok = np.isfinite(r)
        idx = [i for i in range(k) if ok[i] and ok[i + 1] and r[i] * r[i + 1] < 0]
        if not idx:
            return False, None
        i = idx[0]
        a, b = ss[i], ss[i + 1]
        ra, rb = r[i], r[i + 1]
    return min(abs(ra), abs(rb)) < 1e-3 * scale, 0.5 * (a + b)


def scan(m, h, n=2000):
    m = np.array(m, float)
    ss = (np.arange(n) + 0.5) / n
    res = residuals(ss, m, h)
    ok = np.isfinite(res)
    roots, jumps = [], 0
    for i in range(n - 1):
        if ok[i] and ok[i + 1] and res[i] * res[i + 1] < 0:
            good, s0 = is_root(ss[i], ss[i + 1], res[i], res[i + 1], m, h)
            if good:
                roots.append(float(s0))
            else:
                jumps += 1
    return roots, int(ok.sum()), jumps


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    cert = {tuple(float(x) for x in o["masses"]): float(o["start_parameter_s"])
            for o in json.load(open(os.path.join(here, "certificates", "orbits_12.json"), encoding="utf-8"))["orbits"]}
    allok = True
    for m, expect in CASES:
        r1, v1, j1 = scan(m, 2e-3)
        r2, v2, j2 = scan(m, 1e-3)
        found = bool(r1) and bool(r2)
        agree = len(r1) == len(r2) and all(abs(a - b) < 1e-3 for a, b in zip(r1, r2))
        s_cert = cert.get(tuple(float(x) for x in m))
        match = s_cert is None or (len(r1) == 1 and abs(r1[0] - s_cert) < 1e-4)
        ok = (found == expect) and agree and match
        allok &= ok
        print(f"masses {str(m):17s} valid starts {v1}/{v2} of 2000; roots at s = "
              f"{', '.join(f'{x:.5f}' for x in r1) or 'none'} (half step: {', '.join(f'{x:.5f}' for x in r2) or 'none'}); "
              f"sign jumps between branches {j1}/{j2}; "
              f"{'certificate s = %.5f; ' % s_cert if s_cert else ''}expected {'a root' if expect else 'none'}: "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    print("ALL PASS" if allok else "SOME FAILED")


if __name__ == "__main__":
    main()
