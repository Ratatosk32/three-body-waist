"""T0: a numerical map of the mass triangle for the class (12). For each mass triple, minimise the g-length of loops
around C12 (star-shaped in the stereographic chart centred at C12, log-radius given by a Fourier series) from several
starts, and record every local minimum that stays away from the end.
    a loop shorter than the cuff c12  =>  the class (12) is realised (Theorem 5 needs only L* < c12);
    a local minimum away from the end with length above c12  =>  a closed geodesic not predicted by kappa_2 (a surprise);
    every start sliding into the end  =>  no minimiser found.
Compared with the waist criterion kappa_2 > 0 at C12.
    python tools/words_T0.py N [out]    (barycentric grid with N steps per side)"""
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPTS, NMODE = 384, 12
TH = np.linspace(0, 2 * math.pi, NPTS, endpoint=False)
COS = np.array([np.cos(k * TH) for k in range(1, NMODE + 1)])
SIN = np.array([np.sin(k * TH) for k in range(1, NMODE + 1)])


def coeffs(m):
    m1, m2, m3 = m
    M12, M = m1 + m2, m1 + m2 + m3
    mu12, mu3 = m1 * m2 / M12, m3 * M12 / M
    out = [(m1 * m2, 1 / (2 * mu12), np.array([1 / (2 * mu12), 0.0, 0.0]))]
    A = 1 / math.sqrt(mu3)
    for c, B in ((m1 * m3, m2 / (M12 * math.sqrt(mu12))), (m2 * m3, -m1 / (M12 * math.sqrt(mu12)))):
        out.append((c, (A * A + B * B) / 2, np.array([(B * B - A * A) / 2, A * B, 0.0])))
    return out


KK = np.arange(1, NMODE + 1)[:, None]


def loop_xy(p):
    s = p[0] + p[1:NMODE + 1] @ COS + p[NMODE + 1:] @ SIN  # log of the chart radius
    r = np.exp(s)
    return r * np.cos(TH), r * np.sin(TH), r


def length(p, co, guard):
    x, y, r = loop_xy(p)
    ds = (-p[1:NMODE + 1] * KK.T) @ SIN + (p[NMODE + 1:] * KK.T) @ COS  # d(log r)/d theta, exact
    # the loop must keep C13 and C23 outside (class (12)): its radius on their rays below their chart radius
    for th_c, r_c in guard:
        s_c = p[0] + p[1:NMODE + 1] @ np.cos(KK[:, 0] * th_c) + p[NMODE + 1:] @ np.sin(KK[:, 0] * th_c)
        if math.exp(s_c) > 0.98 * r_c:
            return 1e9
    r2 = r * r
    den = 1 + r2
    # chart centred at C12 = (-1, 0, 0) with e1 = (0, 1, 0), e3 = (0, 0, 1); g0 = |dz|^2 / (1 + |z|^2)^2
    w = np.vstack([-(1 - r2) / den, 2 * x / den, 2 * y / den])
    U = sum(c / (a + b @ w) for c, a, b in co)
    integrand = np.sqrt(2 * U) / den * r * np.sqrt(1 + ds * ds)  # |dz| = r sqrt(1 + (log r)'^2) d theta
    return float(np.sum(integrand) * 2 * math.pi / NPTS)  # trapezoid rule, spectral for periodic integrands


def kappa2(m):
    m1, m2, m3 = m
    M = m1 + m2 + m3
    return -(m1 + m2) * (m3 ** 2 * (m1 + m2) ** 3 - m1 ** 2 * m2 ** 2 * M) / (4 * M * m1 ** 4 * m2 ** 4)


def survey(m):
    co = coeffs(m)
    m1, m2, _ = m
    c12 = 2 * math.pi * math.sqrt(2 * m1 * m2 / (m1 + m2) * m1 * m2)
    # the other collisions in the chart: angle and radius
    guard = []
    for _, _, b in co[1:]:
        c = -b / np.linalg.norm(b)
        z = np.array([c[1], c[2]]) / (1 - c[0])
        guard.append((math.atan2(z[1], z[0]), float(np.hypot(*z))))
    rmax = min(g[1] for g in guard)
    found = []
    for r0 in (0.15 * rmax, 0.45 * rmax, 0.8 * rmax):
        p0 = np.zeros(2 * NMODE + 1)
        p0[0] = math.log(r0)
        res = minimize(length, p0, args=(co, guard), method="BFGS", options={"gtol": 1e-8, "maxiter": 600})
        _, _, r = loop_xy(res.x)
        into_end = r.min() < 0.05 or not res.success
        found.append((res.fun, r.min(), r.max(), into_end, res.success))
    return c12, found


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "words", "out", "T0_map.tsv")
    rows = []
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        f.write("m1\tm2\tm3\tkappa2\tc12\tbest_L_minus_c\tbest_rmin\tbest_rmax\tn_local_minima_away\tmax_L_away_minus_c\n")
        for i in range(1, N):
            for j in range(1, N - i):
                k = N - i - j
                m = (i / N, j / N, k / N)
                c12, found = survey(m)
                away = [f_ for f_ in found if not f_[3]]
                best = min(found, key=lambda f_: f_[0])
                maxaway = max((f_[0] for f_ in away), default=float("nan"))
                f.write(f"{m[0]:.4f}\t{m[1]:.4f}\t{m[2]:.4f}\t{kappa2(m):+.4e}\t{c12:.6f}\t{best[0] - c12:+.3e}\t"
                        f"{best[1]:.3e}\t{best[2]:.3e}\t{len(away)}\t{maxaway - c12:+.3e}\n")
                f.flush()
                rows.append((m, kappa2(m), best[0] - c12, len(away)))
    agree = sum(1 for _, k2, gap, _ in rows if (k2 > 0) == (gap < -1e-9))
    print(f"{len(rows)} mass triples, {time.time() - t0:.0f}s; kappa_2 sign agrees with 'a loop shorter than the cuff' "
          f"at {agree}; written {out}")


if __name__ == "__main__":
    main()
