"""Independent check of the waist threshold (no research code imported; sympy and numpy only).
Part A (symbolic): Montgomery's end-curvature criterion at the 12 collision (his 2005 paper, eqs. 7.12-7.13, evaluated
at s3 -> 0, s1 = s2 = 3/2 with p_i = m_j m_k): kappa = (p1 + p2)^(3/2) / sqrt(p3) - sqrt(p1 p2 + p1 p3 + p2 p3).
  kappa = 0  <=>  (p1 + p2)^3 = p3 (p1 p2 + p1 p3 + p2 p3)  <=>  m3^2 (m1 + m2)^3 = m1^2 m2^2 M;
  for masses (1, 1, m): 8 m^2 - m - 2 = 0, positive root (1 + sqrt 65) / 16.
Part B (numerical, configuration space, no shape-sphere coordinates): the Jacobi-Maupertuis length (metric
sqrt(2U) |dq| on horizontal, I-preserving vectors, I = sum m |q - cm|^2 = 1, U = sum m_i m_j / r_ij^2) of the loop in
which the binary 12 at fixed separation eps turns once relative to body 3; compared with the cylinder's circumference
c12 = 2 pi sqrt(2 mu12 m1 m2). Waist  <=>  the loop is shorter than c12 for small eps.
    python verify/verify_threshold.py"""
import math

import numpy as np
import sympy as sp


def part_a():
    m1, m2, m3, m = sp.symbols("m1 m2 m3 m", positive=True)
    p1, p2, p3 = m2 * m3, m1 * m3, m1 * m2
    M = m1 + m2 + m3
    lhs = sp.expand((p1 + p2) ** 3 - p3 * (p1 * p2 + p1 * p3 + p2 * p3))
    rhs = sp.expand(m3 * (m3 ** 2 * (m1 + m2) ** 3 - m1 ** 2 * m2 ** 2 * M))
    ok1 = sp.simplify(lhs - rhs) == 0
    eq = sp.factor((m3 ** 2 * (m1 + m2) ** 3 - m1 ** 2 * m2 ** 2 * M).subs({m1: 1, m2: 1, m3: m}))
    roots = sp.solve(sp.Eq(eq, 0), m)
    ok2 = sp.simplify(eq - (8 * m ** 2 - m - 2)) == 0 and any(sp.simplify(r - (1 + sp.sqrt(65)) / 16) == 0 for r in roots)
    mstar = float((1 + sp.sqrt(65)) / 16)
    print(f"A1 (p1+p2)^3 - p3 (p1p2+p1p3+p2p3) = m3 [m3^2 (m1+m2)^3 - m1^2 m2^2 M]: {'PASS' if ok1 else 'FAIL'}")
    print(f"A2 masses (1,1,m): criterion = {eq}; positive root (1+sqrt 65)/16 = {mstar:.10f}: {'PASS' if ok2 else 'FAIL'}")
    return ok1 and ok2, mstar


def loop_length(mass, eps, n=20000):
    m1, m2, m3 = mass
    M12, M = m1 + m2, m1 + m2 + m3
    mu12, mu3 = m1 * m2 / M12, m3 * M12 / M
    D = math.sqrt((1 - mu12 * eps ** 2) / mu3)  # distance binary centre - body 3, so that I = 1
    m = np.array(mass, float)
    chi = (np.arange(n) + 0.5) * 2 * math.pi / n
    e = np.stack([np.cos(chi), np.sin(chi)])
    de = np.stack([-np.sin(chi), np.cos(chi)])
    cb = np.array([-m3 / M * D, 0.0])[:, None]  # binary centre; body 3 at cb + (D, 0); centre of mass at 0
    q = np.stack([cb - m2 / M12 * eps * e, cb + m1 / M12 * eps * e, cb + np.array([D, 0.0])[:, None] + 0 * e])
    dq = np.stack([-m2 / M12 * eps * de, m1 / M12 * eps * de, 0 * de])
    # remove the rotational part: omega = L / I with L = sum m q x dq, I = sum m |q|^2 (= 1)
    I = np.sum(m[:, None] * np.sum(q * q, 1), 0)
    L = np.sum(m[:, None] * (q[:, 0] * dq[:, 1] - q[:, 1] * dq[:, 0]), 0)
    om = L / I
    dqh = dq - om * np.stack([-q[:, 1], q[:, 0]], axis=1)
    norm = np.sqrt(np.sum(m[:, None] * np.sum(dqh * dqh, 1), 0))
    U = sum(m[i] * m[j] / np.sum((q[i] - q[j]) ** 2, 0) for i, j in ((0, 1), (1, 2), (2, 0)))
    return float(np.sum(np.sqrt(2 * U) * norm) * 2 * math.pi / n), max(abs(I - 1))


def part_b(mstar):
    ok = True
    for mass in [(1, 1, x) for x in (0.4, 0.5, 0.56, 0.565, 0.566, 0.567, 0.57, 0.6, 1.0, 2.0)] + [
            (1, 0.5, 0.30), (1, 0.5, 0.45), (1, 0.2, 0.15), (1, 0.2, 0.19)]:
        m1, m2, m3 = mass
        c12 = 2 * math.pi * math.sqrt(2 * (m1 * m2 / (m1 + m2)) * m1 * m2)
        crit = m3 ** 2 * (m1 + m2) ** 3 - m1 ** 2 * m2 ** 2 * (m1 + m2 + m3)
        d = []
        for eps in (0.02, 0.01, 0.005):
            L, Ierr = loop_length(mass, eps)
            d.append(L - c12)
        waist = d[-1] < 0
        expect = crit < 0
        scale = d[0] / d[1]
        good = waist == expect and 3.5 < scale < 4.5  # l - c ~ eps^2 near the collision
        ok &= good
        print(f"B {str(mass):18s} criterion {crit:+.4f}  l - c12 at eps 0.02/0.01/0.005: "
              f"{d[0]:+.3e} {d[1]:+.3e} {d[2]:+.3e} (ratio {scale:.2f})  waist {waist}  {'PASS' if good else 'FAIL'}")
    return ok


if __name__ == "__main__":
    a, mstar = part_a()
    b = part_b(mstar)
    print("ALL PASS" if a and b else "SOME FAILED")
