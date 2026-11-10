"""Check of the birth law (paper, Section 5): for masses (1, 1, m) the round loops about C12 have length
    l(rho) = c12 [1 - A2 rho^2 + A4 rho^4 + O(rho^6)],   A2 = (1 - V0 / (mu12 m1 m2)) / 8,
and c12 - min_rho l = C (m* - m)^2 (1 + o(1)) with C = c12 A2'(m*)^2 / (4 A4(m*)). Checks: A4(m*) and C against the
paper's values; the ratios (c12 - min l) / (m* - m)^2 rising towards C; and, for all nine certified orbits, that the
action S is below the shortest round loop about C12 (the orbit is the shortest loop of its class, so this must hold),
and the ratios (2 pi - S) / (m* - m)^2 quoted in the paper. High precision (mpmath), trapezoid rule in the angle.
    python verify/verify_birth.py"""
import json
import os

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
mp.mp.dps = 30


def length(m, rho, n=64):
    """g-length of the round loop of unit-sphere radius rho about C12 for g = 2U g0 (g0 = sphere of radius 1/2)."""
    m1, m2, m3 = (mp.mpf(x) for x in m)
    M12, M = m1 + m2, m1 + m2 + m3
    mu12, mu3 = m1 * m2 / M12, m3 * M12 / M
    A = 1 / mp.sqrt(mu3)
    pairs = [(m1 * m2, 1 / (2 * mu12), (1 / (2 * mu12), 0))]
    for c, B in ((m1 * m3, m2 / (M12 * mp.sqrt(mu12))), (m2 * m3, -m1 / (M12 * mp.sqrt(mu12)))):
        pairs.append((c, (A * A + B * B) / 2, ((B * B - A * A) / 2, A * B)))
    tot = 0
    for k in range(n):
        chi = 2 * mp.pi * k / n
        w = (-mp.cos(rho), mp.sin(rho) * mp.cos(chi), mp.sin(rho) * mp.sin(chi))  # C12 = (-1, 0, 0)
        U = sum(c / (a + b[0] * w[0] + b[1] * w[1]) for c, a, b in pairs)
        tot += mp.sqrt(2 * U) * mp.sin(rho) / 2
    return tot * 2 * mp.pi / n


def min_loop(m):
    # round loops are in the class (12) only while they separate C12 from C13 and C23: rho below their distance
    m1, m2, m3 = (mp.mpf(x) for x in m)
    M12, M = m1 + m2, m1 + m2 + m3
    mu12, mu3 = m1 * m2 / M12, m3 * M12 / M
    A = 1 / mp.sqrt(mu3)
    rmax = mp.inf
    for B in (m2 / (M12 * mp.sqrt(mu12)), -m1 / (M12 * mp.sqrt(mu12))):
        b = ((B * B - A * A) / 2, A * B)
        c = (-b[0] / mp.sqrt(b[0] ** 2 + b[1] ** 2), -b[1] / mp.sqrt(b[0] ** 2 + b[1] ** 2))
        rmax = min(rmax, mp.acos(-c[0]))  # distance from C12 = (-1, 0, 0)
    best = min((length(m, r), r) for r in mp.linspace(mp.mpf("0.02"), rmax - mp.mpf("0.05"), 80))
    a, b = max(best[1] - mp.mpf("0.04"), mp.mpf("0.001")), best[1] + mp.mpf("0.04")
    g = (mp.sqrt(5) - 1) / 2
    x1, x2 = b - g * (b - a), a + g * (b - a)
    f1, f2 = length(m, x1), length(m, x2)
    for _ in range(90):  # golden section
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - g * (b - a)
            f1 = length(m, x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + g * (b - a)
            f2 = length(m, x2)
    return min(f1, f2), (a + b) / 2


def main():
    ok_all = True
    ms = (1 + mp.sqrt(65)) / 16
    c12 = 2 * mp.pi
    vals = [(length((1, 1, ms), r, 96) / c12 - 1) / r ** 4 for r in (mp.mpf("0.02"), mp.mpf("0.01"), mp.mpf("0.005"))]
    r1 = [(4 * vals[i + 1] - vals[i]) / 3 for i in range(2)]
    A4 = (16 * r1[1] - r1[0]) / 15
    dA2 = mp.diff(lambda m: (1 - 2 * 4 * m ** 2 / (2 + m)) / 8, ms)  # V0 = 4 m^2 / (2 + m), mu12 m1 m2 = 1/2
    C = c12 * dA2 ** 2 / (4 * A4)
    ok = abs(A4 - mp.mpf("0.0068967")) < 1e-7 and abs(C - mp.mpf("35.12")) < 0.01
    ok_all &= ok
    print(f"A4(m*) = {mp.nstr(A4, 8)} (paper 0.0068967), C = {mp.nstr(C, 7)} (paper 35.12): {'PASS' if ok else 'FAIL'}")
    Cc = mp.pi * (4 + ms) ** 2 / (32 * ms ** 5)
    ok = abs(A4 - ms ** 3 / (4 * (2 + ms) ** 2)) < 1e-12 and abs(C - Cc) < 1e-8
    ok_all &= ok
    print(f"closed forms at m*: A4 = m*^3 / (4 (2 + m*)^2) = {mp.nstr(ms ** 3 / (4 * (2 + ms) ** 2), 12)}, "
          f"C = pi (4 + m*)^2 / (32 m*^5) = {mp.nstr(Cc, 10)}: {'PASS' if ok else 'FAIL'}")
    # A4 along the threshold curve of the end 12 (masses (1, t, m3(t))): numeric against the closed form
    for t in ("0.7", "0.4", "0.2", "0.05"):
        t = mp.mpf(t)
        m3 = (t ** 2 + mp.sqrt(t ** 4 + 4 * (1 + t) ** 4 * t ** 2)) / (2 * (1 + t) ** 3)
        Mt = 1 + t + m3
        closed = m3 ** 3 * (1 + t) ** 3 / (32 * Mt ** 2 * t ** 2)
        c = 2 * mp.pi * mp.sqrt(2 * t / (1 + t) * t)
        v = [(length((1, t, m3), r, 96) / c - 1) / r ** 4 for r in (mp.mpf("0.02"), mp.mpf("0.01"), mp.mpf("0.005"))]
        q = [(4 * v[i + 1] - v[i]) / 3 for i in range(2)]
        num = (16 * q[1] - q[0]) / 15
        ok = abs(num - closed) < 1e-9 * closed and closed > 0
        ok_all &= ok
        print(f"threshold point (1, {mp.nstr(t, 3)}, {mp.nstr(m3, 8)}): A4 numeric {mp.nstr(num, 10)}, closed form "
              f"m3^3 (m1 + m2)^3 / (32 M^2 m1^2 m2^2) = {mp.nstr(closed, 10)}: {'PASS' if ok else 'FAIL'}")
    prev = 0
    for eps, claim in (("0.02", 31.9), ("0.01", 33.4), ("0.005", 34.2), ("0.0025", 34.7), ("0.00125", 34.9)):
        L, rho = min_loop((1, 1, ms - mp.mpf(eps)))
        ratio = (c12 - L) / mp.mpf(eps) ** 2
        ok = abs(ratio - claim) < 0.06 and ratio > prev and ratio < C
        prev = ratio
        ok_all &= ok
        print(f"m* - m = {eps:8s}: shortest round loop at rho {mp.nstr(rho, 5)}, (c12 - min l) / (m* - m)^2 = "
              f"{mp.nstr(ratio, 6)} (paper {claim}): {'PASS' if ok else 'FAIL'}")
    orbs = json.load(open(os.path.join(HERE, "certificates", "orbits_12.json"), encoding="utf-8"))["orbits"]
    claims = {0.5: 28.00, 0.52: 29.48, 0.53: 30.36, 0.54: 31.37, 0.55: 32.57, 0.56: 34.01}
    for o in orbs:
        m = tuple(o["masses"])
        S = mp.mpf(o["action"])
        L, rho = min_loop(m)
        ok = S < L
        line = f"masses {str(m):15s}: S = {mp.nstr(S, 11)} < shortest round loop {mp.nstr(L, 11)} (rho {mp.nstr(rho, 4)})"
        if m[:2] == (1, 1) and m[2] in claims:
            ratio = (2 * mp.pi - S) / (ms - m[2]) ** 2
            ok &= abs(ratio - claims[m[2]]) < 0.006
            line += f"; (2 pi - S) / (m* - m)^2 = {mp.nstr(ratio, 6)} (paper {claims[m[2]]})"
        ok_all &= ok
        print(line + f": {'PASS' if ok else 'FAIL'}", flush=True)
    print("ALL PASS" if ok_all else "SOME FAILED")


if __name__ == "__main__":
    main()
