"""Birth law near the threshold, masses (1, 1, m): the round loops about C12 have length
    l(rho) = c12 [1 - A2 rho^2 + A4 rho^4 + O(rho^6)]   (odd orders average out over the angle),
A2 = (1 - V0/(mu12 m1 m2)) / 8 changes sign at m*, A4 > 0 there. Minimising over rho gives
    c12 - min l = c12 A2^2 / (4 A4),
and with A2 = A2'(m*) (m - m*) + ..., the bound 2 pi - S > c12 A2'^2 / (4 A4) (m* - m)^2 (1 + o(1)) for the minimiser.
High precision (mpmath) with the trapezoid rule in the angle (spectral for periodic integrands).
    python tools/words_birth_a4.py"""
import mpmath as mp

mp.mp.dps = 40


def length(m3, rho, n=96):
    m1 = m2 = mp.mpf(1)
    M12, M = m1 + m2, m1 + m2 + m3
    mu12, mu3 = m1 * m2 / M12, m3 * M12 / M
    A = 1 / mp.sqrt(mu3)
    pairs = [(m1 * m2, 1 / (2 * mu12), (1 / (2 * mu12), 0))]
    for c, B in ((m1 * m3, m2 / (M12 * mp.sqrt(mu12))), (m2 * m3, -m1 / (M12 * mp.sqrt(mu12)))):
        pairs.append((c, (A * A + B * B) / 2, ((B * B - A * A) / 2, A * B)))
    tot = 0
    for k in range(n):
        chi = 2 * mp.pi * k / n
        # w = cos(rho) C12 + sin(rho) (cos chi e1 + sin chi e2), C12 = (-1, 0, 0), e1 = (0, 1, 0), e2 = (0, 0, 1)
        w = (-mp.cos(rho), mp.sin(rho) * mp.cos(chi), mp.sin(rho) * mp.sin(chi))
        U = sum(c / (a + b[0] * w[0] + b[1] * w[1]) for c, a, b in pairs)
        tot += mp.sqrt(2 * U) * mp.sin(rho) / 2
    return tot * 2 * mp.pi / n


def main():
    ms = (1 + mp.sqrt(65)) / 16
    c12 = 2 * mp.pi  # R^2 = 2 mu12 m1 m2 = 1
    # A4 at m*: (l / c12 - 1) / rho^4, Richardson in rho^2
    vals = [(length(ms, r) / c12 - 1) / r ** 4 for r in (mp.mpf("0.02"), mp.mpf("0.01"), mp.mpf("0.005"))]
    r1 = [(4 * vals[i + 1] - vals[i]) / 3 for i in range(2)]
    A4 = (16 * r1[1] - r1[0]) / 15
    # A2'(m*): A2 = (1 - V0 / (mu12 m1 m2)) / 8 = (1 - 2 V0) / 8 with V0 = m^2 * 4 / (2 + m)
    A2 = lambda m: (1 - 2 * 4 * m ** 2 / (2 + m)) / 8
    dA2 = mp.diff(A2, ms)
    C = c12 * dA2 ** 2 / (4 * A4)
    print(f"m* = {mp.nstr(ms, 12)}")
    print(f"A4(m*) = {mp.nstr(A4, 10)}   (Richardson stages {mp.nstr(vals[2], 10)}, {mp.nstr(r1[1], 10)})")
    print(f"A2'(m*) = {mp.nstr(dA2, 12)}   check A2(m*) = {mp.nstr(A2(ms), 5)}")
    print(f"birth constant C = c12 A2'^2 / (4 A4) = {mp.nstr(C, 8)}")
    # direct minimisation of the circle length below m*
    for eps in ("0.02", "0.01", "0.005", "0.0025", "0.00125"):
        m = ms - mp.mpf(eps)
        rho = mp.findroot(lambda r: mp.diff(lambda x: length(m, x, 48), r), mp.sqrt(-A2(m) / (2 * A4)) if A2(m) < 0
                          else mp.sqrt(A2(m) / (2 * A4)))
        gap = c12 - length(m, rho, 48)
        print(f"eps {eps}: rho_min {mp.nstr(rho, 6)}, (2 pi - min l) / eps^2 = {mp.nstr(gap / mp.mpf(eps) ** 2, 8)}")


if __name__ == "__main__":
    main()
