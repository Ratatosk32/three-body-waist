"""Exact check of the end-curvature coefficient at rational masses (sympy, exact arithmetic, no rounding):
for each mass triple, K is expanded in rho at the collision C_12 and the rho^2 coefficient is compared with
kappa_2 = -(m1 + m2) [m3^2 (m1 + m2)^3 - m1^2 m2^2 M] / (4 M m1^4 m2^4); the rho^0 and rho^1 coefficients must vanish.
The direction angle a stays symbolic, so independence of the direction is checked too.
    python verify/verify_kappa2_rational.py"""
import random

import sympy as sp

rho, a = sp.symbols("rho a", positive=True)


def check(m1, m2, m3):
    M12, M = m1 + m2, m1 + m2 + m3
    mu12, mu3 = m1 * m2 / M12, m3 * M12 / M
    A = 1 / sp.sqrt(mu3)
    w1, w2 = -sp.cos(rho), sp.sin(rho) * sp.cos(a)
    r12 = (1 + w1) / (2 * mu12)
    B13, B23 = m2 / (M12 * sp.sqrt(mu12)), -m1 / (M12 * sp.sqrt(mu12))
    r13 = A ** 2 * (1 - w1) / 2 + B13 ** 2 * (1 + w1) / 2 + A * B13 * w2
    r23 = A ** 2 * (1 - w1) / 2 + B23 ** 2 * (1 + w1) / 2 + A * B23 * w2
    U = m1 * m2 / r12 + m1 * m3 / r13 + m2 * m3 / r23
    F = sp.log(U)
    lap = sp.diff(sp.sin(rho) * sp.diff(F, rho), rho) / sp.sin(rho) + sp.diff(F, a, 2) / sp.sin(rho) ** 2
    K = (4 - 2 * lap) / (2 * U)
    ser = sp.series(K, rho, 0, 3).removeO()
    c = [sp.nsimplify(sp.simplify(ser.coeff(rho, k))) for k in range(3)]
    target = -(m1 + m2) * (m3 ** 2 * (m1 + m2) ** 3 - m1 ** 2 * m2 ** 2 * M) / (4 * M * m1 ** 4 * m2 ** 4)
    return c, sp.nsimplify(target)


random.seed(7)
cases = [(1, 1, 1), (1, 1, sp.Rational(1, 2)), (1, sp.Rational(1, 2), sp.Rational(3, 10)), (2, 1, 1)]
cases += [tuple(sp.Rational(random.randint(1, 40), random.randint(1, 20)) for _ in range(3)) for _ in range(6)]
allok = True
for m in cases:
    c, t = check(*[sp.nsimplify(x) for x in m])
    ok = c[0] == 0 and c[1] == 0 and sp.simplify(c[2] - t) == 0
    allok &= ok
    print(f"masses {m}: rho^0 {c[0]}, rho^1 {c[1]}, rho^2 {c[2]} vs formula {t}: {'PASS' if ok else 'FAIL'}", flush=True)
print("ALL PASS" if allok else "SOME FAILED")
