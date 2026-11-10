"""Route A for step (III) (proofs/waist_theorem.md, lemma F3): Fermi coordinates about an equator arc.
Chart: stereographic projection of the shape sphere from -C_a (C_a a collision at an end of the arc), oriented so the
collinear circle is the real axis and the upper hemisphere (w3 > 0) the upper half-plane:
z = (w . e1, w . e3) / (1 + w . c),  c = C_a, e3 = (0, 0, 1), e1 = e3 x c.
Metric g = e^(2 psi) |dz|^2, psi = (1/2) log 2U - log(1 + |z|^2). Normal geodesics leave the arc's points into the
upper half-plane at unit g-speed; along each the Jacobi field G'' = -K G, G(0) = 1, G'(0) = 0 (K exact,
words_curv_exact). Lemma F3: if G > 0 and G' > 0 along every normal until it leaves the hemisphere, no geodesic
leaves the arc into the hemisphere and returns to it (no stutter on that arc).
    python -I tools/words_fermi.py m1,m2,m3 [ns]"""
import math
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import words_curv_exact as C  # noqa: E402


class Chart:
    def __init__(self, c):
        self.c = np.asarray(c, float)
        self.e3 = np.array([0.0, 0.0, 1.0])
        self.e1 = np.cross(self.e3, self.c)

    def to_z(self, w):
        return np.array([w @ self.e1, w @ self.e3]) / (1 + w @ self.c)

    def to_w(self, z):
        r2 = z @ z
        return (2 * z[0] * self.e1 + 2 * z[1] * self.e3 + (1 - r2) * self.c) / (1 + r2)


def make_rhs(m, ch):
    coeffs = C.coeffs(m)

    def geo(z):
        w = ch.to_w(z)
        U, gU = 0.0, np.zeros(3)
        for cc, a, b in coeffs:
            g = a + b @ w
            U += cc / g
            gU -= cc * b / g ** 2
        r2 = z @ z
        # dw/dz (3 x 2)
        J = np.zeros((3, 2))
        for k, ek in enumerate((ch.e1, ch.e3)):
            J[:, k] = (2 * ek * (1 + r2) - 2 * z[k] * (2 * z[0] * ch.e1 + 2 * z[1] * ch.e3 + (1 - r2) * ch.c)) / (1 + r2) ** 2 \
                - 2 * z[k] * ch.c / (1 + r2)
        glogU = J.T @ (gU / U)
        psi = 0.5 * math.log(2 * U) - math.log(1 + r2)
        gpsi = 0.5 * glogU - 2 * z / (1 + r2)
        _, K = C.U_K(w.reshape(3, 1), m)
        return psi, gpsi, float(K[0])

    def rhs(t, y):
        z, v, G, Gp = y[0:2], y[2:4], y[4], y[5]
        psi, gpsi, K = geo(z)
        acc = -2 * (gpsi @ v) * v + (v @ v) * gpsi
        return np.concatenate([v, acc, [Gp, -K * G]])

    return rhs, geo


def collisions_z(m, ch):
    return [ch.to_z(cv) for cv in C.collisions(m)]


R_END = 0.05  # unit-sphere radius of the end caps; K < 0 is checked inside them, where G >= 1, G' >= 0 hold analytically


def cap_max_K(m, n=400):
    """Maximum of the exact K over the caps {unit distance to a collision in (1e-3, R_END)}."""
    out = -np.inf
    for cv in C.collisions(m):
        e1 = np.cross(cv, [0.0, 0.0, 1.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(cv, e1)
        rho = np.exp(np.linspace(math.log(1e-3), math.log(R_END), n))  # double precision is exact to 4 digits from 1e-3 (checked with mpmath)
        ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
        RR, AA = np.meshgrid(rho, ang, indexing="ij")
        W = (np.cos(RR)[None] * cv[:, None, None] + np.sin(RR)[None] *
             (np.cos(AA)[None] * e1[:, None, None] + np.sin(AA)[None] * e2[:, None, None]))
        _, K = C.U_K(W, m)
        out = max(out, float(K.max()))
    return out


def run(m, a, b, ns=400, smax=40.0):
    """Arc between collisions a and b (indices into C.collisions); chart centred at C_a. Normals from the arc points
    outside the end caps; a normal stops when it leaves the upper hemisphere or enters an end cap."""
    cols = C.collisions(m)
    ch = Chart(cols[a])
    xs_cols = sorted(float(ch.to_z(cv)[0]) for cv in cols)
    xb = float(ch.to_z(cols[b])[0])
    rhs, geo = make_rhs(m, ch)
    # arc points: the great-circle arc from C_a to C_b, unit distances in (R_END, d_ab - R_END)
    dab = math.acos(float(np.clip(cols[a] @ cols[b], -1, 1)))
    t = np.linspace(R_END, dab - R_END, ns)
    pts = [(math.sin(dab - tt) * cols[a] + math.sin(tt) * cols[b]) / math.sin(dab) for tt in t]
    res = []
    for w0 in pts:
        z0 = ch.to_z(w0)
        z0[1] = 0.0
        psi, gpsi, K0 = geo(z0)
        v0 = np.array([0.0, math.exp(-psi)])
        def leave(tt, y):
            return y[1]
        leave.terminal, leave.direction = True, -1
        def cap(tt, y):
            w = ch.to_w(y[0:2])
            return min(math.acos(min(1.0, float(w @ cv))) for cv in cols) - R_END
        cap.terminal, cap.direction = True, -1
        y0 = np.concatenate([z0 + np.array([0.0, 1e-9]), v0, [1.0, 0.0]])
        sol = solve_ivp(rhs, (0, smax), y0, method="DOP853", rtol=1e-10, atol=1e-12, events=[leave, cap])
        G, Gp = sol.y[4], sol.y[5]
        i0 = max(1, np.searchsorted(sol.t, 1e-3))
        mG, mGp = float(G.min()), float(Gp[i0:].min()) if len(Gp) > i0 else float("nan")
        we = ch.to_w(sol.y[0:2, -1])
        phi = math.atan2(we[1], we[0])  # position on the equator circle (the exit point is on or near it)
        if sol.t_events[0].size:
            end = ("axis", phi)
        elif sol.t_events[1].size:
            end = ("cap", phi)
        else:
            end = ("s_max", phi)
        res.append((float(z0[0]), mG, mGp, end, float(sol.t[-1])))
    return xs_cols, xb, res
if __name__ == "__main__":
    m = tuple(float(v) for v in sys.argv[1].split(","))
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    print(f"m {m}: max K in the end caps (unit radius {R_END}): {cap_max_K(m):+.3e}", flush=True)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        xs_cols, xb, res = run(m, a, b, ns)
        worst = min(res, key=lambda r: min(r[1], r[2]))
        kinds = {}
        for r in res:
            kinds[r[3][0]] = kinds.get(r[3][0], 0) + 1
        # monotone order of the exits on the axis (no crossing of normals through the boundary)
        # cyclic monotone order of all exit points along the equator (normals do not cross on the boundary)
        ph = np.unwrap([r[3][1] for r in res])
        dph = np.diff(ph)
        mono = bool(np.all(dph > 0) or np.all(dph < 0))
        print(f"  arc C{a+1}-C{b+1}: {len(res)} normals; min G {min(r[1] for r in res):.4f}, min G' (r > 1e-3) "
              f"{min(r[2] for r in res):+.3e} at x {worst[0]:.4f}; endings {kinds}; exits monotone along the equator {mono}; "
              f"longest normal {max(r[4] for r in res):.2f}", flush=True)
