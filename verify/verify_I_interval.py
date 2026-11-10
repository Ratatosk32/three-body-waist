"""Rigorous check of condition (I): K < 0 on the whole region outside the loop sigma_0 (paper, Proposition G), at the
five mass triples of the converse, by interval arithmetic.

Arithmetic. Intervals of IEEE doubles; +, -, *, / and sqrt are correctly rounded in IEEE arithmetic, so every result is
widened outward by one ulp (numpy.nextafter) and encloses the exact value. No other functions are used. The masses are
enclosed as intervals around their decimal values; every quantity derived from them (the coefficients of the r_ij^2,
the collision points, the chart frames) is an interval containing the exact value.

The sign of K without singularities. In the stereographic chart zeta = xi + i eta centred at a collision C_ij (the unit
sphere point w(zeta) = (2 xi e1 + 2 eta e3 + (1 - |zeta|^2) c) / (1 + |zeta|^2)), the metric is
g = 2U / (1 + |zeta|^2)^2 |d zeta|^2. With F = |zeta|^2 U / (1 + |zeta|^2)^2 = m_i m_j / (2 a_ij (1 + |zeta|^2))
+ V |zeta|^2 / (1 + |zeta|^2)^2, where V is the potential of the two other pairs, F is a smooth positive rational
function near zeta = 0, and since log |zeta|^2 is harmonic, K = -e^{-2 Phi} Delta log F, so
    K < 0  <=>  Q = F (F_xx + F_yy) - F_x^2 - F_y^2 > 0.
Q is enclosed over boxes with second-order interval jets (value, gradient, Hessian).

Covering. Each collision c gets its Voronoi cell (points at least as close to c as to the other two collisions); the
cells cover the sphere. In the chart of c a square [-L, L]^2 is cut into boxes, refined adaptively. A box is discarded
when it lies outside the cell (some (c - c').w < 0 on the whole box), or inside sigma_0 (its chain-chart image, as an
interval box, lies inside the polygon with a margin); otherwise Q > 0 must be proved on it. The square contains the
cell: every box of its boundary ring is shown to lie outside the cell, and the cell is connected and contains c.

    python verify/verify_I_interval.py [mass-key]"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INF = np.inf


# ---------------- intervals ----------------
def _down(x):
    return np.nextafter(x, -INF)


def _up(x):
    return np.nextafter(x, INF)


class Iv:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(lo if hi is None else hi, float)

    @staticmethod
    def dec(x):  # an interval around a decimal number given as a float
        return Iv(_down(float(x)), _up(float(x)))

    def __add__(self, o):
        o = _iv(o)
        return Iv(_down(self.lo + o.lo), _up(self.hi + o.hi))

    __radd__ = __add__

    def __neg__(self):
        return Iv(-self.hi, -self.lo)

    def __sub__(self, o):
        o = _iv(o)
        return Iv(_down(self.lo - o.hi), _up(self.hi - o.lo))

    def __rsub__(self, o):
        return _iv(o) - self

    def __mul__(self, o):
        o = _iv(o)
        p = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi]
        return Iv(_down(np.minimum.reduce(p)), _up(np.maximum.reduce(p)))

    __rmul__ = __mul__

    def recip(self):
        bad = (self.lo <= 0) & (self.hi >= 0)
        lo = np.where(bad, -INF, _down(1.0 / np.where(bad, 1.0, self.hi)))
        hi = np.where(bad, INF, _up(1.0 / np.where(bad, 1.0, self.lo)))
        return Iv(lo, hi)

    def __truediv__(self, o):
        return self * _iv(o).recip()

    def __rtruediv__(self, o):
        return _iv(o) * self.recip()

    def sq(self):
        a, b = self.lo * self.lo, self.hi * self.hi
        zero = (self.lo <= 0) & (self.hi >= 0)
        return Iv(np.where(zero, 0.0, _down(np.minimum(a, b))), _up(np.maximum(a, b)))

    def sqrt(self):
        return Iv(_down(np.sqrt(np.maximum(self.lo, 0.0))), _up(np.sqrt(self.hi)))


def _iv(x):
    return x if isinstance(x, Iv) else Iv(x)


# ---------------- second-order jets in two variables over intervals ----------------
class Jet:
    __slots__ = ("v", "x", "y", "xx", "xy", "yy")

    def __init__(self, v, x, y, xx, xy, yy):
        self.v, self.x, self.y, self.xx, self.xy, self.yy = v, x, y, xx, xy, yy

    @staticmethod
    def const(c, shape):
        c = _iv(c)
        z = Iv(np.zeros(shape))
        return Jet(Iv(np.broadcast_to(c.lo, shape), np.broadcast_to(c.hi, shape)), z, z, z, z, z)

    def __add__(self, o):
        if not isinstance(o, Jet):
            return Jet(self.v + o, self.x, self.y, self.xx, self.xy, self.yy)
        return Jet(self.v + o.v, self.x + o.x, self.y + o.y, self.xx + o.xx, self.xy + o.xy, self.yy + o.yy)

    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.v, -self.x, -self.y, -self.xx, -self.xy, -self.yy)

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if not isinstance(o, Jet):
            o = _iv(o)
            return Jet(self.v * o, self.x * o, self.y * o, self.xx * o, self.xy * o, self.yy * o)
        f, g = self, o
        return Jet(f.v * g.v, f.x * g.v + f.v * g.x, f.y * g.v + f.v * g.y,
                   f.xx * g.v + 2.0 * (f.x * g.x) + f.v * g.xx,
                   f.xy * g.v + f.x * g.y + f.y * g.x + f.v * g.xy,
                   f.yy * g.v + 2.0 * (f.y * g.y) + f.v * g.yy)

    __rmul__ = __mul__

    def recip(self):
        h = self.v.recip()
        h2 = h.sq()
        h3 = h2 * h
        return Jet(h, -(self.x * h2), -(self.y * h2),
                   2.0 * (self.x.sq() * h3) - self.xx * h2,
                   2.0 * (self.x * self.y * h3) - self.xy * h2,
                   2.0 * (self.y.sq() * h3) - self.yy * h2)


# ---------------- the problem ----------------
def coeffs(m):
    """Intervals (c, a, b) with r_ij^2 = a + b . w at I = 1, pairs 12, 13, 23; b has no third component."""
    m1, m2, m3 = (Iv(_down(float(x[0])), _up(float(x[1]))) if isinstance(x, tuple) else Iv.dec(x) for x in m)
    M12 = m1 + m2
    M = M12 + m3
    mu12 = m1 * m2 / M12
    mu3 = m3 * M12 / M
    A = mu3.sqrt().recip()
    s12 = mu12.sqrt()
    out = [(m1 * m2, (2.0 * mu12).recip(), ((2.0 * mu12).recip(), Iv(0.0)))]
    for c, B in ((m1 * m3, m2 / (M12 * s12)), (m2 * m3, -(m1 / (M12 * s12)))):
        out.append((c, (A.sq() + B.sq()) * 0.5, ((B.sq() - A.sq()) * 0.5, A * B)))
    return out


def frames(m, co):
    """For each collision: the interval unit vector c = -b / |b| and e1 = (-c2, c1, 0), e3 = (0, 0, 1) (exact frame)."""
    out = []
    for k, (_, a, b) in enumerate(co):
        if k == 0:
            c = (Iv(-1.0), Iv(0.0))
        else:
            nb = (b[0].sq() + b[1].sq()).sqrt()
            c = (-(b[0] / nb), -(b[1] / nb))
        out.append((c, (-c[1], c[0])))
    return out


def evaluate(m, co, fr, k, X0, X1, Y0, Y1):
    """For boxes [X0, X1] x [Y0, Y1] in the chart of collision k: the interval w (3 components), and Q."""
    shape = X0.shape
    xi = Jet(Iv(X0, X1), Iv(np.ones(shape)), Iv(np.zeros(shape)), Iv(np.zeros(shape)), Iv(np.zeros(shape)),
             Iv(np.zeros(shape)))
    eta = Jet(Iv(Y0, Y1), Iv(np.zeros(shape)), Iv(np.ones(shape)), Iv(np.zeros(shape)), Iv(np.zeros(shape)),
              Iv(np.zeros(shape)))
    r2 = xi * xi + eta * eta
    inv = (r2 + 1.0).recip()
    (c0, c1), (e10, e11) = fr[k]
    one_m = 1.0 - r2
    w0 = (2.0 * xi * e10 + one_m * c0) * inv
    w1 = (2.0 * xi * e11 + one_m * c1) * inv
    w2 = 2.0 * eta * inv
    V = None
    for p, (cp, ap, bp) in enumerate(co):
        if p == k:
            continue
        g = w0 * bp[0] + w1 * bp[1] + ap
        t = g.recip() * cp
        V = t if V is None else V + t
    _, ak, _ = co[k]
    own = inv * (co[k][0] / (2.0 * ak))
    F = own + V * r2 * inv * inv
    Q = F.v * (F.xx + F.yy) - F.x.sq() - F.y.sq()
    return (w0.v, w1.v, w2.v), Q


def outside_cell(fr, k, w):
    """True where the box surely lies outside the Voronoi cell of collision k."""
    ck = fr[k][0]
    res = np.zeros(w[0].lo.shape, bool)
    for p in range(3):
        if p == k:
            continue
        cp = fr[p][0]
        d = w[0] * (ck[0] - cp[0]) + w[1] * (ck[1] - cp[1])  # (c_k - c_p) . w, third components vanish
        res |= d.hi < 0
    return res


def inside_sigma0(w, poly, margin=1e-9):
    """True where the box surely lies inside sigma_0: its chain-chart image z = (w2, w3) / (1 - w1) (an interval box) is
    inside the polygon, with a margin."""
    den = 1.0 - w[0]
    ok = den.lo > 0
    zx = w[1] / den
    zy = w[2] / den
    cx, cy = 0.5 * (zx.lo + zx.hi), 0.5 * (zy.lo + zy.hi)
    hd = 0.5 * np.hypot(zx.hi - zx.lo, zy.hi - zy.lo)
    ok &= np.isfinite(cx) & np.isfinite(cy) & np.isfinite(hd)
    cx, cy, hd = np.where(ok, cx, 0.0), np.where(ok, cy, 0.0), np.where(ok, hd, 0.0)
    xs, ys = poly
    xp, yp = np.roll(xs, -1), np.roll(ys, -1)
    inside = np.zeros(cx.shape, bool)
    dmin = np.full(cx.shape, INF)
    for i in range(len(xs)):
        if ys[i] != yp[i]:
            inside ^= ((ys[i] > cy) != (yp[i] > cy)) & (cx < (xp[i] - xs[i]) * (cy - ys[i]) / (yp[i] - ys[i]) + xs[i])
        ex, ey = xp[i] - xs[i], yp[i] - ys[i]
        t = np.clip(((cx - xs[i]) * ex + (cy - ys[i]) * ey) / (ex * ex + ey * ey), 0.0, 1.0)
        dmin = np.minimum(dmin, np.hypot(cx - xs[i] - t * ex, cy - ys[i] - t * ey))
    return ok & inside & (dmin > hd * (1 + margin) + margin)


def check(m, poly, n0=64, maxdepth=14, max_rest=200000):
    co = coeffs(m)
    fr = frames(m, co)
    report = []
    allok = True
    for k in range(3):
        # a square containing the cell: grow L until its boundary ring of boxes is outside the cell
        L = 1.0
        while True:
            t = np.linspace(-L, L, 257)
            X0 = np.concatenate([t[:-1], t[:-1], np.full(256, -L), np.full(256, L - 2 * L / 256)])
            Y0 = np.concatenate([np.full(256, -L), np.full(256, L - 2 * L / 256), t[:-1], t[:-1]])
            h = 2 * L / 256
            w, _ = evaluate(m, co, fr, k, X0, X0 + h, Y0, Y0 + h)
            if np.all(outside_cell(fr, k, w)):
                break
            L *= 1.5
            if L > 1e3:
                raise RuntimeError("cell not bounded")
        g = np.linspace(-L, L, n0 + 1)
        X0, Y0 = np.meshgrid(g[:-1], g[:-1], indexing="ij")
        X0, Y0 = X0.ravel(), Y0.ravel()
        X1, Y1 = X0 + 2 * L / n0, Y0 + 2 * L / n0
        proved = skipped_cell = skipped_D = 0
        failed = []
        minQ = INF
        depth = 0
        while X0.size:
            w, Q = evaluate(m, co, fr, k, X0, X1, Y0, Y1)
            out = outside_cell(fr, k, w)
            ins = ~out & inside_sigma0(w, poly)
            good = ~out & ~ins & (Q.lo > 0)
            skipped_cell += int(out.sum())
            skipped_D += int(ins.sum())
            proved += int(good.sum())
            if good.any():
                minQ = min(minQ, float(Q.lo[good].min()))
            rest = ~(out | ins | good)
            if depth >= maxdepth or rest.sum() > max_rest:  # give up: not proved
                failed += list(zip(X0[rest], Y0[rest], Q.lo[rest]))
                break
            X0, X1, Y0, Y1 = X0[rest], X1[rest], Y0[rest], Y1[rest]
            Xm, Ym = 0.5 * (X0 + X1), 0.5 * (Y0 + Y1)
            X0, X1, Y0, Y1 = (np.concatenate([X0, Xm, X0, Xm]), np.concatenate([Xm, X1, Xm, X1]),
                              np.concatenate([Y0, Y0, Ym, Ym]), np.concatenate([Ym, Ym, Y1, Y1]))
            depth += 1
        ok = not failed
        allok &= ok
        report.append(f"chart of collision {['C12', 'C13', 'C23'][k]} (square half-side {L:.3g}): Q > 0 proved on "
                      f"{proved} boxes, {skipped_cell} outside the cell, {skipped_D} inside sigma_0, deepest level "
                      f"{depth}; {'all covered' if ok else f'{len(failed)} boxes not proved, e.g. {failed[:3]}'}")
    return allok, report


def main():
    chains = np.load(os.path.join(HERE, "certificates", "chains.npz"))
    meta = json.load(open(os.path.join(HERE, "certificates", "chains_meta.json"), encoding="utf-8"))["masses"]
    keys = [sys.argv[1]] if len(sys.argv) > 1 else list(meta)
    allok = True
    for key in keys:
        m = tuple(float(x) for x in key.split(","))
        ok, rep = check(m, chains["m_" + key.replace(",", "_")][0])
        allok &= ok
        print(f"masses {m}: (I) K < 0 outside sigma_0 {'PROVED' if ok else 'NOT PROVED'}", flush=True)
        for r in rep:
            print("  " + r, flush=True)
    if len(sys.argv) == 1:
        # negative control: with a tiny loop in place of sigma_0 the region of positive curvature near C12 is exposed,
        # and the proof must fail
        t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        ok, rep = check((1.0, 1.0, 0.57), np.vstack([0.05 * np.cos(t), 0.05 * np.sin(t)]), maxdepth=8)
        allok &= not ok
        print(f"control: masses (1, 1, 0.57) with a loop of chart radius 0.05 in place of sigma_0: "
              f"{'NOT PROVED, AS EXPECTED' if not ok else 'PROVED (UNEXPECTED)'}", flush=True)
    print("ALL PASS" if allok else "SOME FAILED")


if __name__ == "__main__":
    main()


def cover_slice(m_lo, m_hi, poly, w0=2e-3, wmin=1e-4, base=(1.0, 1.0), progress=False):
    """Prove (I) for masses (base[0], base[1], m3), m3 in [m_lo, m_hi], with one loop sigma_0 for all of them, by
    consecutive mass intervals (widened after success, halved after failure). Returns the m3 reached and the count."""
    m, w, n = m_lo, w0, 0
    while m < m_hi:
        hi = min(m + w, m_hi)
        ok, _ = check((base[0], base[1], (m, hi)), poly, maxdepth=12, max_rest=40000)
        if ok:
            m, n, w = hi, n + 1, min(w * 1.5, 0.05)
            if progress:
                print(f"  proved up to m3 = {m:.6f} ({n} intervals)", flush=True)
        else:
            w /= 2
            if w < wmin:
                break
    return m, n
