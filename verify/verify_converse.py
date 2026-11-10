"""Independent check of the converse ("no end has a waist => no closed orbit in the class (12)") at five mass triples,
following the reduction in the paper (Proposition G):
  (I)   K < 0 outside the outer loop sigma_0, and sigma_0 disjoint from the equator arc 3;
  (II)  a chain of nested convex loops sigma_0 > sigma_1 > ... > sigma_N: for each k one time T_k such that every
        inward normal of sigma_k keeps G > 0, G' < 0 on [0, T_k] (Jacobi field G'' = -K G, G(0) = 1, G'(0) = -k_g),
        the parallels P_t, 0 <= t <= T_k, are simple and nested (so the normal map is an embedding), and P_{T_k} lies
        inside sigma_{k+1}; the last loop inside a cap about C_12 where the round circles are convex (condition C);
  (III) no stutter: the Fermi normals of each equator arc (G(0) = 1, G'(0) = 0) keep G > 0, G' > 0 and leave the
        hemisphere in monotone order, including the normals from inside the collision caps down to rho = 1e-3, which
        stay near their collision (where K < 0) until they leave.
Also: U by two routes, no waist at any end, Lemma K against its closed form.
Shares no code with the research tools: numpy only, its own curvature, charts, fixed-step RK4 in arclength (every
integration repeated with half the step) and polygon tests. The loops are read from certificates/chains.npz (they
came from a curve-shortening flow; nothing about how they were found is assumed: every property used is checked).
    python verify/verify_converse.py [mass-key]      (default: all five)"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R_END = 0.05  # unit-sphere radius of the collision caps


# ---------- the metric g = 2U g0 on the shape sphere (g0 = round sphere of radius 1/2) ----------
def coeffs(m):
    """r_ij^2 = a + b . w at I = 1 for the pairs 12, 13, 23 (w a unit vector of the shape sphere), with the factor c."""
    m1, m2, m3 = m
    M12, M = m1 + m2, m1 + m2 + m3
    mu12, mu3 = m1 * m2 / M12, m3 * M12 / M
    out = [(m1 * m2, 1 / (2 * mu12), np.array([1 / (2 * mu12), 0.0, 0.0]))]
    A = 1 / math.sqrt(mu3)
    for c, B in ((m1 * m3, m2 / (M12 * math.sqrt(mu12))), (m2 * m3, -m1 / (M12 * math.sqrt(mu12)))):
        out.append((c, (A * A + B * B) / 2, np.array([(B * B - A * A) / 2, A * B, 0.0])))
    return out


def U_from_positions(w, m):
    """U from the bodies' positions rebuilt from the Hopf point (a second, independent route to U)."""
    m1, m2, m3 = m
    M12, M = m1 + m2, m1 + m2 + m3
    mu12, mu3 = m1 * m2 / M12, m3 * M12 / M
    a = math.sqrt(max((1 + w[0]) / 2, 0.0))
    c = complex(w[1], w[2]) / 2
    z1, z2 = a, (c.conjugate() / a if a > 0 else math.sqrt(max((1 - w[0]) / 2, 0.0)))
    d12 = z1 / math.sqrt(mu12)  # q2 - q1
    x3 = z2 / math.sqrt(mu3)    # q3 - centre of mass of 1, 2
    q1, q2, q3 = -m2 / M12 * d12, m1 / M12 * d12, x3
    return m1 * m2 / abs(q2 - q1) ** 2 + m1 * m3 / abs(q3 - q1) ** 2 + m2 * m3 / abs(q3 - q2) ** 2


def geom(W, m):
    """U, grad log U (any extension) and K at unit vectors W of shape (3, n)."""
    U = np.zeros(W.shape[1])
    gU = np.zeros_like(W)
    HU = np.zeros((3, 3, W.shape[1]))
    for c, a, b in coeffs(m):
        g = a + b @ W
        U += c / g
        gU -= c * b[:, None] / g ** 2
        HU += 2 * c * np.outer(b, b)[:, :, None] / g ** 3
    gF = gU / U
    HF = HU / U - gU[:, None, :] * gU[None, :, :] / U ** 2
    lapS = np.einsum("iin->n", HF) - np.einsum("in,ijn,jn->n", W, HF, W) - 2 * np.einsum("in,in->n", W, gF)
    return U, gF, (4 - 2 * lapS) / (2 * U)


def collisions(m):
    out = []
    for c, a, b in coeffs(m):
        assert abs(a - np.linalg.norm(b)) < 1e-12  # r^2 vanishes on the sphere exactly at -b/|b|
        out.append(-b / np.linalg.norm(b))
    return out


def kappa2_closed(m, pair=(0, 1)):
    i, j = pair
    k = 3 - i - j
    mi, mj, mk = m[i], m[j], m[k]
    M = sum(m)
    return -(mi + mj) * (mk ** 2 * (mi + mj) ** 3 - mi ** 2 * mj ** 2 * M) / (4 * M * mi ** 4 * mj ** 4)


def frame(c):
    e1 = np.cross(c, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(c, e1)


def ring(c, rho, a):
    """Points at unit-sphere distance rho from c, at angles a (arrays of one shape), as (3, n)."""
    e1, e2 = frame(c)
    rho, a = np.broadcast_arrays(np.asarray(rho, float).ravel(), np.asarray(a, float).ravel())
    return np.cos(rho) * c[:, None] + np.sin(rho) * (np.cos(a) * e1[:, None] + np.sin(a) * e2[:, None])


def dist(W, c):
    return np.arccos(np.clip(c @ W, -1.0, 1.0))


class Chart:
    """Stereographic chart from -c: z = (w.e1, w.e3) / (1 + w.c); metric g = e^(2 psi) |dz|^2."""

    def __init__(self, c, e1):
        self.c, self.e1, self.e3 = np.asarray(c, float), np.asarray(e1, float), np.array([0.0, 0.0, 1.0])

    def to_w(self, Z):
        r2 = np.sum(Z * Z, 0)
        return (2 * np.outer(self.e1, Z[0]) + 2 * np.outer(self.e3, Z[1]) + np.outer(self.c, 1 - r2)) / (1 + r2)

    def to_z(self, W):
        return np.vstack([self.e1 @ W, self.e3 @ W]) / (1 + self.c @ W)

    def psi_grad_K(self, Z, m):
        r2 = np.sum(Z * Z, 0)
        W = self.to_w(Z)
        U, gF, K = geom(W, m)
        N = 2 * np.outer(self.e1, Z[0]) + 2 * np.outer(self.e3, Z[1]) + np.outer(self.c, 1 - r2)
        gpsi = np.empty_like(Z)
        for k, ek in enumerate((self.e1, self.e3)):
            Jk = ((2 * ek[:, None] - 2 * Z[k] * self.c[:, None]) * (1 + r2) - 2 * Z[k] * N) / (1 + r2) ** 2
            gpsi[k] = 0.5 * np.sum(Jk * gF, 0) - 2 * Z[k] / (1 + r2)
        psi = 0.5 * np.log(2 * U) - np.log(1 + r2)
        return psi, gpsi, K, W


def rhs(ch, m, Y):
    """Unit-speed geodesics z(s) of e^(2 psi)|dz|^2 with the scalar Jacobi field G'' = -K G, for columns of Y."""
    psi, gp, K, W = ch.psi_grad_K(Y[0:2], m)
    v = Y[2:4]
    acc = -2 * np.sum(gp * v, 0) * v + np.sum(v * v, 0) * gp
    return np.vstack([v, acc, Y[5], -K * Y[4]]), K


def rk4_step(ch, m, Y, h):
    k1, K = rhs(ch, m, Y)
    k2 = rhs(ch, m, Y + 0.5 * h * k1)[0]
    k3 = rhs(ch, m, Y + 0.5 * h * k2)[0]
    k4 = rhs(ch, m, Y + h * k3)[0]
    return Y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), K


# ---------- polygons ----------
def inside_many(P, poly):
    """Even-odd test for points P (2, n) in the closed polygon poly (2, k)."""
    x, y = P
    res = np.zeros(P.shape[1], bool)
    xs, ys = poly
    xp, yp = np.roll(xs, 1), np.roll(ys, 1)
    for i in range(poly.shape[1]):
        if ys[i] == yp[i]:
            continue
        res ^= ((ys[i] > y) != (yp[i] > y)) & (x < (xp[i] - xs[i]) * (y - ys[i]) / (yp[i] - ys[i]) + xs[i])
    return res


def is_simple(P):
    """No two non-adjacent edges of the closed polygon P (2, n) intersect."""
    A, B = P, np.roll(P, -1, 1)
    n = P.shape[1]

    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    a0, a1 = A[:, :, None], B[:, :, None]
    b0, b1 = A[:, None, :], B[:, None, :]
    cross = (orient(a0, a1, b0) * orient(a0, a1, b1) < 0) & (orient(b0, b1, a0) * orient(b0, b1, a1) < 0)
    idx = np.arange(n)
    gap = np.abs(idx[:, None] - idx[None, :])
    return not np.any(cross & (gap > 1) & (gap < n - 1))


# ---------- the checks ----------
def check_no_waist(m):
    vals = [kappa2_closed(m, p) for p in ((0, 1), (0, 2), (1, 2))]
    return all(v < 0 for v in vals), "kappa_2 at C12, C13, C23: " + ", ".join(f"{v:+.4f}" for v in vals)


def check_lemma_K(m):
    m1, m2, _ = m
    formula = kappa2_closed(m)
    c = collisions(m)[0]
    a = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    vals = [np.mean(geom(ring(c, np.full_like(a, rho), a), m)[2]) / rho ** 2 for rho in (4e-3, 2e-3)]
    k = (4 * vals[1] - vals[0]) / 3
    scale = 4 * (m1 * m2 / (m1 + m2)) * m1 * m2  # kappa_2 * scale = 1 - V0 / (mu12 m1 m2), of order 1
    diff = abs(k - formula) * scale
    return diff <= 2e-3, (f"kappa_2 numeric {k:+.5f}, closed form {formula:+.5f} (normalised difference {diff:.1e} "
                          f"<= 2e-3)")


def check_U_routes(m):
    rng = np.random.default_rng(3)
    cols = collisions(m)
    worst = 0.0
    for _ in range(200):
        w = rng.normal(size=3)
        w /= np.linalg.norm(w)
        if min(dist(w.reshape(3, 1), c)[0] for c in cols) < 1e-2:
            continue
        worst = max(worst, abs(geom(w.reshape(3, 1), m)[0][0] / U_from_positions(w, m) - 1))
    return worst < 1e-10, f"U from the affine formulas vs from the positions: max rel diff {worst:.1e}"


CHAIN_CHART = Chart([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0])  # the chains' chart: z = (w2, w3) / (1 - w1), C12 at 0


def check_I(m, sigma0):
    cols = collisions(m)
    N = 400
    th = (np.arange(N) + 0.5) * math.pi / N
    ph = (np.arange(2 * N) + 0.5) * math.pi / N
    T, P = np.meshgrid(th, ph, indexing="ij")
    W = np.vstack([(np.sin(T) * np.cos(P)).ravel(), (np.sin(T) * np.sin(P)).ravel(), np.cos(T).ravel()])
    W = W[:, np.min([dist(W, c) for c in cols], 0) > 0.01]
    W = W[:, ~inside_many(CHAIN_CHART.to_z(W), sigma0)]
    worst = float(np.max(geom(W, m)[2]))
    # arc 3 (body 3 in the middle) is the equator arc from C13 to C23 not through C12
    c13, c23 = cols[1], cols[2]
    d = math.acos(float(c13 @ c23))
    t = np.linspace(0, d, 2001)
    arc = (np.outer(c13, np.sin(d - t)) + np.outer(c23, np.sin(t))) / math.sin(d)
    assert np.min(dist(arc, cols[0])) > 0.1
    arc = arc[:, 1 - arc[0] > 1e-9]  # the antipode of C12 is the chart's point at infinity, outside every loop
    arc3_out = not np.any(inside_many(CHAIN_CHART.to_z(arc), sigma0))
    return worst < 0 and arc3_out, (f"max K outside sigma_0 (grid 400 x 800, collisions cut at 0.01): {worst:+.3e}; "
                                    f"arc 3 outside sigma_0: {arc3_out}")


def check_C(m, rho1):
    c = collisions(m)[0]
    R, A = np.meshgrid(np.linspace(1e-3, rho1, 400), np.linspace(0, 2 * math.pi, 180, endpoint=False), indexing="ij")
    R, A = R.ravel(), A.ravel()
    W = ring(c, R, A)
    t = (W * np.cos(R) - c[:, None]) / np.sin(R)
    vals = 2 / np.tan(R) + np.sum(geom(W, m)[1] * t, 0)
    worst = float(vals.min())
    return worst > 0, f"condition C on the circles 1e-3 <= rho <= {rho1:.4f}: min of 2 cot rho + d_rho log U {worst:+.3e}"


def leaf_kg(ch, m, z):
    area = 0.5 * np.sum(z[0] * np.roll(z[1], -1) - np.roll(z[0], -1) * z[1])
    if area < 0:
        z = z[:, ::-1]
    zp, zm = np.roll(z, -1, 1), np.roll(z, 1, 1)
    T = zp - zm
    T /= np.linalg.norm(T, axis=0)
    n = np.vstack([-T[1], T[0]])  # counterclockwise: the left normal points inwards
    hm = 0.5 * np.linalg.norm(zp - zm, axis=0)
    kflat = np.sum((zp - 2 * z + zm) * n, 0) / hm ** 2
    psi, gp, _, _ = ch.psi_grad_K(z, m)
    return z, n, psi, np.exp(-psi) * (kflat - np.sum(gp * n, 0))


def pair_run(m, z, n, psi, kg, nxt, h, cols, smax=5.0):
    """All inward normals of one loop at once, up to T = the last first entry into nxt: G, G', caps, parallels."""
    Y = np.vstack([z, n * np.exp(-psi), np.ones(z.shape[1]), -kg])
    first = np.zeros(z.shape[1], bool)
    traj, Gmin, Gpmax, capmin = [Y[0:2].copy()], np.inf, -np.inf, np.inf
    step = 0
    while not first.all() and step * h < smax:
        Y, _ = rk4_step(CHAIN_CHART, m, Y, h)
        step += 1
        traj.append(Y[0:2].copy())
        Gmin, Gpmax = min(Gmin, Y[4].min()), max(Gpmax, Y[5].max())
        Wn = CHAIN_CHART.to_w(Y[0:2])
        capmin = min(capmin, min(float(dist(Wn, c).min()) for c in cols))
        first |= inside_many(Y[0:2], nxt)
    return dict(T=step * h, entered=bool(first.all()), all_in=bool(np.all(inside_many(Y[0:2], nxt))), Gmin=Gmin,
                Gpmax=Gpmax, capmin=capmin, traj=traj)


def check_II(m, chain):
    cols = collisions(m)
    msgs, ok = [], True
    rho_last = 2 * math.atan(float(np.max(np.hypot(*chain[-1]))))
    okC, msgC = check_C(m, rho_last)
    ok &= okC
    msgs.append(f"last loop within rho <= {rho_last:.4f}; {msgC}")
    bad, min_kg, nest_bad, worst_Gp, worst_half, notsimple = [], math.inf, 0, -math.inf, -math.inf, 0
    for k in range(len(chain) - 1):
        z, n, psi, kg = leaf_kg(CHAIN_CHART, m, chain[k])
        min_kg = min(min_kg, float(kg.min()))
        nxt = chain[k + 1]
        nest_bad += int(np.count_nonzero(~inside_many(nxt, z)))
        r = pair_run(m, z, n, psi, kg, nxt, 0.002, cols)
        r2 = pair_run(m, z, n, psi, kg, nxt, 0.001, cols)
        worst_half = max(worst_half, r2["Gpmax"])
        traj = r["traj"]
        js = np.unique(np.linspace(0, len(traj) - 1, 21).astype(int))
        simple = all(is_simple(traj[j]) for j in js)
        nested = all(np.all(inside_many(traj[j2], traj[j1])) for j1, j2 in zip(js[:-1], js[1:]))
        notsimple += (not simple) + (not nested)
        worst_Gp = max(worst_Gp, r["Gpmax"])
        if not (r["entered"] and r["all_in"] and r["Gmin"] > 0 and r["Gpmax"] < 0 and r["capmin"] > R_END
                and simple and nested and r2["all_in"] and r2["Gpmax"] < 0):
            bad.append(k)
    ok &= not bad and min_kg > 0 and nest_bad == 0
    msgs.append(f"{len(chain)} loops: min k_g {min_kg:+.4f}, nesting violations {nest_bad}; one T_k per pair: max G' "
                f"on [0, T_k] {worst_Gp:+.2e} (must be < 0), parallels simple and nested at 21 times per pair "
                f"(failures {notsimple}), P(T_k) inside the next loop; failing pairs {bad if bad else 'none'} of "
                f"{len(chain) - 1}; with half the step max G' {worst_half:+.2e}")
    return ok, "; ".join(msgs)


def fermi_run(ch, m, Z0, h, cols, own=None, smax=80.0):
    """Fermi normals from points Z0 (2, n) of the axis y = 0 into y > 0, all at once, until they cross y = 0 again.
    A normal that enters the cap of a collision other than `own` is followed further: it must either cross the equator
    there ("axis") or go down the end ("end": unit-sphere distance < 0.01 from the collision, moving inwards, at an
    angle to the circles about the collision that is too steep for it to turn back, by the Clairaut relation of the
    near-cylinder end of Lemma E: cos(angle) < exp(-psi_max), psi_max = |kappa_2| R^2 rho^2 / 2 bounding psi there).
    Leaving the cap again (distance > 2 R_END) while still in the hemisphere is reported as "re-emerged"."""
    psi = ch.psi_grad_K(Z0, m)[0]
    n = Z0.shape[1]
    Y = np.vstack([Z0, np.zeros(n), np.exp(-psi), np.ones(n), np.zeros(n)])
    live = np.ones(n, bool)
    why = np.array(["s_max"] * n, dtype=object)
    foreign = np.full(n, -1)
    Gmin, Gpmin, owndmax = np.full(n, np.inf), np.full(n, np.inf), np.zeros(n)
    end = np.zeros((3, n))
    pairs = ((0, 1), (0, 2), (1, 2))
    others = [f for f, c in enumerate(cols) if own is None or not np.allclose(c, own)]
    zc = {f: ch.to_z(cols[f].reshape(3, 1))[:, 0] for f in others}
    psib = {}
    for f in others:
        i, j = pairs[f]
        psib[f] = abs(kappa2_closed(m, pairs[f])) * 2 * (m[i] * m[j] / (m[i] + m[j])) * m[i] * m[j] / 2
    s = 0.0
    while live.any() and s < smax:
        Yold = Y.copy()
        Ynew, _ = rk4_step(ch, m, Y, h)
        s += h
        Y[:, live] = Ynew[:, live]
        Wn = ch.to_w(Y[0:2])
        Gmin[live] = np.minimum(Gmin[live], Y[4, live])
        if s > 1e-3:
            Gpmin[live] = np.minimum(Gpmin[live], Y[5, live])
        if own is not None:
            owndmax[live] = np.maximum(owndmax[live], dist(Wn[:, live], own))
        crossed = live & (Y[1] < 0)
        if crossed.any():  # the crossing point, interpolated linearly in the step
            lam = Yold[1, crossed] / (Yold[1, crossed] - Y[1, crossed])
            Wn[:, crossed] = ch.to_w(Yold[0:2, crossed] + lam * (Y[0:2, crossed] - Yold[0:2, crossed]))
        deep = np.zeros(n, bool)
        back = np.zeros(n, bool)
        for f in others:
            d = dist(Wn, cols[f])
            foreign[live & (foreign < 0) & (d < R_END)] = f
            mine = live & (foreign == f) & ~crossed
            back |= mine & (d > 2 * R_END)
            r = zc[f][:, None] - Y[0:2]
            r /= np.linalg.norm(r, axis=0)
            v = Y[2:4]
            vr = np.sum(v * r, 0)
            cos_par = np.abs(v[0] * r[1] - v[1] * r[0]) / np.linalg.norm(v, axis=0)
            deep |= mine & (d < 0.01) & (vr > 0) & (cos_par < np.exp(-psib[f] * d ** 2))
        for mask, r in ((crossed, "axis"), (deep, "end"), (back, "re-emerged")):
            mask = mask & live
            why[mask] = r
            end[:, mask] = Wn[:, mask]
            live &= ~mask
        for f in others:  # a normal going down an end leaves at that collision
            sel = (why == "end") & (foreign == f)
            end[:, sel] = cols[f][:, None]
    end[:, live] = ch.to_w(Y[0:2])[:, live]
    return dict(why=why, Gmin=Gmin, Gpmin=Gpmin, owndmax=owndmax, end=end, foreign=foreign)


def check_III(m, ns=100, nend=30):
    cols = collisions(m)
    msgs, ok = [], True
    check_III.arcs = []
    R, A = np.meshgrid(np.exp(np.linspace(math.log(1e-3), math.log(2 * R_END), 80)),
                       np.linspace(0, 2 * math.pi, 72, endpoint=False), indexing="ij")
    capK = max(float(geom(ring(c, R, A), m)[2].max()) for c in cols)
    ok &= capK < 0
    msgs.append(f"max K in the collision caps 1e-3 <= rho <= {2 * R_END}: {capK:+.2e}")
    # Pi_H: the total positive curvature of the upper hemisphere (dA_g = (U / 2) dA of the unit sphere)
    N = 600
    th, ph = (np.arange(N) + 0.5) * (math.pi / 2) / N, (np.arange(4 * N) + 0.5) * (2 * math.pi) / (4 * N)
    T, P = np.meshgrid(th, ph, indexing="ij")
    W = np.vstack([(np.sin(T) * np.cos(P)).ravel(), (np.sin(T) * np.sin(P)).ravel(), np.cos(T).ravel()])
    U, _, K = geom(W, m)
    Pi_H = float(np.sum(np.maximum(K, 0) * U / 2 * np.sin(T.ravel())) * (math.pi / 2 / N) * (2 * math.pi / (4 * N)))
    ok &= Pi_H < math.pi
    msgs.append(f"Pi_H (positive curvature of a hemisphere) {Pi_H:.2e} < pi: no geodesic monogon")
    for ia, ib in ((0, 1), (0, 2), (1, 2)):
        ca, cb = cols[ia], cols[ib]
        third = cols[3 - ia - ib]
        dab = math.acos(float(ca @ cb))
        ch = Chart(ca, np.cross([0.0, 0.0, 1.0], ca))
        end_t = np.exp(np.linspace(math.log(1e-3), math.log(R_END), nend + 1))[:-1]
        fams = []
        for ts, own in ((end_t, ca), (np.linspace(R_END, dab - R_END, ns), None), (dab - end_t[::-1], cb)):
            P = (np.outer(ca, np.sin(dab - ts)) + np.outer(cb, np.sin(ts))) / math.sin(dab)
            assert np.min(dist(P, third)) > R_END
            Z0 = ch.to_z(P)
            assert np.max(np.abs(Z0[1])) < 1e-12  # the arc lies on the chart's axis
            Z0[1] = 0.0
            fams.append((own, [fermi_run(ch, m, Z0, h, cols, own) for h in (0.01, 0.005)]))
        goods, angs = [], []
        for r in (0, 1):  # both step sizes must pass
            Gmin_r = min(float(f[1][r]["Gmin"].min()) for f in fams)
            Gpmin_r = min(float(f[1][r]["Gpmin"].min()) for f in fams)
            ends_near = all(np.all(f[1][r]["owndmax"] < 2 * R_END) for f in fams if f[0] is not None)
            ends_axis = all(np.all(f[1][r]["why"] == "axis") for f in fams if f[0] is not None)
            stuck = sum(int(np.count_nonzero(np.isin(f[1][r]["why"], ["s_max", "re-emerged"]))) for f in fams)
            ang = np.unwrap(np.concatenate([np.arctan2(f[1][r]["end"][1], f[1][r]["end"][0]) for f in fams]))
            isend = np.concatenate([f[1][r]["why"] == "end" for f in fams])
            dang = np.diff(ang)
            both = isend[:-1] & isend[1:]  # normals going down the same end share its collision as their exit
            blocks = int(np.count_nonzero(np.diff(isend.astype(int)) == 1)) + int(isend[0])
            mono_r = bool((np.all(dang[~both] > 0) or np.all(dang[~both] < 0)) and np.all(np.abs(dang[both]) < 1e-9)
                          and blocks <= 1)
            goods.append(Gmin_r > 0 and Gpmin_r > 0 and mono_r and ends_near and ends_axis and stuck == 0)
            angs.append(ang)
            if r == 0:
                Gmin, Gpmin, mono = Gmin_r, Gpmin_r, mono_r
        half = float(np.max(np.abs(angs[0] - angs[1])))
        nf = sum(int(np.count_nonzero(f[1][0]["foreign"] >= 0)) for f in fams)
        nf_axis = sum(int(np.count_nonzero((f[1][0]["foreign"] >= 0) & (f[1][0]["why"] == "axis"))) for f in fams)
        nf_end = sum(int(np.count_nonzero((f[1][0]["foreign"] >= 0) & (f[1][0]["why"] == "end"))) for f in fams)
        good = all(goods)
        ok &= good
        check_III.arcs.append(good)
        msgs.append(f"arc between collisions {ia + 1} and {ib + 1} (C12, C13, C23 numbered 1, 2, 3): min G {Gmin:.4f}, "
                    f"min G' {Gpmin:+.2e}, exits monotone {mono} ({2 * nend + ns} normals, {nend} from each cap down to "
                    f"rho 1e-3; cap normals stay within {2 * R_END} of their collision {ends_near} and leave across the "
                    f"equator {ends_axis}); {nf} normals enter another collision's cap: {nf_axis} cross the equator there, {nf_end} go "
                    f"down the end, none re-emerges ({stuck} undecided); the same with half the step {goods[1]}, exits "
                    f"moved by <= {half:.1e}")
    return ok, "; ".join(msgs)


def main():
    meta = json.load(open(os.path.join(HERE, "certificates", "chains_meta.json"), encoding="utf-8"))["masses"]
    chains = np.load(os.path.join(HERE, "certificates", "chains.npz"))
    keys = [sys.argv[1]] if len(sys.argv) > 1 else list(meta)
    allok = True
    for key in keys:
        m = tuple(float(x) for x in key.split(","))
        chain = chains["m_" + key.replace(",", "_")]
        print(f"masses {m}", flush=True)
        for name, fn in (("no waist at any end", lambda: check_no_waist(m)),
                         ("U two routes", lambda: check_U_routes(m)), ("Lemma K", lambda: check_lemma_K(m)),
                         ("(I) curvature outside sigma_0", lambda: check_I(m, chain[0])),
                         ("(II) chain of convex loops", lambda: check_II(m, chain)),
                         ("(III) no stutter (Fermi normals)", lambda: check_III(m))):
            ok, msg = fn()
            allok &= ok
            print(f"  {name:34s} {'PASS' if ok else 'FAIL'}  {msg}", flush=True)
    if len(sys.argv) == 1:
        # negative control: masses (1, 1, 0.5) have a waist at C12; the checks must see it
        m = (1.0, 1.0, 0.5)
        okw, msgw = check_no_waist(m)
        okK, msgK = check_lemma_K(m)
        check_III(m)
        arcs = check_III.arcs
        expect = (not okw) and kappa2_closed(m) > 0 and okK and arcs == [False, False, True]
        allok &= expect
        print(f"control (1, 1, 0.5), a waist at C12: no-waist check fails ({msgw}); {msgK}; (III) per arc C12-C13, "
              f"C12-C23, C13-C23: {['pass' if a else 'fail' for a in arcs]} (expected fail, fail, pass): "
              f"{'AS EXPECTED' if expect else 'NOT AS EXPECTED'}", flush=True)
    print("ALL PASS" if allok else "SOME FAILED")


if __name__ == "__main__":
    main()
