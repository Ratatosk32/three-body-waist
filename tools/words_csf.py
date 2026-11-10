"""Curve shortening flow for the converse of the waist theorem (proofs/waist_theorem.md, lemma F). Exploration, not a
registered test.
Chart: stereographic projection of the shape sphere from the antipode of C_12, so C_12 is z = 0:
z = (w2, w3) / (1 - w1),  w1 = (r^2 - 1)/(r^2 + 1), (w2, w3) = 2 z / (r^2 + 1),  r = |z|.
Metric g = 2U g0 = e^(2 psi) |dz|^2, psi = (1/2) log 2U - log(1 + r^2) (g0 = unit sphere / 4 = |dz|^2 / (1 + r^2)^2).
A polygon loop round z = 0 moves with the g-curvature flow X_t = e^(-2 psi) (k_flat - d_n psi) n (n the flat unit
normal, k_flat the flat curvature towards n); k_g = e^(-psi) (k_flat - d_n psi). Under the flow on a surface
k_t = k_ss + k^3 + K k, so k > 0 (towards C_12) is preserved: the leaves then form the foliation of lemma F.
Outcome: the flow stalls on a closed geodesic (length settles above the cuff's end limit) or runs into the 12-end
(the loop shrinks round z = 0 and its g-length tends to c_12 from above).
    python -I tools/words_csf.py m1,m2,m3 [R0]"""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import words_curv_exact as C  # noqa: E402


def inv_stereo(z):
    r2 = np.sum(z * z, 0)
    return np.vstack([(r2 - 1) / (r2 + 1), 2 * z / (r2 + 1)])


def logU_grad(w, m):
    U = np.zeros(w.shape[1])
    gU = np.zeros_like(w)
    for c, a, b in C.coeffs(m):
        g = a + b @ w
        U += c / g
        gU -= c * b[:, None] / g ** 2
    return U, gU / U


def psi_and_grad(z, m):
    r2 = np.sum(z * z, 0)
    w = inv_stereo(z)
    U, gw = logU_grad(w, m)
    # J = dw / dz (3 x 2 per point)
    J = np.zeros((3, 2, z.shape[1]))
    J[0] = 4 * z / (r2 + 1) ** 2
    for a in range(2):
        for b in range(2):
            J[1 + a, b] = 2 * (a == b) / (r2 + 1) - 4 * z[a] * z[b] / (r2 + 1) ** 2
    glogU = np.einsum("ijk,ik->jk", J, gw)
    psi = 0.5 * np.log(2 * U) - np.log(1 + r2)
    gpsi = 0.5 * glogU - 2 * z / (1 + r2)
    return psi, gpsi


def resample(z, n, psi):
    """Resample the closed polygon to n points equally spaced in g-length."""
    zz = np.hstack([z, z[:, :1]])
    seg = np.linalg.norm(np.diff(zz, axis=1), axis=0)
    e = np.exp(psi)
    ds = seg * 0.5 * (e + np.roll(e, -1))
    s = np.concatenate([[0], np.cumsum(ds)])
    t = np.linspace(0, s[-1], n + 1)[:-1]
    return np.vstack([np.interp(t, s, zz[0]), np.interp(t, s, zz[1])]), s[-1]


def step_geom(z):
    zp, zm = np.roll(z, -1, 1), np.roll(z, 1, 1)
    T = zp - zm
    h = np.linalg.norm(T, axis=0)
    T = T / h
    nrm = np.vstack([-T[1], T[0]])  # left normal; the loop runs counterclockwise round 0, so n points inwards
    d2 = zp - 2 * z + zm
    hm = 0.5 * h
    kflat = np.sum(d2 * nrm, 0) / hm ** 2
    return nrm, kflat, hm


def run(m, R0=None, n=200, steps=3000000, report=10000):
    cols = C.collisions(m)
    zc = [np.array(c[1:]) / (1 - c[0]) for c in cols[1:]]  # images of C13, C23
    rmin = min(np.linalg.norm(p) for p in zc)
    R0 = R0 or 0.8 * rmin
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    if os.environ.get("CSF_DENT"):
        # a big loop (0.8 of the FARTHER collision's radius) with a dent passing on the near side of each collision
        # that lies inside it: r(th) <= |z_c| - margin near the collision's angle, a parabola in the angle elsewhere
        R0 = 0.8 * max(np.linalg.norm(p) for p in zc)
        r = np.full(n, R0)
        for p in zc:
            Rc, tc = np.linalg.norm(p), math.atan2(p[1], p[0])
            if Rc < R0 / 0.8 * 0.999:
                d = np.angle(np.exp(1j * (th - tc)))
                r = np.minimum(r, 0.8 * Rc + 3.0 * Rc * d ** 2)
        z = r * np.vstack([np.cos(th), np.sin(th)])
    else:
        z = R0 * np.vstack([np.cos(th), np.sin(th)])
    c12 = 2 * math.pi * math.sqrt(2 * m[0] * m[1] / (m[0] + m[1]) * m[0] * m[1])
    print(f"m {m}: C13, C23 at |z| = {', '.join(f'{np.linalg.norm(p):.3f}' for p in zc)}; start circle R0 {R0:.3f}; "
          f"c12 {c12:.6f}", flush=True)
    hist = []
    for it in range(steps):
        psi, gpsi = psi_and_grad(z, m)
        if it % 20 == 0:
            z, L = resample(z, n, psi)
            psi, gpsi = psi_and_grad(z, m)
        nrm, kflat, hm = step_geom(z)
        dnpsi = np.sum(gpsi * nrm, 0)
        kg = np.exp(-psi) * (kflat - dnpsi)
        v = np.exp(-2 * psi) * (kflat - dnpsi)
        dt = 0.2 * np.min(hm ** 2 * np.exp(2 * psi))
        if it > 0 and kg.min() > 0 and not getattr(run, "saved", False):
            # the first everywhere-convex leaf: sigma_0 of theorem G (saved for the outside check)
            np.save(os.path.join(ROOT, "words/out/csf", "sigma0_" + ",".join(f"{x:g}" for x in m) + ".npy"), z)
            run.saved = True
            print(f"  sigma_0 saved at it {it} (first leaf with k_g > 0 everywhere)", flush=True)
        for thr in (0.05, 0.1, 0.15):  # later, more convex leaves (II': their inward parallels must stay convex)
            key = f"saved{thr}"
            if it > 0 and kg.min() > thr and not getattr(run, key, False):
                np.save(os.path.join(ROOT, "words/out/csf", f"sigma{thr}_" + ",".join(f"{x:g}" for x in m) + ".npy"), z)
                setattr(run, key, True)
                print(f"  sigma (k_g > {thr}) saved at it {it}", flush=True)
        if os.environ.get("CSF_SIGMA0_ONLY") and getattr(run, "saved0.15", False):
            return
        if os.environ.get("CSF_CHAIN") and it % 20 == 0 and kg.min() > 0:
            # the chain of leaves (II''): save a convex leaf whenever the g-length dropped by CHAIN_DL since the last one;
            # stop once the leaf lies inside the cap of chart radius CSF_CAP
            zz = np.hstack([z, z[:, :1]])
            e = np.exp(psi)
            Lnow = float(np.sum(np.linalg.norm(np.diff(zz, axis=1), axis=0) * 0.5 * (e + np.roll(e, -1))))
            last = getattr(run, "chain_L", None)
            rnow = float(np.hypot(*z).max())
            rlast = getattr(run, "chain_R", None)
            # a new leaf when the g-length dropped by CHAIN_DL or (near the end, where the length hardly changes)
            # the loop's chart radius shrank by the factor CHAIN_DR
            # or (CHAIN_DD, for one parallel time T_k per pair) some point moved a g-distance CHAIN_DD since the last leaf
            if (last is None or last - Lnow >= float(os.environ.get("CHAIN_DL", "0.05"))
                    or rnow < rlast * (1 - float(os.environ.get("CHAIN_DR", "1")))
                    or getattr(run, "chain_D", 0.0) >= float(os.environ.get("CHAIN_DD", "inf"))):
                run.chain = getattr(run, "chain", []) + [z.copy()]
                run.chain_L, run.chain_R, run.chain_D = Lnow, rnow, 0.0
            if np.hypot(*z).max() < float(os.environ["CSF_CAP"]):
                if not np.array_equal(run.chain[-1], z):  # the leaf may have just been saved by the spacing rule
                    run.chain.append(z.copy())
                np.save(os.path.join(ROOT, "words/out/csf", "chain_" + ",".join(f"{x:g}" for x in m) + ".npy"),
                        np.array(run.chain))
                print(f"  chain of {len(run.chain)} convex leaves saved (last inside the cap) at it {it}", flush=True)
                return
        if os.environ.get("CSF_CHAIN"):
            run.chain_D = getattr(run, "chain_D", 0.0) + dt * float(np.max(np.abs(v) * np.exp(psi)))
        z = z + dt * v * nrm
        if it % report == 0:
            zz = np.hstack([z, z[:, :1]])
            e = np.exp(psi)
            L = float(np.sum(np.linalg.norm(np.diff(zz, axis=1), axis=0) * 0.5 * (e + np.roll(e, -1))))
            rr = np.linalg.norm(z, axis=0)
            hist.append(L)
            print(f"  it {it:7d}  g-length {L:.6f}  (minus c12 {L - c12:+.2e})  |z| {rr.min():.4f}..{rr.max():.4f}  "
                  f"k_g min {kg.min():+.3e} max {kg.max():+.3e}  speed {np.max(np.abs(v)) * np.exp(psi).max():.1e}",
                  flush=True)
            if rr.max() < 1e-3:
                print("  -> runs into the 12-end (no closed geodesic in the swept region)")
                return
            if len(hist) > 6 and abs(hist[-1] - hist[-6]) < 1e-9:
                print(f"  -> stalls: closed geodesic of length {L:.6f}")
                return


if __name__ == "__main__":
    m = tuple(float(v) for v in sys.argv[1].split(","))
    run(m, float(sys.argv[2]) if len(sys.argv) > 2 else None)
