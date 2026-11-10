"""Exact Gaussian curvature of Montgomery's a = 2 metric g = 2U g0 on the shape sphere, any masses (m1, m2, m3).
Hopf coordinates at I = 1 (mass Jacobi z1 = sqrt(mu12)(q2 - q1), z2 = sqrt(mu3)(q3 - cm12), w = (|z1|^2 - |z2|^2,
2 z1 conj z2) on the unit sphere): every r_ij^2 is affine in w, r^2 = a + b . w, so U = sum c_k / (a_k + b_k . w) and
the sphere Laplacian of log U is exact: Lap_S F = Lap F - x^T Hess F x - 2 x . grad F at |x| = 1.
K = (4 - 2 Lap_S log U) / (2 U)   (g0 = round sphere of radius 1/2, Lap_0 = 4 Lap_S);  dA = (U / 2) dA_unit.
    python -I tools/words_curv_exact.py check   -> agreement with the finite-difference K and Gauss-Bonnet
    python -I tools/words_curv_exact.py map m1,m2,m3 [...]   -> where K > 0, its area and total"""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def coeffs(m):
    """(c_k, a_k, b_k) for the pairs 12, 13, 23 with r_k^2 = a_k + b_k . w."""
    m1, m2, m3 = m
    M12, M = m1 + m2, m1 + m2 + m3
    mu12, mu3 = m1 * m2 / M12, m3 * M12 / M
    out = [(m1 * m2, 1 / (2 * mu12), np.array([1 / (2 * mu12), 0.0, 0.0]))]
    for c, B in ((m1 * m3, m2 / (M12 * math.sqrt(mu12))), (m2 * m3, -m1 / (M12 * math.sqrt(mu12)))):
        A = 1 / math.sqrt(mu3)
        out.append((c, (A * A + B * B) / 2, np.array([(B * B - A * A) / 2, A * B, 0.0])))
    return out


def U_K(W, m):
    """W: array (3, ...) of unit vectors. Returns U and the exact K."""
    x = W.reshape(3, -1)
    U = np.zeros(x.shape[1])
    gU = np.zeros_like(x)
    HU = np.zeros((3, 3, x.shape[1]))
    LU = np.zeros(x.shape[1])
    for c, a, b in coeffs(m):
        g = a + b @ x
        U += c / g
        gU -= c * b[:, None] / g ** 2
        HU += 2 * c * np.einsum("i,j,k->ijk", b, b, 1 / g ** 3)
        LU += 2 * c * (b @ b) / g ** 3
    gF = gU / U
    HF = HU / U - np.einsum("ik,jk->ijk", gU, gU) / U ** 2
    LF = LU / U - np.sum(gU * gU, 0) / U ** 2
    LS = LF - np.einsum("ik,ijk,jk->k", x, HF, x) - 2 * np.sum(x * gF, 0)
    K = (4 - 2 * LS) / (2 * U)
    return U.reshape(W.shape[1:]), K.reshape(W.shape[1:])


def grid(N=1600):
    th = (np.arange(N) + 0.5) * math.pi / N
    ph = (np.arange(2 * N) + 0.5) * math.pi / N
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    W = np.array([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)])
    dAu = np.sin(TH) * (math.pi / N) ** 2
    return W, dAu


def collisions(m):
    """Unit vectors of the three collision points: where r_k^2 = a_k + b_k . w vanishes on the sphere."""
    out = []
    for c, a, b in coeffs(m):
        nb = np.linalg.norm(b)
        out.append(-b / nb)  # a = |b| for a collision on the sphere: r^2 = |b| (1 + bhat . w) -> 0 at w = -bhat
    return out


def check():
    import words_pants_q as PQ
    for m in ((1, 1, 1), (1, 1, 0.5), (1, 1, 2)):
        for c, a, b in coeffs(m):
            assert abs(a - np.linalg.norm(b)) < 1e-12, (m, a, np.linalg.norm(b))
        W, dAu = grid()
        U, K = U_K(W, m)
        Kfd, dA, mask = PQ.field(m[2])
        Ufd = PQ.U_grid(*W, m[2])
        d = min(np.min(np.arccos(np.clip(np.tensordot(cv, W, 1), -1, 1))) for cv in collisions(m))
        dist = np.min([np.arccos(np.clip(np.tensordot(cv, W, 1), -1, 1)) for cv in collisions(m)], axis=0)
        far = dist > 0.1
        print(f"m {m}: U vs words_pants_q max rel {np.max(np.abs(U / Ufd - 1)):.1e}; K exact vs FD (dist > 0.1) max "
              f"{np.max(np.abs(K - Kfd)[far]):.1e}; total K dA (cut 0.01) {np.sum((K * U / 2 * dAu)[dist > 0.01]):.4f}"
              f"; max K (dist > 0.01) {np.max(K[dist > 0.01]):+.2e}")


def kmap(m):
    W, dAu = grid()
    U, K = U_K(W, m)
    dA = U / 2 * dAu
    cols = collisions(m)
    dist = np.array([np.arccos(np.clip(np.tensordot(cv, W, 1), -1, 1)) for cv in cols])
    ok = dist.min(0) > 1e-3
    pos = ok & (K > 0)
    nearest = np.argmin(dist, axis=0)
    line = (f"m {m}: K max {np.max(K[ok]):+.3e}, min {np.min(K[ok]):+.3e}; K > 0: area {np.sum(dA[pos]):.4f}, "
            f"total {np.sum((K * dA)[pos]):+.5f}")
    for k, name in enumerate(("12", "13", "23")):
        s = pos & (nearest == k)
        if s.any():
            line += f"; nearest {name}: total {np.sum((K * dA)[s]):+.5f}, closest unit distance {dist[k][s].min():.4f}"
    print(line, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    else:
        for a in sys.argv[2:]:
            kmap(tuple(float(v) for v in a.split(",")))
