"""Topology check: does the refined orbit have the published free-group word?

Shape sphere (Montgomery): the equator is the set of collinear configurations; the three collision
points cut it into three arcs, arc i = "body i in the middle".  A syzygy (collinear instant) is a
crossing of arc i.  Reading a word: with one collision point at infinity the other two are the
punctures of the plane, the middle segment between them is one arc (M), the two outer rays are the
others (L, R).  Li-Liao: a = clockwise around the right puncture = down through R, up through M;
b = counter-clockwise around the left puncture = down through L, up through M; A, B inverses.  So
    a -> (R, M),  A -> (M, R),  b -> (L, M),  B -> (M, L).
Concatenating gives a cyclic syzygy sequence; removing stutters (two consecutive crossings of the
same arc, a loop that can be pulled back) gives a reduced cyclic sequence that determines the free
homotopy class.  We compare classes up to relabelling of the bodies (6 permutations), reversal
(time / orientation) and cyclic shift: canonical = lexicographically least such sequence.  The
pairing convention (down/up vs up/down) only shifts the cyclic sequence by one place, so it does
not matter.

The orbit side: integrate the refined orbit (tb3, double precision) forward over [0, T/2] and
backward over [-T/2, 0] (deviation from the true orbit grows over half a period only), sample the
signed area d12 x d23 at 32 leapfrog substeps per accepted step, and record the middle body (the
one opposite the longest side) at every sign change.  The start (Li-Liao isosceles collinear,
body 3 between bodies 1 and 2) is itself a syzygy with body 3 in the middle.

python -I topology.py <words_dir>   ->  data/work/topology.jsonl
"""
import csv
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tb3  # noqa: E402


def word_to_syzygy(word, b_ccw=True):
    R, M, L = 2, 3, 1
    m = {"a": (R, M), "A": (M, R)}
    if b_ccw:
        m.update({"b": (L, M), "B": (M, L)})
    else:
        m.update({"b": (M, L), "B": (L, M)})
    out = []
    for ch in word:
        out.extend(m[ch])
    return out


def reduce_cyclic(seq):
    s = list(seq)
    changed = True
    while changed and s:
        changed = False
        out = []
        for x in s:
            if out and out[-1] == x:
                out.pop()
                changed = True
            else:
                out.append(x)
        while len(out) >= 2 and out[0] == out[-1]:
            out = out[1:-1]
            changed = True
        s = out
    return s


def least_rotation(s):
    """Booth's algorithm: index of the lexicographically least rotation."""
    s2 = s + s
    n = len(s)
    f = [-1] * len(s2)
    k = 0
    for j in range(1, len(s2)):
        sj = s2[j]
        i = f[j - k - 1]
        while i != -1 and sj != s2[k + i + 1]:
            if sj < s2[k + i + 1]:
                k = j - i - 1
            i = f[i]
        if sj != s2[k + i + 1]:
            if sj < s2[k]:
                k = j
            f[j - k] = -1
        else:
            f[j - k] = i + 1
    return k % n if n else 0


def canonical(seq):
    best = None
    for perm in itertools.permutations((1, 2, 3)):
        mp_ = {1: perm[0], 2: perm[1], 3: perm[2]}
        for rev in (False, True):
            s = [mp_[x] for x in (seq[::-1] if rev else seq)]
            if not s:
                return ""
            k = least_rotation(s)
            r = s[k:] + s[:k]
            if best is None or r < best:
                best = r
    return "".join(str(x) for x in best)


def syzygies(v1, v2, T, tol=1e-13, nsub=32):
    """Vectorised over orbits.  Returns list of syzygy label lists (cyclic, starting at -T/2)."""
    n = len(v1)
    X0 = tb3.initial_state(np.concatenate([v1, v1]), np.concatenate([v2, v2]))
    B = -tb3.energy(X0)
    Z = tb3.to_rel(X0)
    G = 2 * n
    sgn = np.concatenate([np.ones(n), -np.ones(n)])
    tstop = np.concatenate([T, T]) / 2
    H = sgn * 0.05
    done = np.zeros(G, bool)
    ev = [[] for _ in range(G)]           # (t, label)
    # sign of the area just after t = 0 (the start is collinear)
    p = 1.0 / (2 * len(tb3.SEQ) - 1)
    prev_sign = np.zeros(G)
    it = 0
    while not done.all():
        it += 1
        idx = np.nonzero(~done)[0]
        h = H[idx]
        Y, D, _ = tb3.bs_step(Z[:, idx], h, B[idx])
        e = tb3.err_norm(Y, D)
        e = np.where(np.isfinite(e), e, np.inf)
        acc = e <= tol
        ia = idx[acc]
        if ia.size:
            # fine substeps from the old state with the same step: approximate trajectory
            Zs = Z[:, ia].copy()
            hh = h[acc] / nsub
            Bs = B[ia]
            K = tb3.kinetic_rel(Zs)
            dtf = 0.5 * hh / (K + Bs)
            Zs[0:6] += dtf * Zs[6:12]; Zs[12] += dtf
            for k in range(nsub):
                a, U, r = tb3.accel_rel(Zs[0:6])
                Zs[6:12] += (hh / U) * a
                K = tb3.kinetic_rel(Zs)
                dtf = (hh if k < nsub - 1 else 0.5 * hh) / (K + Bs)
                Zs[0:6] += dtf * Zs[6:12]; Zs[12] += dtf
                area = Zs[0] * Zs[3] - Zs[1] * Zs[2]
                sg = np.sign(area)
                tt = Zs[12]
                inside = np.abs(tt) <= tstop[ia]
                first = prev_sign[ia] == 0
                prev_sign[ia[first]] = sg[first]
                ch = inside & ~first & (sg != prev_sign[ia]) & (sg != 0)
                for j in np.nonzero(ch)[0]:
                    g = ia[j]
                    rr = (r[0][j], r[1][j], r[2][j])   # r12, r23, r31
                    lab = (3, 1, 2)[int(np.argmax(rr))]
                    ev[g].append((float(tt[j]), lab))
                prev_sign[ia[inside]] = np.where(sg[inside] != 0, sg[inside], prev_sign[ia[inside]])
            Z[:, ia] = Y[:, acc]
            fin = ia[np.abs(Z[12, ia]) >= tstop[ia]]
            done[fin] = True
        with np.errstate(divide="ignore"):
            fac = np.where(e > 0, 0.9 * (tol / e) ** p, 3.0)
        H[idx] = h * np.clip(fac, 0.2, 3.0)
        bad = idx[(np.abs(H[idx]) < 1e-12) | ~np.isfinite(H[idx])]
        done[bad] = True
    out = []
    for i in range(n):
        back = sorted(ev[n + i])          # times in (-T/2, 0), ascending
        fwd = sorted(ev[i])
        out.append([lab for _, lab in back] + [3] + [lab for _, lab in fwd])
    return out


def main():
    words = sys.argv[1]
    rows = {r["id"]: r for r in csv.DictReader(open(os.path.join(words, "data", "catalogue.csv"), encoding="utf-8"))}
    res = [json.loads(l) for l in open(os.path.join(words, "data", "work", "refine.jsonl"))]
    res = [r for r in res if "T" in r]
    outp = os.path.join(words, "data", "work", "topology.jsonl")
    done = set()
    if os.path.exists(outp):
        done = {json.loads(l)["id"] for l in open(outp)}
    res = [r for r in res if r["id"] not in done]
    res.sort(key=lambda r: float(r["T"]))
    B = 40
    for i in range(0, len(res), B):
        chunk = res[i:i + B]
        v1 = np.array([float(r["v1"]) for r in chunk]); v2 = np.array([float(r["v2"]) for r in chunk])
        T = np.array([float(r["T"]) for r in chunk])
        seqs = syzygies(v1, v2, T)
        with open(outp, "a") as f:
            for r, s in zip(chunk, seqs):
                red = reduce_cyclic(s)
                rec = dict(id=r["id"], n_syzygy=len(s), syzygy="".join(map(str, s)),
                           reduced="".join(map(str, red)), canon=canonical(red))
                w = rows[r["id"]]["word"]
                if w:
                    for ccw in (True, False):
                        ws = reduce_cyclic(word_to_syzygy(w, ccw))
                        rec["word_canon_ccw" if ccw else "word_canon_cw"] = canonical(ws)
                    rec["match"] = rec["word_canon_ccw"] == rec["canon"]
                    rec["match_cw"] = rec["word_canon_cw"] == rec["canon"]
                f.write(json.dumps(rec) + "\n")
        print(f"{i + len(chunk)}/{len(res)}", flush=True)


if __name__ == "__main__":
    main()
