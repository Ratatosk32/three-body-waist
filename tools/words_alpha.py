"""Continuation of symmetric orbits in the exponent a of U_a = sum 1/r^a, a = 1 -> 2.
Per orbit: Newton on (vx, vy) with rho = 0 and w = 0 at the k = 3 half-period syzygy (as words_newton.py); linear
predictor in a; at every step the canonical syzygy word, the JM length S = int 2K dt over the period, E and
S* = S |E|^((2 - a) / (2a)).
    python -I tools/words_alpha.py <chunk> <nchunks>  ->  words/out/alpha_<chunk>.tsv"""
import csv
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "words", "tools"))
import topology as topo  # noqa: E402

PAIRS = ((0, 1), (1, 2), (2, 0))
AS = [1 + 0.025 * k for k in range(41)]


def make_rhs(a):
    def rhs(t, y):
        q = y[:6].reshape(3, 2)
        acc = np.zeros((3, 2))
        for i, j in PAIRS:
            d = q[j] - q[i]
            f = a * d / (d @ d) ** (a / 2 + 1)
            acc[i] += f
            acc[j] -= f
        v = y[6:12]
        return np.concatenate([v, acc.ravel(), [v @ v]])  # last: d/dt int 2K dt (masses 1)
    return rhs


def area(t, y):
    return (y[2] - y[0]) * (y[5] - y[1]) - (y[3] - y[1]) * (y[4] - y[0])


def y0_of(vx, vy):
    return np.array([-1, 0, 1, 0, 0, 0, vx, vy, vx, vy, -2 * vx, -2 * vy, 0.0])


def energy(y, a):
    q = y[:6].reshape(3, 2)
    v = y[6:12]
    return 0.5 * v @ v - sum(1 / np.linalg.norm(q[i] - q[j]) ** a for i, j in PAIRS)


def middle(y):
    q = y[:6].reshape(3, 2)
    r = [np.linalg.norm(q[1] - q[2]), np.linalg.norm(q[2] - q[0]), np.linalg.norm(q[0] - q[1])]
    return int(np.argmax(r))


def resid(y):
    q = y[:6].reshape(3, 2)
    v = y[6:12].reshape(3, 2)
    d0, d1 = np.linalg.norm(q[2] - q[0]), np.linalg.norm(q[2] - q[1])
    u = (q[1] - q[0]) / np.linalg.norm(q[1] - q[0])
    dv = v[0] - v[1]
    return np.array([(d0 - d1) / (d0 + d1), (dv @ u) / np.linalg.norm(v)])


def half(a, vx, vy, t_est):
    sol = solve_ivp(make_rhs(a), (0, t_est * 1.08 + 0.5), y0_of(vx, vy), method="DOP853", rtol=1e-12, atol=1e-12,
                    events=area)
    best = None
    for t, y in zip(sol.t_events[0], sol.y_events[0]):
        if t > 1e-6 and middle(y) == 2 and (best is None or abs(t - t_est) < abs(best[0] - t_est)):
            best = (t, y)
    return best


def solve(a, vx, vy, t):
    st = {"t": t}

    def F(p):
        hp = half(a, p[0], p[1], st["t"])
        if hp is None:
            return np.array([1.0, 1.0])
        st["t"] = hp[0]
        return resid(hp[1])

    r = root(F, [vx, vy], method="hybr", options={"xtol": 1e-13, "eps": 1e-9})
    res = float(np.max(np.abs(F(r.x))))
    return r.x[0], r.x[1], st["t"], res


def measure(a, vx, vy, th):
    T = 2 * th
    y0 = y0_of(vx, vy)
    sol = solve_ivp(make_rhs(a), (0, T), y0, method="DOP853", rtol=1e-12, atol=1e-12, events=area)
    eps = 1e-6 * T
    seq = [3] + [middle(y) + 1 for t, y in zip(sol.t_events[0], sol.y_events[0]) if eps < t < T - eps]
    S = sol.y[12, -1]
    E = energy(y0, a)
    Sst = S * abs(E) ** ((2 - a) / (2 * a))
    return E, S, Sst, topo.canonical(topo.reduce_cyclic(seq)), len(seq)


def sample():
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "words/data/orbits.csv"), encoding="utf-8"))
            if r["converged"] == "True" and not r["duplicate_of"] and r["Tstar"] and r["word"]]
    cnt = {}
    for r in rows:  # canonical syzygy class from the published word (catalogue convention, b ccw)
        r["syzygy_canon"] = topo.canonical(topo.reduce_cyclic(topo.word_to_syzygy(r["word"])))
        cnt[r["syzygy_canon"]] = cnt.get(r["syzygy_canon"], 0) + 1
    rows = [r for r in rows if float(r["Tstar"]) <= 60 and cnt[r["syzygy_canon"]] == 1]
    rows.sort(key=lambda r: float(r["Tstar"]))
    return rows[:40]


def main():
    k, K = int(sys.argv[1]), int(sys.argv[2])
    out = open(os.path.join(ROOT, f"words/out/alpha_{k}.tsv"), "w", encoding="utf-8")
    out.write("id\tTstar_cat\ta\tvx\tvy\tT\tE\tS\tSstar\tres\tcanon_ok\tn_syz\tnote\n")
    for r in sample()[k::K]:
        canon0 = r["syzygy_canon"]
        x = np.array([float(r["v1"]), float(r["v2"]), float(r["T"]) / 2])
        vx, vy, th, res = solve(1.0, *x)
        E, S, Sst, canon, n = measure(1.0, vx, vy, th)
        out.write(f"{r['id']}\t{r['Tstar']}\t1.000\t{vx:.12f}\t{vy:.12f}\t{2 * th:.10f}\t{E:.10e}\t{S:.10f}\t"
                  f"{Sst:.10f}\t{res:.1e}\t{canon == canon0}\t{n}\tstart\n")
        out.flush()
        if res > 1e-9 or canon != canon0:
            continue
        # adaptive steps: a step is accepted when Newton converges, the word is kept and S* moves by < 5 %;
        # otherwise the step is halved (down to 1e-4, then the branch stops); checkpoints on the 0.025 grid
        prev = [(1.0, np.array([vx, vy, th]), Sst)]
        a_cur, h, stop = 1.0, 0.005, ""
        for a_t in AS[1:]:
            while a_cur < a_t - 1e-12 and not stop:
                a = min(a_cur + h, a_t)
                if len(prev) >= 2:
                    (a1, x1, _), (a2, x2, _) = prev[-2], prev[-1]
                    guess = x2 + (x2 - x1) * (a - a2) / (a2 - a1)
                else:
                    guess = prev[-1][1]
                try:
                    vx, vy, th, res = solve(a, *guess)
                    ok = res <= 1e-9
                    if ok:
                        E, S, Sst, canon, n = measure(a, vx, vy, th)
                        ok = canon == canon0 and abs(Sst - prev[-1][2]) < 0.05 * prev[-1][2]
                except Exception:  # noqa: BLE001
                    ok = False
                if ok:
                    prev.append((a, np.array([vx, vy, th]), Sst))
                    a_cur = a
                    h = min(h * 1.5, 0.025)
                    if abs(a - a_t) < 1e-12:
                        out.write(f"{r['id']}\t{r['Tstar']}\t{a:.3f}\t{vx:.12f}\t{vy:.12f}\t{2 * th:.10f}\t{E:.10e}\t"
                                  f"{S:.10f}\t{Sst:.10f}\t{res:.1e}\tTrue\t{n}\t\n")
                        out.flush()
                else:
                    h /= 2
                    if h < 1e-4:
                        stop = f"stopped at a {a_cur:.4f} (fold or jump)"
            if stop:
                out.write(f"{r['id']}\t{r['Tstar']}\t{a_cur:.4f}\t\t\t\t\t\t\t\t\t\t{stop}\n")
                out.flush()
                break
    out.close()


if __name__ == "__main__":
    main()
