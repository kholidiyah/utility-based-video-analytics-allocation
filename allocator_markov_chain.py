#!/usr/bin/env python3
# allocator_markov_chain.py

import argparse
import itertools
import numpy as np
import pandas as pd
from acc_interpolator_stream import Profiler, Knob


# -------------------- Helpers (same spirit as your alpha script) --------------------

def key_bounds(p: Profiler, k: Knob):
    df = p.df[
        (p.df.stream == k.stream) &
        (p.df.model == k.model) &
        (p.df.QP == k.QP) &
        (p.df.FPS == k.FPS) &
        (p.df.Resolution == k.Resolution)
    ]
    return float(df.Bitrate.min()), float(df.Bitrate.max())


def total_gpu(p: Profiler, alloc: dict):
    return sum(p.query(k, b)[2] for k, b in alloc.items())


def violates_latency(p: Profiler, k: Knob, b: float, sla_ms):
    if sla_ms is None:
        return False
    _, lat, _ = p.query(k, b)
    return lat > sla_ms


def violates_accmin(p: Profiler, k: Knob, b: float, acc_min_map):
    if not acc_min_map or k.stream not in acc_min_map:
        return False
    acc, _, _ = p.query(k, b)
    return acc < acc_min_map[k.stream]


def pareto_filter(p: Profiler, keys):
    stats = []
    for k in keys:
        mn, _ = key_bounds(p, k)
        acc, lat, gpu = p.query(k, mn)
        stats.append((k, mn, acc, lat, gpu))

    def dominates(A, B):
        _, ba, aa, la, ga = A
        _, bb, ab, lb, gb = B
        cond = (aa >= ab) and (la <= lb) and (ga <= gb) and (ba <= bb)
        strict = (aa > ab) or (la < lb) or (ga < gb) or (ba < bb)
        return cond and strict

    front = []
    for i, A in enumerate(stats):
        if not any(dominates(B, A) for j, B in enumerate(stats) if j != i):
            front.append(A[0])
    return front if front else keys


# -------------------- Markov chain utilities --------------------

def parse_matrix(P_str: str):
    """
    Format: "0.7 0.3 0.0; 0.2 0.6 0.2; 0.0 0.3 0.7"
    -> np.array([...])
    """
    rows = []
    for r in P_str.strip().split(";"):
        r = r.strip()
        if not r:
            continue
        rows.append([float(x) for x in r.split()])
    P = np.array(rows, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square matrix.")

    # normalize row-stochastic
    rs = P.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return P / rs


def steady_state(P: np.ndarray, tol=1e-12, iters=100000):
    """
    Compute stationary distribution mu s.t. mu = mu P, sum(mu)=1
    Power iteration.
    """
    n = P.shape[0]
    mu = np.ones(n, dtype=float) / n
    for _ in range(iters):
        mu2 = mu @ P
        if np.max(np.abs(mu2 - mu)) < tol:
            mu = mu2
            break
        mu = mu2
    mu = np.clip(mu, 0, None)
    s = mu.sum()
    return mu / s if s > 0 else np.ones(n) / n


# -------------------- Per-state allocator (maximize sum weighted accuracy) --------------------

def objective_accsum(p: Profiler, alloc: dict, omega, sla, acc_min):
    """
    Pure Markov reward per state:
      U = Σ_i ω_i * Acc_i(b_i)
    constraints handled as infeasible => -inf
    """
    U = 0.0
    for k, b in alloc.items():
        acc, _, _ = p.query(k, b)
        if violates_latency(p, k, b, None if sla is None else sla.get(k.stream)):
            return -np.inf
        if acc_min and violates_accmin(p, k, b, acc_min):
            return -np.inf
        w = 1.0 if omega is None else omega.get(k.stream, 1.0)
        U += w * float(acc)
    return float(U)


def greedy_fill_acc(p: Profiler, choice_keys, B: float, step: float, G: float, sla, acc_min, omega):
    """
    Start at min bitrate; add 'step' to stream giving best marginal weighted accuracy gain.
    """
    alloc, bmax = {}, {}
    for k in choice_keys:
        mn, mx = key_bounds(p, k)
        alloc[k] = mn
        bmax[k] = mx
        if violates_latency(p, k, mn, None if sla is None else sla.get(k.stream)):
            return None, -np.inf

    curB = sum(alloc.values())
    if curB > B or total_gpu(p, alloc) > G:
        return None, -np.inf

    # lift to accmin first
    if acc_min:
        changed = True
        while changed:
            changed = False
            for k in choice_keys:
                tgt = acc_min.get(k.stream)
                if tgt is None:
                    continue
                acc, _, _ = p.query(k, alloc[k])
                while acc < tgt and alloc[k] + step <= bmax[k] and curB + step <= B:
                    nb = alloc[k] + step
                    if violates_latency(p, k, nb, None if sla is None else sla.get(k.stream)):
                        break
                    tmp = dict(alloc); tmp[k] = nb
                    if total_gpu(p, tmp) > G:
                        break
                    alloc[k] = nb
                    curB += step
                    acc, _, _ = p.query(k, alloc[k])
                    changed = True
            if curB + step > B:
                break

        for k in choice_keys:
            tgt = acc_min.get(k.stream)
            if tgt is not None and p.query(k, alloc[k])[0] < tgt:
                return None, -np.inf

    # greedy fill remaining budget
    while curB + step <= B:
        best_k, best_gain = None, -1e18
        for k in choice_keys:
            if alloc[k] + step > bmax[k]:
                continue
            nb = alloc[k] + step
            if violates_latency(p, k, nb, None if sla is None else sla.get(k.stream)):
                continue
            tmp = dict(alloc); tmp[k] = nb
            if total_gpu(p, tmp) > G:
                continue

            acc1, _, _ = p.query(k, alloc[k])
            acc2, _, _ = p.query(k, nb)
            w = 1.0 if omega is None else omega.get(k.stream, 1.0)
            gain = w * (float(acc2) - float(acc1))

            if gain > best_gain:
                best_gain = gain
                best_k = k

        if best_k is None or best_gain <= 0:
            break
        alloc[best_k] += step
        curB += step

    U = objective_accsum(p, alloc, omega, sla, acc_min)
    return alloc, U


def optimize_state(p: Profiler, cand_by_stream: dict, B_state: float, step: float, G: float,
                   sla, acc_min, omega):
    """
    Enumerate knob combination across streams; for each choice do greedy fill.
    """
    streams_sorted = sorted(cand_by_stream.keys())
    best = None
    for choice in itertools.product(*(cand_by_stream[s] for s in streams_sorted)):
        alloc, U = greedy_fill_acc(p, choice, B_state, step, G, sla, acc_min, omega)
        if alloc is None:
            continue
        if best is None or U > best[1]:
            best = (alloc, U, choice)
    return best


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)

    ap.add_argument("--states", type=str, default="Low,Med,High")
    ap.add_argument("--P", type=str, required=True)
    ap.add_argument("--B_states", type=str, required=True)

    ap.add_argument("--G", type=float, default=1e9)
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--sla", type=str, default="")
    ap.add_argument("--omega", type=str, default="")
    ap.add_argument("--accmin", type=str, default="")
    ap.add_argument("--accmin_all", type=float, default=None)
    ap.add_argument("--allow_res", nargs="+", default=None)

    ap.add_argument("--out", type=str, default="alloc_markov_out.csv")
    ap.add_argument("--out_state_prefix", type=str, default="alloc_state_")
    ap.add_argument("--out_delim", type=str, default=",")

    args = ap.parse_args()
    p = Profiler(args.csv)

    # parse states & budgets
    state_names = [x.strip() for x in args.states.split(",") if x.strip()]
    B_states = [float(x) for x in args.B_states.split(",") if x.strip()]
    if len(state_names) != len(B_states):
        raise ValueError("--states and --B_states must have same length.")

    P = parse_matrix(args.P)
    if P.shape[0] != len(state_names):
        raise ValueError("Size of P must match number of states in --states.")
    mu = steady_state(P)

    # parse SLA & weights
    sla = None
    if args.sla:
        sla = {}
        for tok in args.sla.split(","):
            if tok:
                sid, val = tok.split(":")
                sla[int(sid)] = float(val)

    omega = None
    if args.omega:
        omega = {}
        for tok in args.omega.split(","):
            if tok:
                sid, val = tok.split(":")
                omega[int(sid)] = float(val)

    # parse accmin
    acc_min = {}
    if args.accmin_all is not None:
        acc_min["_global"] = float(args.accmin_all)
    if args.accmin:
        for tok in args.accmin.split(","):
            if tok:
                sid, val = tok.split(":")
                acc_min[int(sid)] = float(val)

    # candidates per stream
    allowed_res = set(args.allow_res) if args.allow_res else None
    cand = {}
    for (s, m, qp, fps, res), _ in p.df.groupby(["stream", "model", "QP", "FPS", "Resolution"]):
        if allowed_res and str(res) not in allowed_res:
            continue
        cand.setdefault(int(s), []).append(Knob(int(s), str(m), int(qp), int(fps), str(res)))

    # apply global accmin
    if "_global" in acc_min:
        g = acc_min.pop("_global")
        for sid in cand.keys():
            acc_min.setdefault(int(sid), float(g))

    if not cand:
        print("No candidates after resolution filtering. Check --allow_res.")
        return

    # pareto prune
    for sid in list(cand.keys()):
        cand[sid] = pareto_filter(p, cand[sid])

    # optimize per state
    per_state_best = []
    for idx, (sname, Bst) in enumerate(zip(state_names, B_states)):
        best = optimize_state(
            p, cand, B_state=Bst, step=args.step, G=args.G,
            sla=sla, acc_min=acc_min if acc_min else None, omega=omega
        )
        if best is None:
            print(f"[{sname}] No feasible allocation.")
            per_state_best.append((sname, Bst, None, -np.inf))
            continue

        alloc_best, U_best, _ = best
        per_state_best.append((sname, Bst, alloc_best, U_best))

        # save per-state table
        rows = []
        for k, b in alloc_best.items():
            acc, lat, gpu = p.query(k, b)
            w = 1.0 if omega is None else omega.get(k.stream, 1.0)
            rows.append({
                "state": sname,
                "mu_state": float(mu[idx]),
                "stream": k.stream,
                "model": k.model,
                "QP": k.QP,
                "FPS": k.FPS,
                "Resolution": k.Resolution,
                "Bitrate": round(float(b), 1),
                "Accuracy": round(float(acc), 6),
                "Latency(ms)": round(float(lat), 2),
                "GPU_load(%)": round(float(gpu), 2),
                "Utility": round(float(w * acc), 6),
            })
        pd.DataFrame(rows).sort_values(["stream"]).to_csv(
            f"{args.out_state_prefix}{sname}.csv", index=False, sep=args.out_delim
        )

    # ==================== Expected (Markov) aggregation across states ====================

    exp_contrib, exp_acc, exp_bw, exp_lat, exp_gpu = {}, {}, {}, {}, {}

    feasible_allocs = [x[2] for x in per_state_best if x[2] is not None]
    if not feasible_allocs:
        print("No feasible allocation in any state.")
        return

    for st_idx, (sname, Bst, alloc, Ust) in enumerate(per_state_best):
        if alloc is None:
            continue
        w_s = float(mu[st_idx])
        for k in alloc.keys():
            sid = k.stream
            b = float(alloc[k])
            acc, lat, gpu = p.query(k, b)
            w = 1.0 if omega is None else omega.get(sid, 1.0)

            exp_contrib[sid] = exp_contrib.get(sid, 0.0) + w_s * w * acc
            exp_acc[sid] = exp_acc.get(sid, 0.0) + w_s * acc
            exp_bw[sid] = exp_bw.get(sid, 0.0) + w_s * b
            exp_lat[sid] = exp_lat.get(sid, 0.0) + w_s * lat
            exp_gpu[sid] = exp_gpu.get(sid, 0.0) + w_s * gpu

    J_MC = sum(exp_contrib.values())
    denom = J_MC if abs(J_MC) > 1e-12 else 1.0

    out_rows = []
    for sid in sorted(exp_contrib.keys()):
        share = exp_contrib[sid] / denom          # 0..1
        out_rows.append({
            "stream": sid,
            "Expected_Bitrate": round(exp_bw[sid], 3),
            "Expected_Accuracy": round(exp_acc[sid], 6),
            "Expected_Latency(ms)": round(exp_lat[sid], 3),
            "Expected_GPU_load(%)": round(exp_gpu[sid], 3),
            "Utility": round(exp_contrib[sid], 6),
            "Utility_Share": round(share, 6),
        })

    # print summary
    print("=== Markov Chain Allocator (pure accuracy-utility) ===")
    print(f"States: {state_names}")
    print(f"mu (steady-state): {[round(x, 6) for x in mu.tolist()]}")
    print(f"J_MC (expected utility): {J_MC:.6f}")
    for st_idx, (sname, Bst, alloc, Ust) in enumerate(per_state_best):
        if alloc is None:
            continue
        bw_used = sum(alloc.values())
        gpu_used = total_gpu(p, alloc)
        print(f"[{sname}] mu={mu[st_idx]:.4f} | B={Bst:.1f} | BW_used={bw_used:.1f} | GPU={gpu_used:.2f} | U_state={Ust:.6f}")

    # save expected table ONCE
    pd.DataFrame(out_rows).sort_values(["stream"]).to_csv(args.out, index=False, sep=args.out_delim)
    print(f"Saved expected Markov output: {args.out}")
    print("Saved per-state outputs:", ", ".join([f"{args.out_state_prefix}{n}.csv" for n in state_names]))


if __name__ == "__main__":
    main()
