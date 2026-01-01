#!/usr/bin/env python3
# allocator_lyapunov_fast.py
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from acc_interpolator_stream import Profiler, Knob


# -------------------- Markov helpers (optional state sequence) --------------------

def parse_matrix(P_str: str) -> np.ndarray:
    rows = []
    for r in P_str.strip().split(";"):
        r = r.strip()
        if r:
            rows.append([float(x) for x in r.split()])
    P = np.array(rows, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square matrix.")
    rs = P.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return P / rs

def steady_state(P: np.ndarray, tol=1e-12, iters=200000) -> np.ndarray:
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

def sample_state_sequence(P: np.ndarray, T: int, s0: int = 0, seed: int = 0) -> List[int]:
    rng = np.random.default_rng(seed)
    n = P.shape[0]
    s = int(np.clip(s0, 0, n - 1))
    seq = [s]
    for _ in range(T - 1):
        s = rng.choice(n, p=P[s])
        seq.append(int(s))
    return seq


# -------------------- Profiler helpers --------------------

def key_bounds(p: Profiler, k: Knob) -> Tuple[float, float]:
    df = p.df[
        (p.df.stream == k.stream) &
        (p.df.model == k.model) &
        (p.df.QP == k.QP) &
        (p.df.FPS == k.FPS) &
        (p.df.Resolution == k.Resolution)
    ]
    return float(df.Bitrate.min()), float(df.Bitrate.max())

def build_candidates(p: Profiler, allow_res: Optional[List[str]]) -> Dict[int, List[Knob]]:
    allowed = set(allow_res) if allow_res else None
    cand: Dict[int, List[Knob]] = {}
    for (s, m, qp, fps, res), _ in p.df.groupby(["stream", "model", "QP", "FPS", "Resolution"]):
        if allowed and str(res) not in allowed:
            continue
        cand.setdefault(int(s), []).append(Knob(int(s), str(m), int(qp), int(fps), str(res)))
    return cand

def pareto_filter_minpoint(p: Profiler, keys: List[Knob]) -> List[Knob]:
    """
    Cheap pareto pruning based on metrics at minimum bitrate:
    keep knobs not dominated in (acc high, lat low, gpu low, bmin low).
    """
    stats = []
    for k in keys:
        mn, _ = key_bounds(p, k)
        acc, lat, gpu = p.query(k, mn)
        stats.append((k, mn, float(acc), float(lat), float(gpu)))

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


# -------------------- Actions (discretized) --------------------

@dataclass(frozen=True)
class Action:
    stream: int
    knob: Knob
    bitrate: float
    acc: float
    lat: float
    gpu: float

def discretize_actions_for_stream(
    p: Profiler,
    knobs: List[Knob],
    step: float,
    max_actions_per_stream: int = 300
) -> List[Action]:
    """
    Build discrete actions: (knob, bitrate level) with precomputed metrics.
    Then prune simple envelope to keep the list small.
    """
    actions: List[Action] = []

    for k in knobs:
        mn, mx = key_bounds(p, k)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= 0:
            continue

        # discrete levels
        levels = list(np.arange(mn, mx + 1e-9, step, dtype=float))
        if not levels:
            levels = [mn]
        if abs(levels[-1] - mx) > 1e-6:
            levels.append(mx)

        for b in levels:
            acc, lat, gpu = p.query(k, float(b))
            actions.append(Action(
                stream=k.stream, knob=k,
                bitrate=float(b),
                acc=float(acc), lat=float(lat), gpu=float(gpu)
            ))

    if not actions:
        return []

    # Sort by bitrate asc; for same bitrate prefer higher acc, lower lat/gpu
    actions.sort(key=lambda a: (a.bitrate, -a.acc, a.lat, a.gpu))

    # Envelope pruning: keep actions that improve acc as bitrate increases
    pruned: List[Action] = []
    best_acc = -1e18
    for a in actions:
        if a.acc > best_acc + 1e-9:
            pruned.append(a)
            best_acc = a.acc

    # cap list size (keep spread)
    if len(pruned) > max_actions_per_stream:
        idxs = np.linspace(0, len(pruned) - 1, num=max_actions_per_stream, dtype=int)
        pruned = [pruned[i] for i in idxs]

    return pruned


# -------------------- Parsing maps --------------------

def parse_map(s: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    if not s:
        return out
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        sid, val = tok.split(":")
        out[int(sid)] = float(val)
    return out


# -------------------- Lyapunov DPP objective --------------------

def action_value(
    a: Action,
    omega: Optional[Dict[int, float]],
    sla: Optional[Dict[int, float]],
    acc_min: Optional[Dict[int, float]],
    Y: float,
    Zacc: Dict[int, float],
    Zsla: Dict[int, float],
    V: float
) -> float:
    """
    DPP per-action value (dropping constant +Y*B_target):
      V*w*acc - Y*b - Zacc*(accmin-acc) - Zsla*(lat-sla)
    """
    sid = a.stream
    w = 1.0 if omega is None else float(omega.get(sid, 1.0))
    val = V * w * a.acc - Y * a.bitrate

    if acc_min and sid in acc_min:
        val -= float(Zacc.get(sid, 0.0)) * (float(acc_min[sid]) - a.acc)

    if sla and sid in sla:
        val -= float(Zsla.get(sid, 0.0)) * (a.lat - float(sla[sid]))

    return float(val)


def pick_actions_greedy_upgrade(
    actions_by_stream: Dict[int, List[Action]],
    B_hard: float,
    G: float,
    omega: Optional[Dict[int, float]],
    sla: Optional[Dict[int, float]],
    acc_min: Optional[Dict[int, float]],
    Y: float,
    Zacc: Dict[int, float],
    Zsla: Dict[int, float],
    V: float,
    hard_sla: bool = False,
    hard_accmin: bool = False
) -> Optional[Dict[int, Action]]:
    """
    Fast approximate multi-choice knapsack:
      - pick one action per stream
      - meet HARD constraints: sum bitrate <= B_hard and sum gpu <= G
      - maximize sum action_value()

    Strategy:
      1) baseline = min-bitrate feasible action per stream
      2) iterative upgrades with best (delta_value / delta_bitrate)
    """
    streams = sorted(actions_by_stream.keys())
    chosen: Dict[int, Action] = {}

    def feasible_action(a: Action) -> bool:
        if hard_accmin and acc_min and a.stream in acc_min and a.acc < acc_min[a.stream]:
            return False
        if hard_sla and sla and a.stream in sla and a.lat > sla[a.stream]:
            return False
        return True

    # baseline: minimal bitrate per stream
    for sid in streams:
        cand = [a for a in actions_by_stream[sid] if feasible_action(a)]
        if not cand:
            return None
        cand.sort(key=lambda a: (a.bitrate, -action_value(a, omega, sla, acc_min, Y, Zacc, Zsla, V)))
        chosen[sid] = cand[0]

    sum_b = sum(a.bitrate for a in chosen.values())
    sum_g = sum(a.gpu for a in chosen.values())
    if sum_b > B_hard or sum_g > G:
        return None

    # sort each stream actions by bitrate
    lists: Dict[int, List[Action]] = {}
    idx_map: Dict[int, int] = {}
    for sid in streams:
        lst = [a for a in actions_by_stream[sid] if feasible_action(a)]
        lst.sort(key=lambda a: a.bitrate)
        lists[sid] = lst
        # find chosen index
        try:
            idx_map[sid] = lst.index(chosen[sid])
        except ValueError:
            # fallback by bitrate
            i0 = int(np.argmin([abs(a.bitrate - chosen[sid].bitrate) for a in lst]))
            idx_map[sid] = i0

    # upgrades loop
    while True:
        best_sid = None
        best_next = None
        best_eff = -1e18

        cur_val = {sid: action_value(chosen[sid], omega, sla, acc_min, Y, Zacc, Zsla, V) for sid in streams}

        for sid in streams:
            lst = lists[sid]
            i = idx_map[sid]
            if i >= len(lst) - 1:
                continue

            cur = chosen[sid]
            # check a few next actions (local lookahead)
            for j in range(i + 1, min(len(lst), i + 10)):
                nxt = lst[j]
                db = nxt.bitrate - cur.bitrate
                dg = nxt.gpu - cur.gpu
                if db <= 1e-12 and dg <= 1e-12:
                    continue
                if sum_b + db > B_hard:
                    continue
                if sum_g + dg > G:
                    continue

                dv = action_value(nxt, omega, sla, acc_min, Y, Zacc, Zsla, V) - cur_val[sid]
                if dv <= 1e-12:
                    continue

                eff = dv / max(db, 1e-12)
                if eff > best_eff + 1e-12:
                    best_eff = eff
                    best_sid = sid
                    best_next = nxt

        if best_sid is None or best_next is None:
            break

        # apply upgrade
        cur = chosen[best_sid]
        sum_b += (best_next.bitrate - cur.bitrate)
        sum_g += (best_next.gpu - cur.gpu)
        chosen[best_sid] = best_next
        idx_map[best_sid] = min(idx_map[best_sid] + 1, len(lists[best_sid]) - 1)

    return chosen


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)

    ap.add_argument("--states", type=str, default="Low,Med,High")
    ap.add_argument("--B_states", type=str, required=True)
    ap.add_argument("--P", type=str, default="")

    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--s0", type=int, default=0)

    ap.add_argument("--G", type=float, default=1e9)
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--sla", type=str, default="")
    ap.add_argument("--omega", type=str, default="")
    ap.add_argument("--accmin", type=str, default="")
    ap.add_argument("--accmin_all", type=float, default=None)
    ap.add_argument("--allow_res", nargs="+", default=None)

    ap.add_argument("--V", type=float, default=100.0)
    ap.add_argument("--B_target", type=float, default=None)

    ap.add_argument("--hard_sla", action="store_true")
    ap.add_argument("--hard_accmin", action="store_true")

    ap.add_argument("--out", type=str, default="lyapunov_expected.csv")
    ap.add_argument("--out_trace", type=str, default="lyapunov_trace.csv")
    ap.add_argument("--out_delim", type=str, default=",")
    ap.add_argument("--progress_every", type=int, default=200)

    args = ap.parse_args()
    p = Profiler(args.csv)

    # states
    state_names = [x.strip() for x in args.states.split(",") if x.strip()]
    B_states = [float(x) for x in args.B_states.split(",") if x.strip()]
    if len(state_names) != len(B_states):
        raise ValueError("--states and --B_states must have same length.")
    nS = len(state_names)

    # Markov sequence (optional)
    if args.P.strip():
        P = parse_matrix(args.P)
        if P.shape[0] != nS:
            raise ValueError("Size of P must match number of states.")
        mu = steady_state(P)
        state_seq = sample_state_sequence(P, args.T, s0=args.s0, seed=args.seed)
    else:
        mu = np.ones(nS) / nS
        state_seq = [(t % nS) for t in range(args.T)]

    # maps
    sla = parse_map(args.sla) if args.sla else None
    omega = parse_map(args.omega) if args.omega else None

    acc_min: Dict[int, float] = {}
    if args.accmin_all is not None:
        for_global = float(args.accmin_all)
    else:
        for_global = None
    if args.accmin:
        acc_min.update(parse_map(args.accmin))
    if for_global is not None:
        # apply later once we know which streams exist
        pass

    # candidates
    cand_knobs = build_candidates(p, args.allow_res)
    if not cand_knobs:
        print("No candidates after resolution filtering. Check --allow_res.")
        return

    # apply global accmin
    if for_global is not None:
        for sid in cand_knobs.keys():
            acc_min.setdefault(int(sid), float(for_global))
    if not acc_min:
        acc_min = None  # keep it clean

    # pareto prune
    for sid in list(cand_knobs.keys()):
        cand_knobs[sid] = pareto_filter_minpoint(p, cand_knobs[sid])

    # discretize actions per stream (THIS makes it fast)
    actions_by_stream: Dict[int, List[Action]] = {}
    for sid in sorted(cand_knobs.keys()):
        acts = discretize_actions_for_stream(p, cand_knobs[sid], step=args.step, max_actions_per_stream=250)
        if not acts:
            print(f"Stream {sid}: no actions after discretization. Check data / step.")
            return
        actions_by_stream[sid] = acts

    # choose B_target
    if args.B_target is not None:
        B_target = float(args.B_target)
    else:
        B_target = float(np.dot(mu, np.array(B_states, dtype=float)))

    # virtual queues
    Y = 0.0
    Zacc = {sid: 0.0 for sid in actions_by_stream.keys()}
    Zsla = {sid: 0.0 for sid in actions_by_stream.keys()}

    # accumulators (time-average)
    sumU = {sid: 0.0 for sid in actions_by_stream.keys()}
    sumAcc = {sid: 0.0 for sid in actions_by_stream.keys()}
    sumBw = {sid: 0.0 for sid in actions_by_stream.keys()}
    sumLat = {sid: 0.0 for sid in actions_by_stream.keys()}
    sumGpu = {sid: 0.0 for sid in actions_by_stream.keys()}
    feasible_slots = 0

    trace_rows = []

    print("=== Lyapunov (fast DPP) start ===")
    print(f"T={args.T} | V={args.V} | B_target={B_target:.3f} | hard_sla={args.hard_sla} | hard_accmin={args.hard_accmin}")
    print(f"States={state_names} | mu={ [round(x,6) for x in mu.tolist()] }")
    print(f"Streams={sorted(actions_by_stream.keys())}")
    print("-------------------------------------------------------")

    for t in range(args.T):
        s = int(state_seq[t])
        B_hard = float(B_states[s])

        chosen = pick_actions_greedy_upgrade(
            actions_by_stream=actions_by_stream,
            B_hard=B_hard,
            G=args.G,
            omega=omega,
            sla=sla,
            acc_min=acc_min,
            Y=Y,
            Zacc=Zacc,
            Zsla=Zsla,
            V=args.V,
            hard_sla=args.hard_sla,
            hard_accmin=args.hard_accmin
        )

        if chosen is None:
            trace_rows.append({"t": t, "state": state_names[s], "B_state": B_hard, "feasible": 0})
            continue

        feasible_slots += 1

        # compute slot totals + update queues
        sum_b = float(sum(a.bitrate for a in chosen.values()))
        sum_g = float(sum(a.gpu for a in chosen.values()))

        # Y queue for average budget (virtual)
        Y = max(0.0, Y + (sum_b - B_target))

        # per stream update
        for sid, a in chosen.items():
            w = 1.0 if omega is None else float(omega.get(sid, 1.0))
            U_i = w * a.acc  # reward

            sumU[sid] += U_i
            sumAcc[sid] += a.acc
            sumBw[sid] += a.bitrate
            sumLat[sid] += a.lat
            sumGpu[sid] += a.gpu

            if acc_min and sid in acc_min:
                Zacc[sid] = max(0.0, Zacc[sid] + (float(acc_min[sid]) - a.acc))
            if sla and sid in sla:
                Zsla[sid] = max(0.0, Zsla[sid] + (a.lat - float(sla[sid])))

        trace_rows.append({
            "t": t,
            "state": state_names[s],
            "B_state": B_hard,
            "BW_used": round(sum_b, 3),
            "GPU_used": round(sum_g, 3),
            "Y_queue": round(Y, 6),
            "feasible": 1
        })

        if args.progress_every > 0 and (t + 1) % args.progress_every == 0:
            print(f"[t={t+1}] feasible={feasible_slots} | state={state_names[s]} | BW={sum_b:.1f}/{B_hard:.1f} | Y={Y:.2f}")

    if feasible_slots == 0:
        print("No feasible slot found. Check constraints / budgets / step.")
        pd.DataFrame(trace_rows).to_csv(args.out_trace, index=False, sep=args.out_delim)
        return

    # build expected / time-average table
    U_total_bar = sum(sumU.values()) / feasible_slots
    denom = U_total_bar if abs(U_total_bar) > 1e-12 else 1.0

    out_rows = []
    for sid in sorted(sumU.keys()):
        Ubar = sumU[sid] / feasible_slots
        share = Ubar / denom
        out_rows.append({
            "stream": sid,
            "Expected_Bitrate": round(sumBw[sid] / feasible_slots, 3),
            "Expected_Accuracy": round(sumAcc[sid] / feasible_slots, 6),
            "Expected_Latency(ms)": round(sumLat[sid] / feasible_slots, 3),
            "Expected_GPU_load(%)": round(sumGpu[sid] / feasible_slots, 3),
            "Utility": round(Ubar, 6),
            "Utility_Share": round(share, 6),
            "Utility_Share(%)": round(100.0 * share, 4),
        })

    pd.DataFrame(out_rows).sort_values(["stream"]).to_csv(args.out, index=False, sep=args.out_delim)
    pd.DataFrame(trace_rows).to_csv(args.out_trace, index=False, sep=args.out_delim)

    print("=== Done ===")
    print(f"feasible_slots={feasible_slots}/{args.T}")
    print(f"Saved expected: {args.out}")
    print(f"Saved trace:    {args.out_trace}")


if __name__ == "__main__":
    main()
