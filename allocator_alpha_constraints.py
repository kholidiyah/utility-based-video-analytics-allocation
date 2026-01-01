#!/usr/bin/env python3
import argparse, itertools, numpy as np, pandas as pd
from acc_interpolator_stream import Profiler, Knob

# ====================  Globals (GradFair normalization) ====================

GRADFAIR_UMAX = 1.0  # will be set in main() when u_mode=gradfair


def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


# ====================  (A) Utility & Welfare (sesuai Eq. 8.1) ====================

def welfare_component(u, alpha=1.0, eps=1e-12):
    x = np.clip(np.asarray(u, dtype=float), eps, None)
    if np.isclose(alpha, 1.0):
        out = np.log(x)
    else:
        out = np.power(x, (1.0 - alpha)) / (1.0 - alpha)
    return float(out) if np.ndim(u) == 0 else out


def u_from_acc(acc):
    return float(acc)


def u_from_powerlaw(bitrate, ai, gamma):
    bitrate = max(float(bitrate), 0.0)
    gamma = float(gamma)
    ai = max(float(ai), 1e-12)
    return (bitrate ** gamma) / ai


# -------- GradFair: pakai Prop. 8.1 sebagai skala, dikalikan gradien akurasi -----

def grad_accuracy(acc, bitrate):
    return float(acc) / max(float(bitrate), 1e-12)


def u_alpha_closedform(grad_acc_i, alpha, B_total, n_streams, ai_i, denom_sum=None):
    """
    Closed-form scaling ala Prop. 8.1 (tanpa normalisasi [0,1]):
      - alpha = 1:  CF_i = B / (n * a_i)
      - alpha != 1: CF_i = B * a_i^(-1/alpha) / sum_j a_j^(1-1/alpha)
    Utility GradFair mentah: u_raw = grad_acc_i * CF_i.
    Normalisasi ke [0,1] dilakukan terpisah dengan GRADFAIR_UMAX.
    """
    ai_i = max(float(ai_i), 1e-12)
    if np.isclose(alpha, 1.0):
        base = B_total / (max(n_streams, 1) * ai_i)
    else:
        denom = max(float(denom_sum), 1e-12)
        base = (B_total * (ai_i ** (-1.0 / alpha))) / denom
    return grad_acc_i * base


def gradfair_umax(alpha, B_total, ai_map, bmin_global):
    """
    Upper bound konservatif untuk u_raw GradFair guna normalisasi ke [0,1].

    Asumsi:
      - Acc_i(b) <= 1 untuk semua i
      - g_i(b) = Acc_i(b)/b <= 1 / b_min_global
      - a_i diambil dari ai_map
    """
    if bmin_global <= 0 or not np.isfinite(bmin_global):
        return 1.0

    if not ai_map:
        ai_map = {0: 1.0}

    a_vals = [max(a, 1e-12) for a in ai_map.values()]
    a_min = min(a_vals)
    n = max(len(a_vals), 1)

    if np.isclose(alpha, 1.0):
        # CF_i = B / (n * a_i), maksimum ketika a_i = a_min
        CF_max = B_total / (n * a_min)
    else:
        # CF_i = B * a_i^(-1/alpha) / Σ_j a_j^(1-1/alpha), maksimum di a_min
        denom = sum(a ** (1.0 - 1.0 / alpha) for a in a_vals)
        CF_max = B_total * (a_min ** (-1.0 / alpha)) / max(denom, 1e-12)

    U_max = CF_max / bmin_global   # karena g_i <= 1 / bmin_global
    return max(U_max, 1e-12)


# ====================  (B) Helpers ====================

def key_bounds(p: Profiler, k: Knob):
    df = p.df[
        (p.df.stream == k.stream) &
        (p.df.model == k.model) &
        (p.df.QP == k.QP) &
        (p.df.FPS == k.FPS) &
        (p.df.Resolution == k.Resolution)
    ]
    return float(df.Bitrate.min()), float(df.Bitrate.max())


def stream_global_bounds(p: Profiler, keys_for_stream):
    mins, maxs = [], []
    for k in keys_for_stream:
        mn, mx = key_bounds(p, k)
        mins.append(mn)
        maxs.append(mx)
    return (min(mins), max(maxs)) if mins else (0.0, 0.0)


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


# ---------- stabilisasi finite-diff & Pareto pruning ----------

def adaptive_eps(p: Profiler, k: Knob, b: float, base_eps: float):
    mn, mx = key_bounds(p, k)
    left = max(mn, b - base_eps)
    right = min(mx, b + base_eps)
    eff_half = min(b - left, right - b)
    if eff_half <= 0:
        eff_half = max(1e-6, min(right - b, b - left, base_eps))
    return eff_half


def pareto_filter(p: Profiler, keys):
    stats = []
    for k in keys:
        mn, mx = key_bounds(p, k)
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


# ====================  (C) Marginal gain wrt bitrate ====================

def marginal_gain_welfare(p: Profiler, k: Knob, b: float, eps: float, alpha: float,
                          mode_u: str, ai_map: dict | None, gamma: float | None,
                          B_total: float | None = None, n_streams: int | None = None,
                          denom_sum: float | None = None):
    de = adaptive_eps(p, k, b, eps)
    mn, mx = key_bounds(p, k)
    b1 = max(mn, b - de)
    b2 = min(mx, b + de)

    if mode_u == "powerlaw":
        u1 = u_from_powerlaw(b1, ai_map[k.stream], gamma)
        u2 = u_from_powerlaw(b2, ai_map[k.stream], gamma)

    elif mode_u == "gradfair":
        a1, _, _ = p.query(k, b1)
        a2, _, _ = p.query(k, b2)
        g1 = grad_accuracy(a1, b1)
        g2 = grad_accuracy(a2, b2)
        ai = ai_map.get(k.stream, 1.0)
        u1_raw = u_alpha_closedform(g1, alpha, B_total, n_streams, ai, denom_sum)
        u2_raw = u_alpha_closedform(g2, alpha, B_total, n_streams, ai, denom_sum)
        # normalisasi ke [0,1]
        u1 = clip01(u1_raw / GRADFAIR_UMAX)
        u2 = clip01(u2_raw / GRADFAIR_UMAX)

    else:
        a1, _, _ = p.query(k, b1)
        a2, _, _ = p.query(k, b2)
        u1 = u_from_acc(a1)
        u2 = u_from_acc(a2)

    W1 = welfare_component(u1, alpha)
    W2 = welfare_component(u2, alpha)
    return (W2 - W1) / max(1e-12, (b2 - b1))


# ====================  (D) Lift ke acc_min ====================

def lift_to_accmin(p: Profiler, choice_keys, alloc, bmax, B, step, G, sla, acc_min):
    curB = sum(alloc.values())
    changed = True
    while changed:
        changed = False
        for k in choice_keys:
            tgt = None if acc_min is None else acc_min.get(k.stream)
            if tgt is None:
                continue
            acc, _, _ = p.query(k, alloc[k])
            if acc >= tgt:
                continue
            while acc < tgt and alloc[k] + step <= bmax[k] and curB + step <= B:
                nb = alloc[k] + step
                if sla is not None and violates_latency(p, k, nb, sla.get(k.stream)):
                    break
                tmp = dict(alloc)
                tmp[k] = nb
                if total_gpu(p, tmp) > G:
                    break
                alloc[k] = nb
                curB += step
                acc, _, _ = p.query(k, alloc[k])
                changed = True
        if curB + step > B:
            break
    if acc_min:
        for k in choice_keys:
            tgt = acc_min.get(k.stream)
            if tgt is not None and p.query(k, alloc[k])[0] < tgt:
                return None
    return alloc


# ====================  (E) Water-filling ====================

def water_filling(p: Profiler, choice_keys, B: float, step: float, eps: float,
                  alpha: float, omega, G: float, sla, acc_min, lift_first: bool,
                  mode_u: str, ai_map: dict | None, gamma: float | None):
    alloc, bmin, bmax = {}, {}, {}
    for k in choice_keys:
        mn, mx = key_bounds(p, k)
        bmin[k], bmax[k] = mn, mx
        alloc[k] = mn
        if violates_latency(p, k, alloc[k], None if sla is None else sla.get(k.stream)):
            return None, -np.inf

    curB = sum(alloc.values())
    if curB > B:
        return None, -np.inf
    if total_gpu(p, alloc) > G:
        return None, -np.inf

    # konstanta gradfair
    n_streams = len(choice_keys)
    denom_sum = None
    if mode_u == "gradfair" and not np.isclose(alpha, 1.0):
        denom_sum = sum((ai_map.get(k.stream, 1.0)) ** (1.0 - 1.0 / alpha) for k in choice_keys)

    if lift_first and acc_min:
        lifted = lift_to_accmin(p, choice_keys, alloc, bmax, B, step, G, sla, acc_min)
        if lifted is None:
            return None, -np.inf
        alloc = lifted
        curB = sum(alloc.values())

    while curB + step <= B:
        best_k, best_gain = None, -1e18
        for k in choice_keys:
            if alloc[k] + step > bmax[k]:
                continue
            nb = alloc[k] + step
            if violates_latency(p, k, nb, sla.get(k.stream) if sla else None):
                continue
            tmp = dict(alloc)
            tmp[k] = nb
            if total_gpu(p, tmp) > G:
                continue
            w = 1.0 if (omega is None) else omega.get(k.stream, 1.0)
            gain = w * marginal_gain_welfare(
                p, k, alloc[k], eps, alpha, mode_u, ai_map, gamma,
                B_total=B, n_streams=n_streams, denom_sum=denom_sum
            )
            if gain > best_gain:
                best_k, best_gain = k, gain
        if best_k is None or best_gain <= 0.0:
            break
        alloc[best_k] += step
        curB += step

    return alloc, objective_value(p, alloc, alpha, omega, sla, acc_min, mode_u, ai_map, gamma)


# ====================  (F) Objective = W_alpha(u) ====================

def objective_value(p: Profiler, alloc: dict, alpha: float, omega, sla, acc_min,
                    mode_u: str, ai_map: dict | None, gamma: float | None):
    W = 0.0
    streams = list(alloc.keys())
    n_streams = len(streams)
    denom_sum = None
    if mode_u == "gradfair" and not np.isclose(alpha, 1.0):
        denom_sum = sum((ai_map.get(k.stream, 1.0)) ** (1.0 - 1.0 / alpha) for k in streams)

    B_used = sum(alloc.values())

    for k, b in alloc.items():
        acc, lat, _ = p.query(k, b)
        if violates_latency(p, k, b, None if sla is None else sla.get(k.stream)):
            return -np.inf
        if acc_min and violates_accmin(p, k, b, acc_min):
            return -np.inf

        if mode_u == "powerlaw":
            u_i = u_from_powerlaw(b, ai_map[k.stream], gamma)

        elif mode_u == "gradfair":
            ai = ai_map.get(k.stream, 1.0)
            g_i = grad_accuracy(acc, b)
            u_raw = u_alpha_closedform(g_i, alpha, B_used, n_streams, ai, denom_sum)
            u_i = clip01(u_raw / GRADFAIR_UMAX)

        else:
            u_i = u_from_acc(acc)

        w = 1.0 if (omega is None) else omega.get(k.stream, 1.0)
        W += w * welfare_component(u_i, alpha)

    return W


# ====================  (G) Fill to budget ====================

def fill_to_budget(p: Profiler, alloc: dict, choice_keys, B: float, step: float,
                   G: float, sla, acc_min):
    curB = sum(alloc.values())
    if curB >= B:
        return alloc
    changed = True
    while changed and curB + step <= B:
        changed = False
        for k in choice_keys:
            nb = alloc[k] + step
            mn, mx = key_bounds(p, k)
            if nb > mx:
                continue
            if violates_latency(p, k, nb, None if sla is None else sla.get(k.stream)):
                continue
            tmp = dict(alloc)
            tmp[k] = nb
            if total_gpu(p, tmp) > G:
                continue
            alloc[k] = nb
            curB += step
            changed = True
            if curB + step > B:
                break
    return alloc


# ====================  (H) a_i & incentives (opsional) ====================

def parse_ai_linear(s: str, ai_default: float):
    out = {}
    if s:
        for tok in s.split(","):
            if not tok:
                continue
            sid, val = tok.split(":")
            out[int(sid)] = float(val)
    return out, float(ai_default)


def estimate_ai_from_profiler(p: Profiler, k: Knob, gamma: float, how: str = "mid"):
    mn, mx = key_bounds(p, k)
    if how == "min":
        b_ref = mn
    elif how == "max":
        b_ref = mx
    else:
        b_ref = np.sqrt(mn * mx) if (mn > 0 and mx > 0) else 0.5 * (mn + mx)
    acc_ref, _, _ = p.query(k, b_ref)
    acc_ref = max(acc_ref, 1e-9)
    return float((b_ref ** gamma) / acc_ref)


def estimate_ai_local(p: Profiler, k: Knob, b_star: float, eps_b: float = 5.0):
    mn, mx = key_bounds(p, k)
    b1 = max(mn, b_star - eps_b)
    b2 = min(mx, b_star + eps_b)
    a1, _, _ = p.query(k, b1)
    a2, _, _ = p.query(k, b2)
    du = max(1e-12, (a2 - a1))
    db = max(1e-9, (b2 - b1))
    return float(db / du)


def compute_incentives(a_map: dict, alpha: float):
    if alpha <= 0:
        return None, None
    sids = sorted(a_map.keys())
    if np.isclose(alpha, 1.0):
        return ({i: 1.0 for i in sids}, {i: 0.0 for i in sids})
    pw = {i: (a_map[i] ** (1.0 - 1.0 / alpha)) for i in sids}
    denom = sum(pw.values()) if pw else 0.0
    incentives, xsub = {}, {}
    for i in sids:
        share = 0.0 if denom == 0.0 else pw[i] / denom
        incentives[i] = (1.0 / alpha) + (1.0 - 1.0 / alpha) * share
        xsub[i] = (1.0 - 1.0 / alpha) * share
    return incentives, xsub


# ====================  (I) Budget resolver ====================

def resolve_bandwidth_budget(p: Profiler, cand: dict, mode: str,
                             B_abs: float, B_percent: float, B_margin: float):
    sum_min, sum_max = 0.0, 0.0
    for _, keys in cand.items():
        mn, mx = stream_global_bounds(p, keys)
        sum_min += mn
        sum_max += mx
    if mode == "abs":
        return float(B_abs)
    if mode == "sum_min":
        return float(sum_min)
    if mode == "sum_max":
        return float(sum_max)
    if mode == "percent_max":
        return float((B_percent / 100.0) * sum_max)
    if mode == "add_to_min":
        return float(sum_min + B_margin)
    return float(B_abs)


# ====================  (J) Main ====================

def main():
    global GRADFAIR_UMAX

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)

    # bandwidth
    ap.add_argument("--B_mode", choices=["abs", "sum_min", "sum_max", "percent_max", "add_to_min"],
                    default="abs")
    ap.add_argument("--B", type=float, default=0.0)
    ap.add_argument("--B_percent", type=float, default=100.0)
    ap.add_argument("--B_margin", type=float, default=0.0)

    ap.add_argument("--G", type=float, default=1e9)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--eps", type=float, default=5.0)
    ap.add_argument("--sla", type=str, default="")
    ap.add_argument("--omega", type=str, default="")
    ap.add_argument("--accmin", type=str, default="")
    ap.add_argument("--accmin_all", type=float, default=None)
    ap.add_argument("--allow_res", nargs="+", default=None)

    # pilihan utility individu
    ap.add_argument("--u_mode", choices=["acc", "powerlaw", "gradfair"], default="acc",
                    help="acc: u_i=Accuracy; powerlaw: u_i=x_i^gamma/a_i; "
                         "gradfair: u_i=normalized((Acc_i/b_i)*CF(alpha,a_i))")
    ap.add_argument("--gamma", type=float, default=None,
                    help="gamma untuk u_mode=powerlaw (0<gamma<1 disarankan)")

    # budget coefficient a_i (berlaku untuk powerlaw & gradfair)
    ap.add_argument("--ai_map", type=str, default="", help="Format '1:5,2:6' dst.")
    ap.add_argument("--ai_default", type=float, default=1.0,
                    help="Nilai default a_i bila tidak dispesifikkan.")

    # incentives (opsional)
    ap.add_argument("--incentive_mode", choices=["none", "linear", "powerlaw", "local"],
                    default="none")
    ap.add_argument("--powerlaw_calib", choices=["mid", "min", "max"], default="mid")

    # output
    ap.add_argument("--out", type=str, default="alloc_out.csv")
    ap.add_argument("--out_delim", type=str, default=",")
    ap.add_argument("--percent_text", action="store_true")
    ap.add_argument("--no_lift_first", action="store_true")

    args = ap.parse_args()
    p = Profiler(args.csv)

    # parse SLA & weights
    sla = None
    if args.sla:
        sla = {}
        for tok in args.sla.split(","):
            if not tok:
                continue
            sid, val = tok.split(":")
            sla[int(sid)] = float(val)

    omega = None
    if args.omega:
        omega = {}
        for tok in args.omega.split(","):
            if not tok:
                continue
            sid, val = tok.split(":")
            omega[int(sid)] = float(val)

    acc_min = {}
    if args.accmin_all is not None:
        acc_min["_global"] = float(args.accmin_all)
    if args.accmin:
        for tok in args.accmin.split(","):
            if not tok:
                continue
            sid, val = tok.split(":")
            acc_min[int(sid)] = float(val)

    # kandidat per stream (+ filter resolusi)
    allowed_res = set(args.allow_res) if args.allow_res else None
    cand = {}
    for (s, m, qp, fps, res), _ in p.df.groupby(
            ["stream", "model", "QP", "FPS", "Resolution"]):
        if allowed_res and str(res) not in allowed_res:
            continue
        cand.setdefault(int(s), []).append(
            Knob(int(s), str(m), int(qp), int(fps), str(res))
        )

    # terapkan global acc_min ke semua stream jika ada
    if "_global" in acc_min:
        g = acc_min.pop("_global")
        for sid in cand.keys():
            acc_min.setdefault(int(sid), float(g))

    if not cand:
        print("No candidates after resolution filtering. Check --allow_res.")
        return

    # Pareto prune
    for sid in list(cand.keys()):
        cand[sid] = pareto_filter(p, cand[sid])

    # budget
    B_budget = resolve_bandwidth_budget(p, cand, args.B_mode,
                                        args.B, args.B_percent, args.B_margin)
    if B_budget <= 0:
        print("Derived bandwidth B <= 0. Please check --B_mode and parameters.")
        return

    # --- siapkan a_i untuk powerlaw/gradfair (atau incentives) ---
    ai_map_raw, ai_default = parse_ai_linear(args.ai_map, args.ai_default)
    ai_map = dict(ai_map_raw)
    for sid in cand.keys():
        ai_map.setdefault(int(sid), ai_default)

    # validasi powerlaw (opsional estimasi a_i)
    if args.u_mode == "powerlaw":
        if not (args.gamma and 0.0 < args.gamma < 1.0):
            print("u_mode=powerlaw but --gamma invalid; please set 0<gamma<1.")
            return
        if args.ai_map == "":
            for sid in cand.keys():
                k0 = cand[sid][0]
                ai_map[sid] = estimate_ai_from_profiler(p, k0, args.gamma, how="mid")

    # --- GradFair: hitung konstanta normalisasi U_max ---
    if args.u_mode == "gradfair":
        bmin_global = float("inf")
        for keys in cand.values():
            for k in keys:
                mn, mx = key_bounds(p, k)
                bmin_global = min(bmin_global, mn)
        if not np.isfinite(bmin_global):
            bmin_global = 1.0
        GRADFAIR_UMAX = gradfair_umax(args.alpha, B_budget, ai_map, bmin_global)

    # ====================  Optimasi kombinasi knob ====================

    best = None
    streams_sorted = sorted(cand.keys())
    for choice in itertools.product(*(cand[s] for s in streams_sorted)):
        alloc, W = water_filling(
            p, choice, B_budget, step=args.step, eps=args.eps,
            alpha=args.alpha, omega=omega, G=args.G, sla=sla,
            acc_min=acc_min if acc_min else None,
            lift_first=(not args.no_lift_first),
            mode_u=args.u_mode, ai_map=ai_map, gamma=args.gamma
        )
        if alloc is None:
            continue
        alloc = fill_to_budget(p, alloc, choice, B_budget,
                               args.step, args.G, sla,
                               acc_min if acc_min else None)
        W = objective_value(p, alloc, args.alpha, omega, sla, acc_min,
                            mode_u=args.u_mode, ai_map=ai_map, gamma=args.gamma)
        if best is None or W > best[1]:
            best = (alloc, W, choice)

    if best is None:
        print("No feasible allocation under given constraints.")
        return

    # ====================  Output ====================

    alloc_best, W_best, _ = best
    total_gpu_used = sum(p.query(k, b)[2] for k, b in alloc_best.items())
    total_bw_used = sum(alloc_best.values())
    print(f"Best welfare W_alpha (alpha={args.alpha}): {W_best}")
    print(f"Total BW used: {total_bw_used:.1f} / {B_budget:.1f} kb/s | "
          f"Total GPU: {total_gpu_used:.2f} (budget {args.G})")
    print(f"[B_mode={args.B_mode}]  [u_mode={args.u_mode}]")

    rows = []
    W_parts = {}

    B_total_used = total_bw_used
    nS = len(alloc_best)
    if args.u_mode == "gradfair" and not np.isclose(args.alpha, 1.0):
        denom_sum_out = sum(
            (ai_map[k.stream]) ** (1.0 - 1.0 / args.alpha)
            for k in alloc_best.keys()
        )
    else:
        denom_sum_out = None

    for k, v in alloc_best.items():
        acc, lat, gpu = p.query(k, v)

        if args.u_mode == "powerlaw":
            u_i = u_from_powerlaw(v, ai_map[k.stream], args.gamma)
        elif args.u_mode == "gradfair":
            ai = ai_map.get(k.stream, 1.0)
            g_i = grad_accuracy(acc, v)
            u_raw = u_alpha_closedform(
                g_i, args.alpha, B_total_used, nS, ai, denom_sum_out
            )
            u_i = clip01(u_raw / GRADFAIR_UMAX)
        else:
            u_i = u_from_acc(acc)

        w = 1.0 if (omega is None) else omega.get(k.stream, 1.0)
        W_i = w * welfare_component(u_i, args.alpha)
        W_parts[k.stream] = W_parts.get(k.stream, 0.0) + W_i

        rows.append({
            "stream": k.stream,
            "model": k.model,
            "QP": k.QP,
            "FPS": k.FPS,
            "Resolution": k.Resolution,
            "Bitrate": round(v, 1),
            "Accuracy": round(acc, 6),
            "GradientAccuracy": round(grad_accuracy(acc, v), 9),
            "Latency": round(lat, 2),
            "GPU_load": round(gpu, 2),
            "a_i": ai_map.get(k.stream, 1.0),
            "u_i": float(u_i),
            "W_i": float(W_i),
        })

    W_total = sum(W_parts.values()) if W_parts else 0.0
    if abs(W_total) < 1e-12:
        W_total = 1.0
    BW_total = sum(r["Bitrate"] for r in rows) or 1.0

    def pct(x): return f"{x:.2f}%"

    sum_w_share = 0.0
    sum_bw_share = 0.0
    for r in rows:
        w_share = 100.0 * (W_parts[r["stream"]] / W_total) if W_total != 0 else 0.0
        bw_share = 100.0 * (r["Bitrate"] / BW_total)
        w_share = max(0.0, min(100.0, w_share))
        bw_share = max(0.0, min(100.0, bw_share))
        r["W_share"] = round(w_share, 4)
        r["BW_share"] = round(bw_share, 4)
        if args.percent_text:
            r["W_share_text"] = pct(w_share)
            r["BW_share_text"] = pct(bw_share)
        sum_w_share += w_share
        sum_bw_share += bw_share

    print(f"Check: sum W_share ≈ {sum_w_share:.2f}% | "
          f"sum BW_share ≈ {sum_bw_share:.2f}%")

    for r in sorted(rows, key=lambda x: x["stream"]):
        w_txt = r.get("W_share_text", f"{r['W_share']:.2f}%")
        b_txt = r.get("BW_share_text", f"{r['BW_share']:.2f}%")
        print(
            f"stream {r['stream']} -> {r['model']}, QP{r['QP']}, "
            f"{r['FPS']}@{r['Resolution']}: {r['Bitrate']:.1f} kb/s | "
            f"Acc={r['Accuracy']:.3f}, Lat={r['Latency']:.2f} ms, "
            f"GPU={r['GPU_load']:.2f}% | a_i={r['a_i']:.4g} | "
            f"u_i={r['u_i']:.4f}, W_i={r['W_i']:.4f}, "
            f"Wshare={w_txt}, BWshare={b_txt}"
        )

    pd.DataFrame(rows).sort_values(["stream"]).to_csv(
        args.out, index=False, sep=args.out_delim
    )
    print(f"Saved: {args.out} (sep='{args.out_delim}')")
    print(f"Derived B (kb/s) = {B_budget:.1f}  | "
          f"BW_total_out = {BW_total:.1f}")

    # ----- Incentives & Sharing (opsional; memakai a_i yang sama) -----
    if args.incentive_mode != "none":
        chosen_keys = {k.stream: k for k in alloc_best.keys()}
        a_map = {}
        if args.incentive_mode == "linear":
            a_map.update(ai_map)
        elif args.incentive_mode == "powerlaw":
            if not (args.gamma and 0.0 < args.gamma < 1.0):
                print("powerlaw incentive mode: invalid --gamma; skip incentives.")
            else:
                for sid, k in chosen_keys.items():
                    a_map[sid] = estimate_ai_from_profiler(
                        p, k, args.gamma, args.powerlaw_calib
                    )
        elif args.incentive_mode == "local":
            for k, b in alloc_best.items():
                a_map[k.stream] = estimate_ai_local(p, k, b, eps_b=args.eps)

        if a_map:
            incentives, xsub = compute_incentives(a_map, args.alpha)
            if incentives is not None:
                print("\n[Incentives & Sharing]")
                for sid in sorted(a_map.keys()):
                    print(
                        f"stream {sid}: a_i={a_map[sid]:.6g} | "
                        f"incentive={incentives[sid]:.6f} | "
                        f"cross-subsidy-rate={xsub[sid]:.6f}"
                    )
        else:
            print("No a_i computed for incentives (check incentive options).")


if __name__ == "__main__":
    main()
