"""Daily learning reports, measured discoveries, and the data files behind the dashboard."""
import collections
import math

import numpy as np

from learning import OFFICIAL_T, boot_ci, brier, calibration, vol_regime
from util import money, sgt_day


def _m(v):
    return ("+" if v >= 0 else "-") + f"${abs(v):,.2f}"


def _pnl_stats(trades):
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return {"trades": len(pnls), "wins": len(wins), "win_rate": (len(wins) / len(pnls)) if pnls else None,
            "pnl": sum(pnls), "avg_win": float(np.mean(wins)) if wins else None,
            "avg_loss": float(np.mean(losses)) if losses else None,
            "fees": sum(t.get("fee", 0) + (t.get("exit_fee") or 0) for t in trades),
            "staked": sum(t["cost"] for t in trades)}


def _official(brain, since=0, until=None):
    ps, ys, pm = [], [], []
    mk = next((m.id for m in brain.models.values() if m.kind == "market"), None)
    for k, w in brain.s["windows"].items():
        s = int(k)
        if s < since or (until and s >= until) or not w.get("outcome") or not w.get("official"):
            continue
        y = 1 if w["outcome"] == "UP" else 0
        ps.append(w["official"]["p_up"])
        ys.append(y)
        pm.append(w["preds"].get(mk, {}).get(str(OFFICIAL_T)) if mk else None)
    return ps, ys, pm


def daily_report(brain, day, day_start, day_end):
    acct = brain.account
    trades = [t for t in acct.closed if day_start <= t.get("closed", 0) < day_end]
    st = _pnl_stats(trades)
    ds = brain.s["daily_stats"].get(day, {})
    ps, ys, pm = _official(brain, day_start - 300, day_end - 300)
    b = brier(ps, ys)
    pairs = [(p, m, y) for p, m, y in zip(ps, pm, ys) if m is not None]
    bm = brier([m for _, m, _ in pairs], [y for _, _, y in pairs]) if pairs else None
    bc = brier([p for p, _, _ in pairs], [y for _, _, y in pairs]) if pairs else None
    obs, proposals = [], []
    # --- measured observations (only with enough data, always with n)
    by_reg = collections.defaultdict(list)
    for t in trades:
        by_reg[vol_regime((t.get("feat") or {}).get("vol_ratio"))].append(t)
    for reg, ts in by_reg.items():
        if len(ts) >= 8:
            s = _pnl_stats(ts)
            obs.append({"text": f"In {reg} conditions: {s['trades']} trades, win rate {s['win_rate']:.0%}, P&L {_m(s['pnl'])}.",
                        "evidence": "measured", "n": s["trades"]})
    early = [t for t in trades if t.get("t_entry", 0) < 150]
    late = [t for t in trades if t.get("t_entry", 0) >= 150]
    if len(early) >= 8 and len(late) >= 8:
        a, c = _pnl_stats(early), _pnl_stats(late)
        better = "first" if a["pnl"] / len(early) > c["pnl"] / len(late) else "second"
        obs.append({"text": f"Entries in the first half of the window: {_m(a['pnl'])} over {a['trades']}; second half: "
                            f"{_m(c['pnl'])} over {c['trades']}. The {better} half did better per trade today.",
                    "evidence": "measured", "n": len(trades)})
    if st["trades"] >= 5 and st["staked"] > 0:
        gross = st["pnl"] + st["fees"]
        obs.append({"text": f"Fees paid: ${st['fees']:.2f} ({st['fees'] / st['staked']:.1%} of money staked); P&L before fees "
                            f"would have been {_m(gross)}.", "evidence": "measured", "n": st["trades"]})
    if bm is not None and bc is not None and len(pairs) >= 50:
        diffs = [(m - y) ** 2 - (p - y) ** 2 for p, m, y in pairs]
        lo, hi, mean = boot_ci(diffs, alpha=0.05)
        verdict = ("more accurate than" if lo is not None and lo > 0 else
                   "less accurate than" if hi is not None and hi < 0 else "statistically level with")
        obs.append({"text": f"At the {OFFICIAL_T}s mark the champion model's Brier score was {bc:.4f} vs the market price's {bm:.4f} "
                            f"over {len(pairs)} markets: {verdict} the market.", "evidence": "measured", "n": len(pairs)})
    tags = collections.Counter(tag for e in brain.j.trades if day_start <= e["ts"] < day_end and e["pnl"] <= 0 for tag in e["tags"])
    for tag, n in tags.most_common(3):
        if n >= 3:
            proposals.append({"text": {"reversal": "Investigate requiring a larger edge or later entries to reduce reversal losses.",
                                       "vol_spike": "Investigate volatility-sensitive entry thresholds.",
                                       "late_entry": "Test an earlier cut-off for entries.",
                                       "early_entry": "Test waiting longer before entering.",
                                       "thin_edge": "Test a higher minimum edge so fees do not eat the profit.",
                                       "overconfident": "Test temperature-scaled (less confident) probability models.",
                                       "expensive_price": "Test a lower maximum entry price.",
                                       "cheap_price": "Test a higher minimum entry price.",
                                       "thin_book": "Test a minimum order-book depth requirement."}.get(tag, f"Investigate '{tag}' losses."),
                              "evidence": "hypothesis", "why": f"'{tag}' appeared on {n} losing trades today"})
    exps = [e for e in brain.j.experiments if day_start <= e["ts"] < day_end]
    for e in exps:
        if e.get("stage") == "testing":
            proposals.append({"text": f"{e['title']} (now in shadow testing)", "evidence": "hypothesis", "why": e.get("goal", "")})
    report = {
        "day": day, "markets_analysed": ds.get("valid", 0), "markets_seen": ds.get("markets", 0),
        "predictions_made": ds.get("calls", 0), "correct": ds.get("correct", 0),
        "incorrect": ds.get("calls", 0) - ds.get("correct", 0), "brier": b, "market_brier": bm,
        "simulated_trades": st["trades"], "win_rate": st["win_rate"], "pnl": st["pnl"], "fees": st["fees"],
        "experiments": {"started": sum(e.get("stage") == "testing" for e in exps),
                        "accepted": sum(e.get("stage") == "accepted" for e in exps),
                        "rejected": sum(str(e.get("stage", "")).startswith("rejected") for e in exps)},
        "observations": obs or [{"text": "Not enough trades today for a reliable breakdown (the AI waits for 8+ trades per group).",
                                 "evidence": "measured", "n": st["trades"]}],
        "proposals": proposals[:6],
        "champion": brain.s.get("champion"),
        "equity_end": acct.day.get(day, {}).get("equity"),
    }
    brain.j.daily_report(report)
    brain.j.activity(f"Daily learning report for {day} written: {st['trades']} trades, P&L {money(st['pnl'])}", kind="report")
    return report


def discoveries(brain, db):
    """Measured findings refreshed a few times a day. Each has a stable key so it updates rather than repeats."""
    j = brain.j
    # 1. reference tracking: does our Chainlink-based TWAP match the official close?
    errs = []
    for w in list(brain.s["windows"].values())[-600:]:
        if w.get("our_twap") and w.get("close"):
            errs.append((w["our_twap"] - w["close"]) / w["close"] * 1e4)
    if len(errs) >= 30:
        a = np.abs(errs)
        j.discovery("Settlement reference tracking",
                    f"Over {len(errs)} markets, the AI's own 60-second Chainlink TWAP differed from Polymarket's official close "
                    f"by a median {np.median(a):.2f} bp (90th percentile {np.quantile(a, 0.9):.2f} bp, mean signed "
                    f"{np.mean(errs):+.2f} bp). This tells the model how much to trust its own estimate in the final seconds.",
                    "measured", {"n": len(errs), "median_bp": float(np.median(a))}, key="ref_tracking")
    # 2. base rate
    outs = [w["outcome"] for w in brain.s["windows"].values() if w.get("outcome")]
    if len(outs) >= 100:
        up = outs.count("UP") / len(outs)
        se = math.sqrt(0.25 / len(outs))
        sig = abs(up - 0.5) > 2 * se
        j.discovery("UP/DOWN base rate", f"UP won {up:.1%} of the last {len(outs)} markets "
                    + ("- a statistically meaningful tilt." if sig else f"- consistent with 50/50 (±{2 * se:.1%})."),
                    "measured", {"n": len(outs), "up": up}, key="base_rate")
    # 3. model vs market
    ps, ys, pm = _official(brain)
    pairs = [(p, m, y) for p, m, y in zip(ps, pm, ys) if m is not None]
    if len(pairs) >= 100:
        diffs = [(m - y) ** 2 - (p - y) ** 2 for p, m, y in pairs]
        lo, hi, mean = boot_ci(diffs)
        if lo is not None and lo > 0:
            body, ev = (f"The champion model's probabilities beat the market price on Brier score over {len(pairs)} markets "
                        f"(improvement {mean:.4f}, lower bound {lo:.4f}).", "measured")
        elif hi is not None and hi < 0:
            body, ev = (f"The market price is more accurate than the champion model ({-mean:.4f} Brier gap over {len(pairs)} "
                        f"markets). Edges are hard to find; the AI should stay selective.", "measured")
        else:
            body, ev = (f"No significant accuracy difference between the champion model and the market price over {len(pairs)} "
                        f"markets.", "measured")
        j.discovery("Model accuracy vs the market", body, ev, {"n": len(pairs)}, key="model_vs_market")
    # 4. calibration warning
    cps = [(t["prob"], 1 if t["pnl"] > 0 else 0) for t in brain.account.closed if t.get("prob") and t.get("exit_reason") == "settlement"]
    if len(cps) >= 50:
        mean_p = float(np.mean([p for p, _ in cps]))
        hit = float(np.mean([y for _, y in cps]))
        se = math.sqrt(max(hit * (1 - hit), 0.05) / len(cps))
        if mean_p - hit > 2 * se:
            j.discovery("Trades are over-confident", f"Across {len(cps)} settled trades the model's average confidence was "
                        f"{mean_p:.0%} but it won {hit:.0%}. Calibration needs work.", "measured", {"n": len(cps)},
                        key="trade_calibration")
        else:
            j.discovery("Trade confidence is calibrated", f"Across {len(cps)} settled trades average confidence {mean_p:.0%}, "
                        f"actual win rate {hit:.0%} (within noise).", "measured", {"n": len(cps)}, key="trade_calibration")


def overview(brain, marks, now):
    acct = brain.account
    eq = acct.equity(marks)
    closed = acct.closed
    st = _pnl_stats(closed)
    ar = acct.archive
    if ar["trades"]:
        st["trades"] += ar["trades"]
        st["wins"] += ar["wins"]
        st["pnl"] += ar["pnl"]
        st["fees"] += ar["fees"]
        st["win_rate"] = st["wins"] / st["trades"]
    day = sgt_day(now)

    def window_pnl(days):
        cut = now - days * 86400
        return sum(t["pnl"] for t in closed if t.get("closed", 0) >= cut)

    ps, ys, _ = _official(brain)
    ps7, ys7, _ = _official(brain, now - 7 * 86400)
    unreal = acct.open_value(marks) - sum(p["cost"] for p in acct.positions)
    return {"start_balance": acct.start_balance, "cash": acct.cash, "equity": eq, "net_pnl": eq - acct.start_balance,
            "roi": (eq - acct.start_balance) / acct.start_balance, "realized": st["pnl"], "unrealized": unreal,
            "max_drawdown": acct.max_dd, "trades": st["trades"], "win_rate": st["win_rate"], "avg_win": st["avg_win"],
            "avg_loss": st["avg_loss"], "fees": st["fees"], "daily_pnl": acct.day.get(day, {}).get("pnl", 0.0),
            "weekly_pnl": window_pnl(7), "monthly_pnl": window_pnl(30), "brier": brier(ps, ys), "brier_7d": brier(ps7, ys7),
            "brier_n": len(ps), "open_positions": len(acct.positions), "champion": brain.s.get("champion"),
            "paused_until": acct.paused_until if acct.paused_until > now else None}


def lab(brain):
    out = []
    for sid, c in brain.s["challengers"].items():
        s = brain.strats.get(sid)
        if not s:
            continue
        led = brain.s["ledgers"].get(sid, {})
        m = brain.models.get(s.model_id)
        out.append({"id": sid, "status": c["status"], "origin": c.get("origin"), "born": c.get("born"),
                    "note": s.note, "hypothesis": c.get("hypothesis"), "rules": s.describe(), "rules_raw": s.rules,
                    "model": s.model_id, "model_desc": m.describe() if m else "?", "replay": c.get("replay"),
                    "test": c.get("test"), "retired_reason": c.get("retired_reason"), "exp": c.get("exp"),
                    "shadow": {"windows": led.get("n_windows", 0), "trades": led.get("trades", 0), "wins": led.get("wins", 0),
                               "pnl": sum(led.get("pnl", {}).values()), "recent": led.get("recent", [])[-20:]}})
    order = {"champion": 0, "backup": 1, "shadow": 2, "promoted": 3, "retired": 4}
    out.sort(key=lambda x: (order.get(x["status"], 9), -(x["born"] or 0)))
    return out[:80]


def performance(brain):
    acct = brain.account
    ps, ys, pm = _official(brain)
    tp = [(t["prob"], 1 if t["pnl"] > 0 else 0) for t in acct.closed if t.get("prob") and t.get("exit_reason") == "settlement"]
    days = []
    for d, ds in sorted(brain.s["daily_stats"].items())[-60:]:
        dt = acct.day.get(d, {})
        days.append({"day": d, "markets": ds.get("markets"), "calls": ds.get("calls"), "correct": ds.get("correct"),
                     "brier": ds["brier_sum"] / ds["brier_n"] if ds.get("brier_n") else None,
                     "market_brier": ds["mkt_brier_sum"] / ds["mkt_brier_n"] if ds.get("mkt_brier_n") else None,
                     "pnl": dt.get("pnl")})
    # rolling log loss per model over the last 300 resolved markets
    from learning import logloss
    wins = [w for w in brain.s["windows"].values() if w.get("outcome")][-300:]
    model_cmp = []
    for mid, m in brain.models.items():
        lls = []
        for w in wins:
            y = 1 if w["outcome"] == "UP" else 0
            pr = w["preds"].get(mid)
            if pr:
                lls.append(np.mean([logloss(p, y) for p in pr.values()]))
        if len(lls) >= 10:
            model_cmp.append({"model": mid, "desc": m.describe(), "markets": len(lls), "logloss": float(np.mean(lls))})
    model_cmp.sort(key=lambda r: r["logloss"])
    return {"calibration_official": calibration(ps, ys), "calibration_trades": calibration([p for p, _ in tp], [y for _, y in tp],
                                                                                         [0.5 + i * 0.05 for i in range(11)]),
            "days": days, "model_compare": model_cmp, "official_n": len(ps),
            "research": brain.s["last"].get("research_result")}


def history(brain):
    hist = []
    for h in brain.s["champion_history"]:
        s = brain.strats.get(h["sid"])
        trades = [t for t in brain.account.closed if t.get("strategy") == h["sid"]
                  and h["since"] <= t.get("closed", 0) < (h.get("until") or 1e12)]
        st = _pnl_stats(trades)
        hist.append(h | {"rules": s.describe() if s else "", "model_desc": brain.models[h["model"]].describe()
                         if h["model"] in brain.models else "", "live": st})
    models = [{"id": m.id, "kind": m.kind, "desc": m.describe(), "note": m.note, "created": m.created, "parent": m.parent,
               "trained_until": m.trained_until} for m in brain.models.values()]
    return {"champions": hist, "models": models}
