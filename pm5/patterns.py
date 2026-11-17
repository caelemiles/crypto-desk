"""Market explanations and pattern mining.

Every 5 minutes, when a market resolves, the AI writes a short measured account of what the price did and
what was going on around it (order flow, the other coin, futures, volatility, the crowd's odds).

Every hour it mines all recorded live markets for patterns: for each signal (momentum, order flow, Binance
leading Chainlink, the other coin, liquidations, ...) at several points in the window, it asks two questions:
  1. Does the signal point to the right direction? ("explains the move")
  2. Does it predict the result BETTER THAN THE MARKET PRICE ALREADY DID? ("tradable": the only kind that can
     make money, because the price you pay already contains everything the crowd knows)
Only patterns that pass question 2 with a significance test (corrected for how many patterns were tried)
are turned into strategies, and those still have to win in shadow trading before they handle the account.
"""
import time


import config as C
import features as F
from learning import boot_ci
from util import money

SIGNALS = [
    ("ret_bn_60", "1-minute momentum"),
    ("ret_bn_180", "3-minute momentum"),
    ("ret_bn_10", "10-second momentum"),
    ("flow_30", "Binance taker buying (30s)"),
    ("flow_120", "Binance taker buying (2 min)"),
    ("lead_5", "Binance moving ahead of Chainlink"),
    ("eth_ret_60", "other coin's last minute"),
    ("eth_ret_300", "other coin's last 5 minutes"),
    ("book_imb", "Binance order-book imbalance"),
    ("pm_imb", "Polymarket order-book imbalance"),
    ("fut_flow", "futures taker flow"),
    ("liq_imb", "liquidations (shorts vs longs)"),
    ("prev_outcome", "previous market's result"),
    ("basis", "Chainlink vs Coinbase gap"),
]
MINE_T = (30, 90, 150, 210, 270)
LABEL = dict(SIGNALS)


def _bp(x):
    return x * 1e4


def explain(brain, start, outcome, open_ref, close_twap, path, extra):
    """Measured account of one market. path: [(ts, chainlink)], extra: dict of window statistics."""
    E = start + C.WINDOW
    name = C.A["name"]
    facts, tags = [], []
    if open_ref and path:
        pts = [(ts - start, p) for ts, p in path if start <= ts <= E]
        if len(pts) > 30:
            rel = [(t, _bp(p / open_ref - 1)) for t, p in pts]
            hi = max(r for _, r in rel)
            lo = min(r for _, r in rel)
            signs = [1 if r >= 0 else -1 for _, r in rel]
            crosses = [rel[i][0] for i in range(1, len(signs)) if signs[i] != signs[i - 1]]
            last_cross = crosses[-1] if crosses else None
            at = {T: next((r for t, r in rel if t >= T), None) for T in (60, 150, 240)}
            move = _bp(close_twap / open_ref - 1) if close_twap else rel[-1][1]
            facts.append(f"{name} finished {outcome} (settlement average {move:+.1f} bp vs the price to beat). "
                         f"Range during the window: {lo:+.1f} to {hi:+.1f} bp.")
            if not crosses:
                facts.append("It never crossed the price to beat after the open: one-way market.")
                tags.append("one_way")
            else:
                facts.append(f"It crossed the price to beat {len(crosses)} time(s); the last crossing was at "
                             f"{last_cross // 60}:{last_cross % 60:02d}.")
                if last_cross >= 210:
                    tags.append("late_reversal")
                if len(crosses) >= 4:
                    tags.append("choppy")
            side = {T: (None if r is None else ("UP" if r >= 0 else "DOWN")) for T, r in at.items()}
            held = [T for T, s in side.items() if s == outcome]
            if side.get(60) and side[60] != outcome:
                facts.append("The early leader (at 1:00) lost: the move reversed.")
                tags.append("early_leader_lost")
            elif side.get(60) == outcome and len(held) == 3:
                facts.append("The side leading at 1:00 stayed in front the whole way.")
                tags.append("early_leader_held")
    x = extra or {}
    if x.get("flow") is not None:
        f = x["flow"]
        agree = (f > 0) == (outcome == "UP")
        facts.append(f"Binance taker buying over the window: {50 + f * 100:.0f}% of volume "
                     f"({'in line with' if agree else 'against'} the result).")
        tags.append("flow_agreed" if agree else "flow_disagreed")
    if x.get("other_bp") is not None:
        o = x["other_bp"]
        agree = (o > 0) == (outcome == "UP")
        facts.append(f"{C.A['other']} moved {o:+.1f} bp in the same 5 minutes ({'same' if agree else 'opposite'} direction).")
        tags.append("other_coin_agreed" if agree else "other_coin_disagreed")
    if x.get("vol_ratio") is not None:
        v = x["vol_ratio"]
        if v > 1.6:
            facts.append(f"Volatility was {v:.1f}x the previous 15 minutes: a fast market.")
            tags.append("vol_spike")
        elif v < 0.6:
            facts.append(f"Volatility was only {v:.1f}x the previous 15 minutes: a quiet market.")
            tags.append("quiet")
    if x.get("liq_usd"):
        facts.append(f"Futures liquidations near the close: {money(x['liq_usd']).lstrip('+')} "
                     f"({'mostly shorts squeezed' if x.get('liq_short_share', 0.5) > 0.6 else 'mostly longs flushed' if x.get('liq_short_share', 0.5) < 0.4 else 'mixed'}).")
    w = brain.window(start)
    mk = next((m.id for m in brain.models.values() if m.kind == "market"), None)
    crowd = (w.get("preds", {}).get(mk, {}) or {}).get("150") if mk else None
    if crowd is not None:
        right = (crowd >= 0.5) == (outcome == "UP")
        facts.append(f"The crowd priced UP at {crowd:.0%} at 2:30 and was {'right' if right else 'wrong'}.")
        tags.append("crowd_right" if right else "crowd_wrong")
    entry = {"market": start, "title": f"{C.A['name']} {time.strftime('%H:%M', time.gmtime(start + 8 * 3600))}"
             f"-{time.strftime('%H:%M', time.gmtime(E + 8 * 3600))} SGT", "outcome": outcome, "facts": facts,
             "tags": tags, "evidence": "measured"}
    brain.j.market_entry(entry)
    return entry


def mine(brain, db, now):
    """Search recorded live markets for signals that beat the market price. Returns the results table."""
    rows = db.labelled_checkpoints(F.SCHEMA, since=now - 21 * 86400, live_only=True)
    data = [(start, t, f, y) for _, start, t, f, _, y in rows if t in MINE_T]
    markets = len({s for s, _, _, _ in data})
    out = []
    n_tests = len(SIGNALS) * len(MINE_T)
    alpha = 0.05 / n_tests
    for feat, label in SIGNALS:
        for T in MINE_T:
            pts = []
            for s, t, f, y in data:
                if t != T:
                    continue
                v, m = F.get(f, feat), F.get(f, "pm_mid_up")
                if v is None or m is None or not (0.05 < m < 0.95):
                    continue
                pts.append((v, m, y))
            if len(pts) < 60:
                continue
            mags = sorted(abs(v) for v, _, _ in pts)
            dz = mags[len(mags) // 4] if feat != "prev_outcome" else 0.5   # ignore the weakest quarter of readings
            sel = [(1 if v > 0 else -1, m, y) for v, m, y in pts if abs(v) > dz and v != 0]
            if len(sel) < 50:
                continue
            aligned = [sg * (y - m) for sg, m, y in sel]                       # beats the price?
            hit = sum(1 for sg, _, y in sel if (sg > 0) == (y == 1)) / len(sel)  # explains the direction?
            mkt_hit = sum(1 for _, m, y in sel if (m >= 0.5) == (y == 1)) / len(sel)
            lo, hi, mean = boot_ci(aligned, alpha=alpha)
            # trade with the signal at the mid: win rate and rough P&L per $1 after a typical fee
            status = ("tradable" if lo is not None and lo > 0.02 else
                      "beats the price (too small after fees)" if lo is not None and lo > 0 else
                      "promising, not proven yet" if mean > 0.02 else
                      "already priced in" if hit > 0.55 else "no effect")
            out.append({"feature": feat, "label": label, "t": T, "n": len(sel), "dz": dz, "hit": hit, "mkt_hit": mkt_hit,
                        "edge": mean, "lo": lo, "hi": hi, "status": status})
    out.sort(key=lambda r: (r["lo"] if r["lo"] is not None else -1), reverse=True)
    brain.s["patterns"] = {"ts": now, "markets": markets, "tests": n_tests, "alpha": alpha, "rows": out[:80]}
    _journal(brain, out, markets)
    return out


def _journal(brain, out, markets):
    j = brain.j
    explains = [r for r in out if r["hit"] >= 0.58 and r["n"] >= 80]
    tradable = [r for r in out if r["status"] == "tradable"]
    if explains:
        top = sorted(explains, key=lambda r: r["hit"], reverse=True)[:3]
        j.discovery("What moves these markets", "Signals that point the right way most often (measured on "
                    f"{markets} markets): " + "; ".join(f"{r['label']} at {r['t']}s right {r['hit']:.0%} of the time "
                                                         f"(n={r['n']}, the market price itself was right {r['mkt_hit']:.0%})"
                                                         for r in top)
                    + ". Pointing the right way is not enough to profit: it also has to beat the price you pay.",
                    "measured", {"n": markets}, key="explains")
    if tradable:
        for r in tradable[:3]:
            j.discovery(f"Pattern found: {r['label']} at {r['t']}s",
                        f"When {r['label']} is strong at {r['t']}s, the result beats what the market price implied by "
                        f"{r['edge'] * 100:+.1f} points on average (conservative estimate {r['lo'] * 100:+.1f}, n={r['n']}, "
                        f"corrected for {len(SIGNALS) * len(MINE_T)} patterns tested). Being turned into a strategy for shadow testing.",
                        "measured", {"n": r["n"], "edge": r["edge"]}, key=f"pat_{r['feature']}_{r['t']}")
    elif out:
        best = out[0]
        j.discovery("No tradable pattern yet",
                    f"Tested {len(out)} signal/time combinations on {markets} markets. None beats the market price by enough "
                    f"to cover fees yet. Closest: {best['label']} at {best['t']}s "
                    f"({best['edge'] * 100:+.1f} points vs the price, n={best['n']}). The AI keeps collecting data.",
                    "measured", {"n": markets}, key="no_pattern")


def launch_patterns(brain, rows_fn):
    """Turn new tradable patterns into pattern models + strategies (each still has to pass replay and shadow)."""
    pats = brain.s.get("patterns", {}).get("rows", [])
    launched = brain.s.setdefault("patterns_launched", [])
    room = brain.cfg["max_challengers"] - sum(1 for c in brain.s["challengers"].values() if c["status"] == "shadow")
    made = 0
    for r in pats:
        if r["status"] != "tradable" or made >= max(room, 1) or made >= 2:
            continue
        key = f"{r['feature']}@{r['t']}"
        if key in launched:
            continue
        launched.append(key)
        delta = round(max(r["lo"], 0.5 * r["edge"]), 4)      # shrink: use the conservative effect size
        m = brain.add_model("pattern", {"feature": r["feature"], "dz": r["dz"], "delta": delta, "t": r["t"],
                                        "label": r["label"]},
                            note=f"Market price adjusted by '{r['label']}' (measured {r['edge'] * 100:+.1f} pts, n={r['n']})")
        rules = {"t_min": max(5, r["t"] - 20), "t_max": min(292, r["t"] + 25), "min_edge": 0.01, "min_prob": 0.5,
                 "min_price": 0.15, "max_price": 0.85}
        base = brain._base_strategy() or brain.champion
        brain._launch(base, m.id, rules, f"Pattern: when {r['label']} is strong at {r['t']}s the market under-reacts "
                                         f"(measured {r['edge'] * 100:+.1f} pts vs price, n={r['n']}).",
                      "pattern mining", rows_fn(7), f"pattern strategy: {r['label']} @ {r['t']}s")
        made += 1


def win_rate_note(target):
    """Plain-language break-even table for a win-rate target (after a typical taker fee)."""
    rows = []
    for price in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        fee = 0.07 * price * (1 - price)
        rows.append({"price": price, "breakeven": price + fee, "profit_at_target": target - (price + fee)})
    return rows

