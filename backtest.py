#!/usr/bin/env python3
"""Crypto Desk backtester - replays the REAL engine (engine.py) on past Binance candles.

It starts a fresh 100 USDT paper bot, gives it 10 days of history to learn from (like the live bot's
first run), then walks forward through the test period one step at a time. At every step the engine
only sees candles that had fully closed by then (no peeking at the future), and pays the same fees
and slippage as live. The result is written to docs/backtest.json for the dashboard.

  python backtest.py --days 30                      # last 30 days, current settings
  python backtest.py --days 30 --set atrStop=2 rr=2 # what-if: try other settings
  python backtest.py --synthetic                    # offline test on made-up prices
"""
import argparse, json, math, os, time
import engine as E

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "docs", "backtest.json")
WARM = 960  # 10 days of 15m bars to learn from before the test starts


def parse_set(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        v = v.strip()
        out[k] = None if v.lower() in ("none", "null") else (v.lower() == "true" if v.lower() in ("true", "false") else float(v))
    return out


def load_candles(days, synthetic):
    n = int(WARM + days * 96)
    if synthetic:
        t0 = (int(time.time() * 1000) // E.BAR - n - 2) * E.BAR
        spec = {"BTC": (70000, .004, .0001), "ETH": (3000, .005, 0), "SOL": (140, .006, .0002), "DOGE": (.15, .007, -.0001),
                "PEPE": (.00001, .009, .0003), "WIF": (1.5, .01, -.0003)}
        return {c: [[b[0], b[1], b[2], b[3], b[4], b[5] * 5e7 / s[0]] for b in E.synth(n, i + 7, drift=s[2], vol=s[1], start=s[0], t0=t0)]
                for i, (c, s) in enumerate(spec.items())}
    import bot as B
    now = int(time.time() * 1000)
    end = now // E.BAR * E.BAR  # the bar that is still open is not used
    start = end - n * E.BAR
    data = {}
    for c in E.COINS:
        rows = [r for r in B.fetch(c, start, end - 1) if r[0] < end]
        if len(rows) > WARM:
            data[c] = rows
        else:
            print(f"skip {c}: only {len(rows)} bars")
    return data


def feed(data, upto_close, after):
    """Completed bars with after < open time and close <= upto_close, plus a flat 'open bar' at the last close."""
    new = {}
    for c, rows in data.items():
        got = [r for r in rows if after < r[0] and r[0] + E.BAR <= upto_close]
        last = [r for r in rows if r[0] + E.BAR <= upto_close]
        if not last: continue
        lc = last[-1][4]
        new[c] = got + [[upto_close, lc, lc, lc, lc, 0]]  # the still-open bar: price only, no future high/low
    return new


def stats(trades, curve, start_eq, days, btc):
    n = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    gw = sum(t["pnl"] for t in wins); gl = -sum(t["pnl"] for t in trades if t["pnl"] <= 0)
    peak, mdd = start_eq, 0.0
    for _, e in curve:
        peak = max(peak, e); mdd = max(mdd, (peak - e) / peak * 100)
    end_eq = curve[-1][1] if curve else start_eq
    daily = {}
    for t in trades:
        d = E.day_key(t["exitTime"]); daily[d] = daily.get(d, 0) + t["pnl"]
    def group(keyf):
        g = {}
        for t in trades:
            a = g.setdefault(keyf(t), {"n": 0, "wins": 0, "pnl": 0.0, "R": 0.0})
            a["n"] += 1; a["wins"] += t["pnl"] > 0; a["pnl"] += t["pnl"]; a["R"] += t.get("R", 0)
        return sorted([{"k": k, "n": v["n"], "win": round(v["wins"] / v["n"], 3), "pnl": round(v["pnl"], 2),
                        "avgR": round(v["R"] / v["n"], 2)} for k, v in g.items()], key=lambda x: -x["pnl"])
    return {"startEq": start_eq, "endEq": round(end_eq, 2), "returnPct": round((end_eq / start_eq - 1) * 100, 2),
            "trades": n, "winRate": round(len(wins) / n, 3) if n else None,
            "profitFactor": round(gw / gl, 2) if gl > 0 else (None if not n else 99),
            "avgR": round(sum(t.get("R", 0) for t in trades) / n, 3) if n else None,
            "avgWin": round(gw / len(wins), 2) if wins else None, "avgLoss": round(gl / (n - len(wins)), 2) if n > len(wins) else None,
            "fees": round(sum(t.get("fees", 0) for t in trades), 2), "maxDDPct": round(mdd, 2),
            "perDay": round((end_eq - start_eq) / max(days, 1e-9), 2), "tradesPerDay": round(n / max(days, 1e-9), 1),
            "bestDay": round(max(daily.values()), 2) if daily else None, "worstDay": round(min(daily.values()), 2) if daily else None,
            "greenDays": sum(1 for v in daily.values() if v > 0), "days": round(days, 1),
            "btcHoldPct": round((btc[1] / btc[0] - 1) * 100, 2) if btc and btc[0] else None,
            "byCoin": group(lambda t: t["coin"]), "bySide": group(lambda t: t["side"]),
            "byExit": group(lambda t: t["exitReason"].split(" (")[0]), "byStrategy": group(lambda t: t.get("leadLabel") or "?")}


def verdict(s):
    if not s["trades"]:
        return "No trades in the test period: the entry rules were too strict for these market conditions."
    good = s["profitFactor"] is not None and s["profitFactor"] >= 1.2 and s["returnPct"] > 0 and s["trades"] >= 30
    ok = s["returnPct"] > 0
    if good:
        return (f"Promising: +{s['returnPct']}% over {s['days']} days with profit factor {s['profitFactor']} on {s['trades']} trades. "
                "Keep paper trading to confirm it holds up live.")
    if ok:
        return (f"Slightly positive (+{s['returnPct']}%) but not convincing yet (profit factor {s['profitFactor']}, {s['trades']} trades). "
                "Could be luck: needs more trades or a longer test.")
    return (f"Losing: {s['returnPct']}% over {s['days']} days (profit factor {s['profitFactor']}, win rate "
            f"{round((s['winRate'] or 0) * 100)}%). These settings do not have an edge on this period.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=30)
    ap.add_argument("--step", type=int, default=2, help="15m bars between engine runs (live runs every bar)")
    ap.add_argument("--set", nargs="*", help="what-if settings, e.g. atrStop=2 rr=2 riskPct=0.01")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    overrides = parse_set(a.set)
    t_start = time.time()
    data = load_candles(a.days, a.synthetic)
    if not data or "BTC" not in data:
        raise SystemExit("no candle data")
    times = sorted({r[0] for r in data["BTC"]})
    t0 = times[WARM]  # first bar of the test period
    t_end = times[-1] + E.BAR
    state = {"overrides": overrides}
    fed_to = -1
    trades, curve, journal = [], [], []
    now = t0
    first = True
    while now <= t_end:
        close_t = now  # every bar opening before `now` has closed by `now`
        new = feed(data, close_t, fed_to)
        for c, rows in new.items():  # safety check: the engine never gets a bar that closes after `now`
            assert all(r[0] + E.BAR <= close_t for r in rows[:-1]), "lookahead"
        rep = E.run(state, new, close_t + 60000)
        fed_to = close_t - E.BAR
        trades += rep["closed"]
        curve.append([close_t, rep["bot"]["equity"]])
        state = {"bot": rep["bot"], "model": rep["model"], "history": rep["history"], "overrides": overrides}
        if first:
            first = False
            if not a.synthetic:
                print(f"warm-up done: {rep['model']['samples']} shadow ideas learned")
        now += a.step * E.BAR
    bot = state["bot"]; model = state["model"]
    # close whatever is still open at the last price so the result is complete
    last_px = {c: rows[-1][4] for c, rows in data.items()}
    for pos in list(bot["positions"]):
        E.close_position(bot, model, pos, last_px[pos["coin"]], t_end, "End of test", trades)
    bot["positions"] = []
    end_eq = round(E.equity(bot, last_px), 2)
    curve.append([t_end, end_eq])
    days = (t_end - t0) / (24 * E.H)
    btc_rows = [r for r in data["BTC"] if r[0] >= t0]
    s = stats(trades, curve, bot["startBalance"], days, (btc_rows[0][1], btc_rows[-1][4]) if btc_rows else None)
    shadow = [r["R"] for r in model["resolved"]]
    out = {"generated": int(time.time() * 1000), "synthetic": a.synthetic, "engineVersion": model.get("engineVersion"),
           "from": t0, "to": t_end, "coins": sorted(data), "step": a.step, "overrides": overrides,
           "params": {k: model["params"].get(k) for k in ("riskPct", "lev", "minEvR", "atrStop", "rr", "timeStopH", "hardStopH",
                                                          "maxOpen", "lossStreakN", "dailyLossPct", "coinCooldownH", "minQuoteVol24")},
           "stats": s, "verdict": verdict(s),
           "shadow": {"n": len(shadow), "avgR": round(sum(shadow) / len(shadow), 3) if shadow else None,
                      "settings": (model.get("settings") or {}).get("table")},
           "curve": curve[:: max(1, len(curve) // 400)] + [curve[-1]],
           "trades": [{k: t.get(k) for k in ("id", "coin", "side", "entryTime", "exitTime", "entry", "exitPrice", "pnl", "R",
                                              "exitReason", "holdH", "leadLabel")} for t in trades[-200:]],
           "protections": [j for j in model.get("journal", []) if j["kind"] == "protect"][-30:],
           "seconds": round(time.time() - t_start, 1)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, allow_nan=False, separators=(",", ":"))
    print(json.dumps({k: v for k, v in s.items() if not k.startswith("by")}, indent=1))
    print(out["verdict"])
    print(f"shadow ideas: {out['shadow']}  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
