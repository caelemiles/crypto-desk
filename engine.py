#!/usr/bin/env python3
"""
Crypto Desk engine v5.1 (fixed 10 USDT stake, quick trades) - self-learning memecoin DAY TRADER (paper). Long AND short, USDT, 15m candles.
v5 (15 Nov 2026): fixes the over-confident win chance (raw model said 97-100% on trades that lost):
every entry now uses a CALIBRATED win chance and expected R measured from the bot's own resolved ideas
in the same confidence band, and only trades with a real expected edge (+0.10R or more after fees).
Trades must agree with the 1h trend unless the measured edge is strong. Exits let winners run: half
the position is banked at +1R, the stop goes to breakeven, the rest trails 1.5 ATR behind price and
aims for +4R; the 12h time stop only closes trades that are not working (a trade up 0.3R+ may run to 36h).
Positions use up to 2x paper leverage so the 3% risk per trade is real (v4's 50% notional cap cut it to
about 1%). Risk scales with the record: 4.5% after a 20-trade profit factor of 1.5+, 2% when it is under 0.8.
v4: classic day-trader setups as features and strategies (VWAP pullback/rejection, opening-range breakout,
previous-day high/low, market structure HH/HL, liquidity sweeps, relative strength vs BTC, volume thrust),
a stop/target "settings lab" that tests 5 stop/target combinations on every idea and moves the live
setting to the one that pays best, and a 500 USDT goal gate before any live trading is even discussed.
v3: 15m bars, 11 coins (majors + memecoins), resting breakout orders between runs, fee-aware learning,
time-of-day features, pattern stats, 100 USDT start with profit milestones, aggressive 3% risk per trade.

HOW IT LEARNS
  1. Every completed 1h candle on every coin becomes two "shadow" trade ideas (one long, one short):
     entry at that close, stop atrStop*ATR away, target rr times the risk. When later candles resolve
     an idea (target first = win, stop first = loss, 48h time-out = win only if >= +1R) the side's
     online logistic model is updated with the features seen at that moment. That is ~380 labelled
     samples a day, so it learns from far more than its own trades.
  2. Each side has strategy weights (trend / breakout / dip-or-rip / momentum) updated by a Hedge rule
     from the R-multiple of every resolved idea: strategies that keep working get more say.
  3. The entry bar (minProb) is re-tuned each run to the level with the best expectancy over the last
     400 resolved ideas, moving at most 0.01 per run (range 0.38-0.60).
  4. Per-coin bias per side learns which coins and directions the model reads well.
  Short features are the long features mirrored (price falling looks like price rising), so the same
  strategy code scores both sides while each side keeps its own learned weights.
RISK
  1% of equity at risk per trade (0.5x-1.5x by confidence), max 25% of equity per position, 1x (no
  leverage), max 4 open, fees 0.1%/side + 0.05% slippage, breakeven after +1R, ATR trail after +1.5R,
  48h time stop, drawdown brake (half risk past 8% DD, no entries past 15%).

COMMANDS
  selftest
  state  --dir DOCS --out state.json      build engine state from downloaded db docs (see runbook)
  run    --state state.json --candles new.json --now MS --out report.json
  export --report report.json --dir OUT --now MS   write one JSON file per db doc + OUT/manifest.json
"""
import json, math, sys, argparse, random, os, datetime as dt

H = 3600000
BAR = 900000  # 15 minutes
COINS = ["BTC", "ETH", "SOL", "DOGE", "PEPE", "SHIB", "WIF", "BONK", "FLOKI", "PENGU", "TRUMP"]
MEMES = {"DOGE", "PEPE", "SHIB", "WIF", "BONK", "FLOKI", "PENGU", "TRUMP"}
SIDES = ["long", "short"]
BASE_V3 = ["mom1", "mom4", "mom24", "mom96", "emaFast", "emaTrend", "rsi", "bbz", "volz", "donch", "btcTrend", "volRegime",
           "sessAsia", "sessEU", "sessUS"]
BASE_V4 = ["vwap", "orb", "prevDay", "structure", "relBTC", "volThrust", "sweep"]
BASE = BASE_V3 + BASE_V4
DIRECTIONAL = {"mom1", "mom4", "mom24", "mom96", "emaFast", "emaTrend", "rsi", "bbz", "donch", "btcTrend",
               "vwap", "orb", "prevDay", "structure", "relBTC", "volThrust", "sweep"}
STRATS_V3 = ["trend", "breakout", "reversal", "momentum"]
STRATS_V4 = ["vwap", "orb", "sweep", "relstr"]
STRATS = STRATS_V3 + STRATS_V4
# order matters: old vectors stay valid, new features are appended
FEATS = BASE_V3 + ["s_" + s for s in STRATS_V3] + BASE_V4 + ["s_" + s for s in STRATS_V4]
STRAT_LABEL = {"long": {"trend": "Trend following", "breakout": "Breakout", "reversal": "Buy the dip", "momentum": "Momentum",
                        "vwap": "VWAP pullback", "orb": "Opening-range breakout", "sweep": "Liquidity sweep reversal",
                        "relstr": "Strong vs BTC"},
               "short": {"trend": "Downtrend following", "breakout": "Breakdown", "reversal": "Sell the rip", "momentum": "Downside momentum",
                         "vwap": "VWAP rejection", "orb": "Opening-range breakdown", "sweep": "Sweep of highs (fade)",
                         "relstr": "Weak vs BTC"}}
CONFIGS = [[1.0, 1.5], [1.5, 1.5], [1.5, 2.0], [2.0, 1.5], [2.0, 2.0]]  # [stop in ATRs, reward:risk]; quick-trade range (v5.1)
FEAT_LABEL = {"mom1": "15m move", "mom4": "1h move", "mom24": "6h move", "mom96": "24h move", "emaFast": "Fast EMA cross",
              "emaTrend": "1h trend", "rsi": "RSI", "bbz": "Bollinger position", "volz": "Volume surge",
              "donch": "12h range position", "btcTrend": "BTC trend", "volRegime": "Volatility",
              "sessAsia": "Asia hours", "sessEU": "Europe hours", "sessUS": "US hours",
              "s_trend": "Trend signal", "s_breakout": "Breakout signal", "s_reversal": "Reversal signal",
              "s_momentum": "Momentum signal",
              "vwap": "Distance from VWAP", "orb": "Opening range position", "prevDay": "Previous-day range position",
              "structure": "Higher highs / higher lows", "relBTC": "Strength vs BTC", "volThrust": "Volume thrust candle",
              "sweep": "Liquidity sweep", "s_vwap": "VWAP setup", "s_orb": "Opening-range setup",
              "s_sweep": "Sweep setup", "s_relstr": "Relative-strength setup"}
KEEP_BARS = 960   # 10 days of 15m bars
MIN_BARS = 64     # 16 hours


def side_model():
    w = {f: 0.0 for f in FEATS}
    w.update({"emaTrend": 0.15, "btcTrend": 0.10, "s_trend": 0.10, "s_momentum": 0.05})  # weak priors
    w.update({"structure": 0.05, "s_vwap": 0.05})
    return {"w": w, "bias": round(logit(0.36), 4), "coinBias": {c: 0.0 for c in COINS},
            "strat": {s: {"w": 1.0, "n": 0, "wins": 0, "sumR": 0.0} for s in STRATS},
            "samples": 0, "wins": 0}


def default_model():
    return {
        "engineVersion": 4, "configs": {}, "configLog": [], "sides": {s: side_model() for s in SIDES}, "lr": 0.03, "l2": 0.0005,
        "params": {"riskPct": 0.03, "maxNotionalPct": 0.5, "maxOpen": 4, "atrStop": 2.5, "rr": 2.0,
                   "minProb": 0.45, "minEdge": -0.6, "timeStopH": 12, "feePct": 0.001, "slipPct": 0.001,
                   "beAfterR": 1.0, "trailAfterR": 1.5, "allowShort": True, "maxSameSide": 3,
                   "orderHours": 4, "orderSlack": 0.05, "minNotional": 5.0},
        "samples": 0, "wins": 0, "pending": [], "resolved": [], "lastBar": {},
        "thresholdHist": [], "lessons": [], "learning": [], "created": None}


def default_bot(start_balance=100.0, now=0):
    return {"startBalance": start_balance, "cash": start_balance, "equity": start_balance,
            "peak": start_balance, "maxDD": 0.0, "realized": 0.0, "fees": 0.0, "positions": [],
            "curve": [], "daily": {}, "startedAt": now, "lastRunAt": None, "runs": 0, "tradeSeq": 0,
            "closedN": 0, "wins": 0, "losses": 0, "grossWin": 0.0, "grossLoss": 0.0,
            "bySide": {s: {"n": 0, "wins": 0, "pnl": 0.0} for s in SIDES}, "byCoin": {},
            "btcStart": None, "status": "paper", "currency": "USDT", "orders": [], "orderSeq": 0,
            "goals": [200, 500, 1000, 5000, 10000], "goalHits": {}}


# ---------------------------------------------------------------- maths
def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(z):
    z = max(-40, min(40, z))
    return 1 / (1 + math.exp(-z))


def clip(x, a=-3.0, b=3.0):
    if x != x: return 0.0
    return max(a, min(b, x))


def ema(vals, n):
    k = 2 / (n + 1); e = vals[0]; out = []
    for v in vals:
        e = v * k + e * (1 - k); out.append(e)
    return out


def atr(bars, n=14):
    trs = [max(bars[i][2] - bars[i][3], abs(bars[i][2] - bars[i - 1][4]), abs(bars[i][3] - bars[i - 1][4]))
           for i in range(1, len(bars))]
    if not trs: return 0.0
    a = sum(trs[:n]) / min(n, len(trs))
    for tr in trs[n:]:
        a = (a * (n - 1) + tr) / n
    return a


def rsi(closes, n=14):
    if len(closes) < n + 1: return 50.0
    g = l = 0.0
    for i in range(1, n + 1):
        d = closes[i] - closes[i - 1]; g += max(d, 0); l += max(-d, 0)
    ag, al = g / n, l / n
    for i in range(n + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        ag = (ag * (n - 1) + max(d, 0)) / n; al = (al * (n - 1) + max(-d, 0)) / n
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


# ---------------------------------------------------------------- features
def session_flags(t):
    hr = ((t // H) + 8) % 24  # Singapore hour
    return (1.0 if 8 <= hr < 16 else 0.0, 1.0 if 16 <= hr < 24 else 0.0, 1.0 if hr < 8 else 0.0)


def day_trader_features(b, price, a, apct, btc_ret24):
    """Classic intraday levels. Crypto's trading day is anchored at 00:00 UTC (08:00 SGT)."""
    day0 = b[-1][0] // (24 * H) * (24 * H)
    today = [x for x in b if x[0] >= day0]
    yday = [x for x in b if day0 - 24 * H <= x[0] < day0]
    # anchored VWAP (today; if the day is under 1h old, include yesterday)
    seg = today if len(today) >= 4 else (yday + today)
    pv = sum((x[2] + x[3] + x[4]) / 3 * x[5] for x in seg); vv = sum(x[5] for x in seg)
    vw = pv / vv if vv > 0 else price
    f = {"vwap": clip((price - vw) / (2 * a))}
    # opening range = first hour of the UTC day
    if len(today) >= 5:
        orh = max(x[2] for x in today[:4]); orl = min(x[3] for x in today[:4])
        f["orb"] = clip((price - (orh + orl) / 2) / max(orh - orl, a))
    else:
        f["orb"] = 0.0
    if len(yday) >= 48:
        pdh = max(x[2] for x in yday); pdl = min(x[3] for x in yday)
        f["prevDay"] = clip(((price - pdl) / ((pdh - pdl) or a)) * 2 - 1)
    else:
        f["prevDay"] = 0.0
    # market structure from three 8-bar blocks
    st = 0.0
    if len(b) >= 24:
        b1, b2, b3 = b[-24:-16], b[-16:-8], b[-8:]
        h1, h2, h3 = (max(x[2] for x in q) for q in (b1, b2, b3))
        l1, l2, l3 = (min(x[3] for x in q) for q in (b1, b2, b3))
        st = 0.5 * ((h3 > h2) - (h3 < h2) + (l3 > l2) - (l3 < l2)) + 0.25 * ((h2 > h1) - (h2 < h1) + (l2 > l1) - (l2 < l1))
    f["structure"] = st
    c = [x[4] for x in b]
    r24 = math.log(c[-1] / c[-25]) if len(c) > 24 else 0.0
    f["relBTC"] = clip((r24 - btc_ret24) / (apct * math.sqrt(24)) / 1.5) if btc_ret24 is not None else 0.0
    last = b[-1]; prior = [x[5] for x in b[-21:-1]]
    rng = (last[2] - last[3]) or a
    ratio = last[5] / (sum(prior) / len(prior)) if prior and sum(prior) > 0 else 1.0
    f["volThrust"] = clip(math.log(max(ratio, 1e-6)) * (last[4] - last[1]) / rng * 1.5)
    sw = 0.0
    if len(b) >= 52:
        ref = b[-50:-2]; hi = max(x[2] for x in ref); lo = min(x[3] for x in ref)
        for x in b[-2:]:
            if x[3] < lo and x[4] > lo: sw = 1.0      # swept the lows, closed back inside: bullish
            if x[2] > hi and x[4] < hi: sw = -1.0     # swept the highs, closed back inside: bearish
    f["sweep"] = sw
    return f


def base_features(bars, btc=None):
    """bars: 15m bars oldest first (>= MIN_BARS). btc: {"emaTrend","ret24"} for alts. Returns (x, atr, price), LONG view."""
    if btc is not None and not isinstance(btc, dict): btc = {"emaTrend": btc, "ret24": None}
    b = bars[-200:]
    c = [x[4] for x in b]
    price = c[-1]
    a = atr(b, 14) or price * 0.01
    apct = a / price
    def mom(k):
        return 0.0 if len(c) <= k else clip(math.log(c[-1] / c[-1 - k]) / (apct * math.sqrt(k)) / 1.5)
    e12, e26 = ema(c, 12)[-1], ema(c, 26)[-1]
    c4 = [c[i] for i in range(len(c) - 1, -1, -4)][::-1]
    e4 = ema(c4, 20)[-1] if len(c4) >= 5 else price
    sma20 = sum(c[-20:]) / 20
    sd20 = math.sqrt(sum((v - sma20) ** 2 for v in c[-20:]) / 20) or price * 0.001
    vols = [x[5] for x in b]
    v4 = sum(vols[-4:]) / 4
    prev = vols[-52:-4] if len(vols) >= 52 else vols[:-4]
    v48 = sum(prev) / max(1, len(prev))
    hi48 = max(x[2] for x in b[-48:]); lo48 = min(x[3] for x in b[-48:])
    apcts = [atr(b[i - 15:i + 1], 14) / b[i][4] for i in range(max(15, len(b) - 96), len(b))]
    med = sorted(apcts)[len(apcts) // 2] if apcts else apct
    sa, se, su = session_flags(b[-1][0])
    x = {"mom1": mom(1), "mom4": mom(4), "mom24": mom(24), "mom96": mom(96),
         "sessAsia": sa, "sessEU": se, "sessUS": su,
         "emaFast": clip((e12 - e26) / a), "emaTrend": clip((price - e4) / (2 * a)),
         "rsi": clip((rsi(c[-60:]) - 50) / 20), "bbz": clip((price - sma20) / sd20 / 2),
         "volz": clip(math.log((v4 + 1e-9) / (v48 + 1e-9))),
         "donch": clip(((price - lo48) / ((hi48 - lo48) or 1)) * 2 - 1, -1, 1),
         "volRegime": clip(math.log(apct / med) if med > 0 else 0)}
    x["btcTrend"] = x["emaTrend"] if btc is None else btc["emaTrend"]
    my24 = math.log(c[-1] / c[-25]) if len(c) > 24 else 0.0
    x.update(day_trader_features(b, price, a, apct, my24 if btc is None else btc.get("ret24")))
    x["_ret24"] = my24
    return x, a, price


def side_view(x, side):
    """Mirror directional features for the short side and add strategy scores."""
    v = {k: (-x[k] if (side == "short" and k in DIRECTIONAL) else x[k]) for k in BASE}
    s = strategy_scores(v)
    for k in STRATS:
        v["s_" + k] = s[k]
    return v, s


def strategy_scores(x):
    trend = math.tanh(0.8 * x["emaTrend"] + 0.4 * x["emaFast"]) if x["btcTrend"] > -1.5 else -0.5
    breakout = 0.0
    if x["donch"] > 0.8:
        breakout = math.tanh(7.5 * (x["donch"] - 0.8) * max(0.2, x["volz"] + 0.5))
    elif x["donch"] < -0.8:
        breakout = -0.5
    reversal = 0.0
    if x["rsi"] < -0.75 and x["bbz"] < -0.5 and x["emaTrend"] > -1.5:
        reversal = math.tanh(-x["bbz"] - (x["rsi"] + 0.75))
    elif x["rsi"] > 1.0:
        reversal = -0.4
    momentum = math.tanh(0.6 * x["mom4"] + 0.4 * x["mom24"])
    vw = 0.0  # VWAP pullback: above VWAP, close to it, structure not broken
    if 0 <= x.get("vwap", 0) < 0.8 and x.get("structure", 0) >= 0:
        vw = math.tanh(0.6 + 0.5 * x.get("structure", 0) - abs(x["vwap"] - 0.3))
    elif x.get("vwap", 0) < -0.6:
        vw = -0.4
    orb = 0.0  # opening-range breakout with volume
    if x.get("orb", 0) > 0.5:
        orb = math.tanh(x["orb"] * max(0.3, x["volz"] + 0.6))
    elif x.get("orb", 0) < -0.5:
        orb = -0.4
    sweep = 0.0  # liquidity sweep reversal (stop hunt then reclaim)
    if x.get("sweep", 0) > 0:
        sweep = 0.7 + 0.3 * (x.get("volThrust", 0) > 0)
    elif x.get("sweep", 0) < 0:
        sweep = -0.5
    relstr = math.tanh(0.7 * x.get("relBTC", 0)) if x["btcTrend"] > -0.5 else 0.0
    return {"trend": round(trend, 4), "breakout": round(breakout, 4), "reversal": round(reversal, 4),
            "momentum": round(momentum, 4), "vwap": round(vw, 4), "orb": round(orb, 4), "sweep": round(sweep, 4),
            "relstr": round(relstr, 4)}


def predict(model, side, coin, v):
    m = model["sides"][side]
    z = m["bias"] + m["coinBias"].get(coin, 0.0)
    for f in FEATS:
        z += m["w"].get(f, 0.0) * v.get(f, 0.0)
    return sigmoid(z)


def ensemble(model, side, s):
    st = model["sides"][side]["strat"]
    tw = sum(st[k]["w"] for k in STRATS)
    return sum(st[k]["w"] * s[k] for k in STRATS) / tw


def lead_strategy(model, side, s):
    st = model["sides"][side]["strat"]
    best, bv = None, 0.0
    for k in STRATS:
        val = st[k]["w"] * s[k]
        if val > bv: best, bv = k, val
    return best


def sgd(model, side, coin, v, y):
    m = model["sides"][side]
    p = predict(model, side, coin, v)
    g = p - y; lr = model["lr"]
    m["bias"] = clip(m["bias"] - lr * g, -4, 2)
    m["coinBias"][coin] = clip(m["coinBias"].get(coin, 0.0) - lr * 0.5 * g, -1, 1)
    for f in FEATS:
        w = m["w"].get(f, 0.0)
        m["w"][f] = round(clip(w - lr * (g * v.get(f, 0.0) + model["l2"] * w), -3, 3), 5)
    m["bias"] = round(m["bias"], 5)
    return p


def hedge(model, side, s, R, eta=0.004):
    st = model["sides"][side]["strat"]
    for k in STRATS:
        if abs(s.get(k, 0)) < 0.15: continue
        st[k]["w"] *= math.exp(eta * clip(R, -2, 3) * s[k])
        if s[k] > 0:
            st[k]["n"] += 1; st[k]["sumR"] = round(st[k]["sumR"] + R, 3)
            if R > 0: st[k]["wins"] += 1
    m = sum(st[k]["w"] for k in STRATS) / len(STRATS)
    for k in STRATS:
        st[k]["w"] = round(min(3.0, max(0.2, st[k]["w"] / m)), 4)


# ---------------------------------------------------------------- data hygiene
def merge_history(hist, new, now):
    warns, live = [], {}
    for coin, rows in (new or {}).items():
        rows = sorted([[int(r[0])] + [float(v) for v in r[1:6]] for r in rows if len(r) >= 6], key=lambda r: r[0])
        good = [r for r in rows if r[3] > 0 and r[2] >= max(r[1], r[4]) * 0.9999 and r[3] <= min(r[1], r[4]) * 1.0001]
        if len(good) < len(rows):
            warns.append(f"{coin}: {len(rows) - len(good)} malformed candles dropped")
        rows = good
        if not rows:
            warns.append(f"{coin}: no usable candles"); continue
        if now - rows[-1][0] > 3 * H:
            warns.append(f"{coin}: data stale (last candle {round((now - rows[-1][0]) / H, 1)}h old) - not traded this run")
            continue
        live[coin] = rows[-1][4]
        old = {r[0]: r for r in hist.get(coin, [])}
        for r in rows:
            if r[0] + BAR <= now:
                old[r[0]] = r
        merged = [old[t] for t in sorted(old)]
        for i in range(1, len(merged)):
            if abs(merged[i][4] / merged[i - 1][4] - 1) > 0.35:
                warns.append(f"{coin}: suspicious >35% jump in one 15m bar at {merged[i][0]} - check the data"); break
        hist[coin] = merged[-KEEP_BARS:]
        gaps = sum(1 for i in range(1, len(hist[coin])) if hist[coin][i][0] - hist[coin][i - 1][0] != BAR)
        if gaps: warns.append(f"{coin}: {gaps} gap(s) in stored history (fetch the missing 15m bars)")
    return hist, live, warns


# ---------------------------------------------------------------- shadow learning
def resolve_idea(idea, bars_after, params):
    e, s, g = idea["e"], idea["s"], idea["g"]
    long = idea["d"] == "long"
    risk = abs(e - s)
    for b in bars_after:
        hit_stop = b[3] <= s if long else b[2] >= s
        hit_tgt = b[2] >= g if long else b[3] <= g
        if hit_stop: return 0, -1.0
        if hit_tgt: return 1, params["rr"]
        if b[0] - idea["t"] >= params["timeStopH"] * H:
            R = ((b[4] - e) if long else (e - b[4])) / risk
            return (1 if R >= 1 else 0), round(R, 3)
    return None


def cost_R(p, price, risk):
    """Round-trip fees + slippage measured in R (multiples of the risk per unit)."""
    return 2 * (p["feePct"] + p["slipPct"]) * price / risk if risk > 0 else 9.0


V5_PARAMS = {"maxNotionalPct": 1.5, "lev": 2.0, "timeStopH": 12, "hardStopH": 36, "trailAfterR": 1.0,
             "trailAtr": 1.5, "partialAtR": 1.0, "partialFrac": 0.5, "runnerR": 4.0, "minEvR": 0.10,
             "strongEvR": 0.30, "dailyTarget": 100.0}


def build_calib(model):
    """Per side: split the last resolved ideas into 8 bands by raw model probability and measure what
    really happened in each band (win rate and average net R). This is the honest win chance."""
    out = {}
    for side in SIDES:
        res = sorted([r for r in model["resolved"] if r["d"] == side], key=lambda r: r["p"])
        if len(res) < 150:
            out[side] = None; continue
        k = 8; bins = []
        for i in range(k):
            chunk = res[i * len(res) // k:(i + 1) * len(res) // k]
            if not chunk: continue
            bins.append({"lo": chunk[0]["p"], "hi": chunk[-1]["p"], "n": len(chunk),
                         "win": round(sum(r["y"] for r in chunk) / len(chunk), 3),
                         "avgR": round(sum(r["R"] for r in chunk) / len(chunk), 3)})
        out[side] = bins
    model["calib2"] = out
    return out


def cal(model, side, p_raw, k_cost=0.0):
    """Calibrated (win chance, expected net R) for a raw probability, shrunk toward caution."""
    bins = (model.get("calib2") or {}).get(side)
    rr = model["params"]["rr"]
    if not bins:
        pc = 0.36 + max(0.0, p_raw - 0.5) * 0.1  # cautious until 150 ideas per side are measured
        return round(pc, 4), round(pc * rr - (1 - pc) - k_cost - 0.1, 3)
    b = bins[-1]
    for x in bins:
        if p_raw <= x["hi"]:
            b = x; break
    n = b["n"]; prior = 20.0
    base = sum(x["win"] * x["n"] for x in bins) / max(1, sum(x["n"] for x in bins))
    pc = (b["win"] * n + base * prior) / (n + prior)
    ev = (b["avgR"] * n - 0.1 * prior) / (n + prior)
    return round(pc, 4), round(ev, 3)


def risk_now(bot, p):
    rec = bot.get("recent", [])[-20:]
    gw = sum(x for x in rec if x > 0); gl = -sum(x for x in rec if x < 0)
    pf = gw / gl if gl > 0 else (9 if gw > 0 else 0)
    base = p["riskPct"]
    if len(rec) >= 20 and pf >= 1.5: return base * 1.5, f"record strong (PF {pf:.2f} over 20 trades): risk {base*150:.0f}% of equity per trade"
    if len(rec) >= 10 and pf < 0.8: return base * 0.5, f"record weak (PF {pf:.2f} over {len(rec)} trades): risk {base*50:.0f}% of equity per trade"
    return base, f"risk {base*100:.0f}% of equity per trade"


# ---------------------------------------------------------------- learning journal (v6)
def jot(model, t, kind, text):
    j = model.setdefault("journal", [])
    j.append({"t": t, "kind": kind, "text": text})
    model["journal"] = j[-300:]


def cs_key(coin, side):
    return f"{coin}:{side}"


def is_blocked(bot, coin, side):
    """A coin+side is benched once its own live record shows it loses: 6+ trades, under 40% wins, below -1.5R in total."""
    r = bot.get("byCS", {}).get(cs_key(coin, side), [])[-12:]
    if len(r) < 6: return False
    return sum(1 for x in r if x > 0) / len(r) < 0.4 and sum(r) < -1.5


def vec(v):
    return [round(v.get(f, 0.0), 3) for f in FEATS]


def unvec(a):
    return {f: (a[i] if i < len(a) else 0.0) for i, f in enumerate(FEATS)}


def learn(model, hist, now):
    p = model["params"]
    rep = {"newIdeas": 0, "resolved": 0, "wins": 0, "bySide": {s: [0, 0] for s in SIDES}, "byCoin": {}}
    btc = hist.get("BTC", [])
    btc_idx = {b[0]: i for i, b in enumerate(btc)}
    btc_cache = {}
    for coin, bars in hist.items():
        last = model["lastBar"].get(coin, 0)
        for i in range(MIN_BARS, len(bars)):
            t = bars[i][0]
            if t <= last: continue
            bt = None
            if coin != "BTC" and btc_idx.get(t, -1) >= MIN_BARS:
                if t not in btc_cache:
                    bx = base_features(btc[:btc_idx[t] + 1])[0]
                    btc_cache[t] = {"emaTrend": bx["emaTrend"], "ret24": bx["_ret24"]}
                bt = btc_cache[t]
            x, a, price = base_features(bars[:i + 1], bt)
            for side in SIDES:
                v, _ = side_view(x, side)
                sgn = 1 if side == "long" else -1
                stop = price - sgn * p["atrStop"] * a
                k = round(cost_R(p, price, p["atrStop"] * a), 3)
                model["pending"].append({"c": coin, "d": side, "t": t, "e": price, "s": stop,
                                         "g": price + sgn * p["rr"] * p["atrStop"] * a, "k": k, "a": a,
                                         "v": vec(v), "p": round(predict(model, side, coin, v), 4),
                                         "q": 1 if predict(model, side, coin, v) >= p["minProb"] else 0})
                rep["newIdeas"] += 1
            model["lastBar"][coin] = t
    model["pending"].sort(key=lambda d: d["t"])
    still = []
    for idea in model["pending"]:
        after = [b for b in hist.get(idea["c"], []) if b[0] > idea["t"]]
        r = resolve_idea(idea, after, p)
        lab = config_outcomes(idea, after, p) if "a" in idea else {}
        if r is None or lab is None:
            if now - idea["t"] < (p["timeStopH"] + 24) * H: still.append(idea)
            continue
        record_configs(model, lab, idea.get("q", 0))
        y, R = r
        side, v = idea["d"], unvec(idea["v"])
        pred = sgd(model, side, idea["c"], v, y)
        s = {k: v["s_" + k] for k in STRATS}
        hedge(model, side, s, R)
        sm = model["sides"][side]; sm["samples"] += 1; sm["wins"] += y
        model["samples"] += 1; model["wins"] += y
        model["resolved"].append({"t": idea["t"], "c": idea["c"], "d": side, "p": round(pred, 4), "y": y,
                                  "R": round(R - idea.get("k", 0), 3), "lead": lead_strategy(model, side, s)})
        rep["resolved"] += 1; rep["wins"] += y; rep["bySide"][side][0] += 1; rep["bySide"][side][1] += y
        bc = rep["byCoin"].setdefault(idea["c"], [0, 0]); bc[0] += 1; bc[1] += y
    model["pending"] = still[-6000:]
    model["resolved"] = model["resolved"][-1000:]
    rep["threshold"] = tune_threshold(model)
    rep["settings"] = tune_config(model, now)
    return rep


def ckey(cf):
    return f"{cf[0]:g}x{cf[1]:g}"


def config_outcomes(idea, after, p):
    """Net R of this idea under every stop/target setting in the lab; None until all are resolved."""
    out = {}
    long = idea["d"] == "long"; sgn = 1 if long else -1
    for cf in CONFIGS:
        risk = cf[0] * idea["a"]
        tmp = {"e": idea["e"], "s": idea["e"] - sgn * risk, "g": idea["e"] + sgn * cf[1] * risk, "d": idea["d"], "t": idea["t"]}
        r = resolve_idea(tmp, after, {**p, "rr": cf[1]})
        if r is None: return None
        out[ckey(cf)] = round(r[1] - cost_R(p, idea["e"], risk), 3)
    return out


def record_configs(model, lab, quality):
    cs = model.setdefault("configs", {})
    for k, R in lab.items():
        for tag in (["all", "taken"] if quality else ["all"]):
            st = cs.setdefault(k, {}).setdefault(tag, {"n": 0.0, "sumR": 0.0, "wins": 0.0})
            d = 0.998  # recency weighting: recent market conditions count more
            st["n"] = round(st["n"] * d + 1, 3); st["sumR"] = round(st["sumR"] * d + R, 3)
            st["wins"] = round(st["wins"] * d + (1 if R > 0 else 0), 3)


def tune_config(model, now=None):
    """Move the live stop/target to the setting that has paid best on recent ideas (after fees)."""
    p = model["params"]; cs = model.get("configs", {})
    cur = ckey([p["atrStop"], p["rr"]])
    def score(k):
        st = cs.get(k, {})
        t = st.get("taken"); a = st.get("all")
        use = t if t and t["n"] >= 60 else a
        if not use or use["n"] < 20: return None, 0
        return use["sumR"] / use["n"], use["n"]
    table = {ckey(cf): score(ckey(cf)) for cf in CONFIGS}
    ready = {k: v for k, v in table.items() if v[0] is not None and v[1] >= 200}
    if len(ready) < len(CONFIGS):
        return {"active": cur, "note": "still collecting: the live setting can switch once every setting has 200+ ideas",
                "table": {k: (round(v[0], 3) if v[0] is not None else None) for k, v in table.items()},
                "n": {k: round(v[1]) for k, v in table.items()}}
    best = max(ready, key=lambda k: ready[k][0])
    if best != cur and ready[best][0] > (ready.get(cur, (-9, 0))[0] or -9) + 0.05:
        cf = next(c for c in CONFIGS if ckey(c) == best)
        p["atrStop"], p["rr"] = cf[0], cf[1]
        model.setdefault("configLog", []).append({"t": now, "from": cur, "to": best, "avgR": round(ready[best][0], 3)})
        model["configLog"] = model["configLog"][-30:]
        cur = best
    return {"active": cur, "table": {k: (round(v[0], 3) if v[0] is not None else None) for k, v in table.items()},
            "n": {k: round(v[1]) for k, v in table.items()}}


def tune_threshold(model):
    p = model["params"]; res = model["resolved"][-600:]
    if len(res) < 120:
        return {"minProb": p["minProb"], "note": f"needs 120 resolved ideas to tune (has {len(res)})"}
    best, best_ev, table = p["minProb"], -9, []
    for th in [x / 100 for x in range(35, 71, 2)]:
        sel = [r for r in res if r["p"] >= th]
        if len(sel) < 25: continue
        ev = sum(r["R"] for r in sel) / len(sel)  # R already net of fees and slippage
        table.append([th, len(sel), round(ev, 3)])
        if ev > best_ev: best, best_ev = th, ev
    old = p["minProb"]
    target = best if best_ev > 0 else min(0.60, old + 0.01)
    step = 0.01 if target > old + 1e-9 else -0.01 if target < old - 1e-9 else 0
    p["minProb"] = round(min(0.60, max(0.38, old + step)), 2)
    return {"minProb": p["minProb"], "from": old, "bestTh": best, "bestEvR": round(best_ev, 3), "table": table}


# ---------------------------------------------------------------- paper trading
def pos_value(pos, price):
    move = (price - pos["entry"]) if pos["side"] == "long" else (pos["entry"] - price)
    return pos.get("margin", pos["notional"]) + move * pos["qty"]


def partial_close(bot, model, pos, price, t, frac):
    """Bank part of a winning trade (v5): realise the P&L on frac of the position, keep the rest running."""
    p = model["params"]; long = pos["side"] == "long"
    fill = price * (1 - p["slipPct"]) if long else price * (1 + p["slipPct"])
    q = pos["qty"] * frac
    fee = fill * q * p["feePct"]
    move = (fill - pos["entry"]) if long else (pos["entry"] - fill)
    efee = pos["entryFee"] * frac
    margin = pos.get("margin", pos["notional"]) * frac
    pnl = move * q - fee - efee
    bot["cash"] += margin + move * q - fee
    bot["fees"] += fee; bot["realized"] += pnl
    d = day_key(t); bot["daily"][d] = round(bot["daily"].get(d, 0) + pnl, 2)
    pos["qty"] -= q; pos["notional"] = round(pos["notional"] * (1 - frac), 2)
    pos["margin"] = pos.get("margin", pos["notional"] / (1 - frac)) - margin
    pos["entryFee"] = pos["entryFee"] - efee
    pos["partialPnl"] = round(pos.get("partialPnl", 0) + pnl, 4)
    pos.setdefault("events", []).append({"t": t, "e": f"Banked {int(frac*100)}% at {fmt_px(fill)} (+{pnl:.2f} USDT)"})


def close_position(bot, model, pos, price, t, reason, closed):
    p = model["params"]
    long = pos["side"] == "long"
    fill = price * (1 - p["slipPct"]) if long else price * (1 + p["slipPct"])
    fee = fill * pos["qty"] * p["feePct"]
    move = (fill - pos["entry"]) if long else (pos["entry"] - fill)
    pnl_rest = move * pos["qty"] - fee - pos["entryFee"]
    bot["cash"] += pos_value(pos, fill) - fee
    bot["fees"] += fee; bot["realized"] += pnl_rest; bot["closedN"] += 1
    d0 = day_key(t); bot["daily"][d0] = round(bot["daily"].get(d0, 0) + pnl_rest, 2)
    pnl = pnl_rest + pos.get("partialPnl", 0)
    bot["recent"] = (bot.get("recent", []) + [round(pnl, 4)])[-30:]
    R = (pnl / pos["riskCash"]) if pos.get("riskCash") else (move / pos["risk0"] if pos["risk0"] > 0 else 0)
    if pnl > 0: bot["wins"] += 1; bot["grossWin"] += pnl
    else: bot["losses"] += 1; bot["grossLoss"] += -pnl
    bs = bot["bySide"][pos["side"]]; bs["n"] += 1; bs["wins"] += pnl > 0; bs["pnl"] = round(bs["pnl"] + pnl, 2)
    bc = bot["byCoin"].setdefault(pos["coin"], {"n": 0, "wins": 0, "pnl": 0.0})
    bc["n"] += 1; bc["wins"] += pnl > 0; bc["pnl"] = round(bc["pnl"] + pnl, 2)
    hedge(model, pos["side"], pos.get("scores", {}), R, eta=0.03)
    k = cs_key(pos["coin"], pos["side"])
    cs = bot.setdefault("byCS", {})
    cs[k] = (cs.get(k, []) + [round(R, 2)])[-12:]
    tr = dict(pos)
    tr.update({"status": "closed", "exitTime": t, "exitPrice": round(fill, 8), "exitReason": reason,
               "pnl": round(pnl, 2), "pnlPct": round(pnl / max(0.01, pos.get("notional0", pos["notional"])) * 100, 2), "R": round(R, 2),
               "fees": round(fee + pos["entryFee"], 2), "holdH": round((t - pos["entryTime"]) / H, 1),
               "last": round(fill, 8), "uPnl": 0})
    tr["review"] = auto_review(tr)
    rec = bot["byCS"][k]; w = sum(1 for x in rec if x > 0)
    lead = pos.get("lead"); sw = model["sides"][pos["side"]]["strat"].get(lead, {}).get("w") if lead else None
    if pnl > 0:
        why = ("hit the target" if reason == "Take profit" else "locked profit with the trailing stop" if reason == "Trailing stop"
               else f"closed by {reason.lower()}")
        lesson = f"Won {pnl:.2f} USDT ({R:+.2f}R) on {pos['coin']} {pos['side']}: {why} after {tr['holdH']}h."
    else:
        why = ("the move went straight against it" if reason == "Stop loss" and tr["holdH"] < 1
               else "it never got going, so the time stop cut it" if reason.startswith("Time stop")
               else "the signal flipped, so it got out early" if reason.startswith("AI exit") else f"{reason.lower()}")
        lesson = f"Lost {abs(pnl):.2f} USDT ({R:+.2f}R) on {pos['coin']} {pos['side']}: {why}."
    lesson += f" {pos['coin']} {pos['side']}s are now {w}/{len(rec)} wins."
    if sw is not None: lesson += f" Trust in {pos.get('leadLabel', 'this setup')} is now {sw:.2f}."
    jot(model, t, "trade", lesson)
    closed.append(tr)


def manage_positions(bot, model, hist, live, now, scan_x):
    p = model["params"]
    closed, keep = [], []
    for pos in bot["positions"]:
        coin, long = pos["coin"], pos["side"] == "long"
        done = False
        for b in [b for b in hist.get(coin, []) if b[0] > pos.get("checkedTo", pos["entryBar"])]:
            t, o, h, l, c = b[0], b[1], b[2], b[3], b[4]
            pos["checkedTo"] = t
            if long:
                pos["best"] = max(pos.get("best", pos["entry"]), h)
                stop_hit, tgt_hit = l <= pos["stop"], h >= pos["target"]
            else:
                pos["best"] = min(pos.get("best", pos["entry"]), l)
                stop_hit, tgt_hit = h >= pos["stop"], l <= pos["target"]
            if stop_hit:
                px = min(o, pos["stop"]) if long else max(o, pos["stop"])
                beyond = (pos["stop"] - pos["entry"]) * (1 if long else -1)
                reason = "Stop loss" if beyond < 0 else ("Breakeven stop" if abs(beyond) / pos["entry"] < 0.004 else "Trailing stop")
                close_position(bot, model, pos, px, t + BAR, reason, closed); done = True; break
            if tgt_hit:
                px = max(o, pos["target"]) if long else min(o, pos["target"])
                close_position(bot, model, pos, px, t + BAR, "Take profit", closed); done = True; break
            fav = (pos["best"] - pos["entry"]) if long else (pos["entry"] - pos["best"])
            r_now = fav / pos["risk0"]
            sgn = 1 if long else -1
            if r_now >= p.get("partialAtR", 99) and not pos.get("partialPnl") and pos["qty"] > 0:
                lvl = pos["entry"] + sgn * p["partialAtR"] * pos["risk0"]
                partial_close(bot, model, pos, lvl, t + BAR, p.get("partialFrac", 0.5))
                pos["target"] = round(pos["entry"] + sgn * p.get("runnerR", 4.0) * pos["risk0"], 10)
            if r_now >= p["beAfterR"] and (pos["stop"] - pos["entry"]) * sgn < 0:
                pos["stop"] = round(pos["entry"] * (1 + sgn * 2 * p["feePct"]), 8)
                pos.setdefault("events", []).append({"t": t + BAR, "e": "Stop moved to breakeven"})
            if r_now >= p["trailAfterR"]:
                trail = pos["best"] - sgn * max(p.get("trailAtr", 1.5) * pos["atr"], 0.6 * pos["risk0"])
                if (trail - pos["stop"]) * sgn > 0:
                    if not any(e["e"].startswith("Trailing") for e in pos.get("events", [])):
                        pos.setdefault("events", []).append({"t": t + BAR, "e": "Trailing stop engaged"})
                    pos["stop"] = round(trail, 8)
            age = t + BAR - pos["entryTime"]
            cur_r = ((c - pos["entry"]) * sgn) / pos["risk0"] if pos["risk0"] > 0 else 0
            if age >= p.get("hardStopH", p["timeStopH"]) * H or (age >= p["timeStopH"] * H and cur_r < 0.3):
                lab = p.get("hardStopH", p["timeStopH"]) if age >= p.get("hardStopH", p["timeStopH"]) * H else p["timeStopH"]
                close_position(bot, model, pos, c, t + BAR, f"Time stop ({lab}h)", closed); done = True; break
        if done: continue
        sx = scan_x.get((coin, pos["side"]))
        if sx and coin in live and sx.get("evR", 0) < -0.3 and sx["ens"] < -0.25:
            close_position(bot, model, pos, live[coin], now, "AI exit: signal flipped against the trade", closed)
            continue
        if coin in live:
            px = live[coin] * (1 - p["slipPct"]) if long else live[coin] * (1 + p["slipPct"])
            pos["last"] = live[coin]
            pos["uPnl"] = round(((px - pos["entry"]) if long else (pos["entry"] - px)) * pos["qty"] - pos["entryFee"], 2)
            pos["uR"] = round(((px - pos["entry"]) if long else (pos["entry"] - px)) / pos["risk0"], 2)
        keep.append(pos)
    bot["positions"] = keep
    return closed


def equity(bot, live):
    return bot["cash"] + sum(pos_value(pos, live.get(pos["coin"], pos.get("last", pos["entry"]))) for pos in bot["positions"])


def day_key(t):
    return dt.datetime.fromtimestamp((t + 8 * H) / 1000, dt.timezone.utc).strftime("%Y-%m-%d")


def fmt_px(x):
    if x >= 100: return f"{x:,.2f}"
    if x >= 1: return f"{x:,.4f}"
    if x >= 0.01: return f"{x:.5f}"
    return f"{x:.4g}" if x >= 1e-4 else f"{x:.10f}".rstrip("0")


def auto_thesis(pos):
    side = "Long" if pos["side"] == "long" else "Short"
    pro = [d["label"] for d in pos["top"] if d["v"] > 0][:3]
    con = [d["label"] for d in pos["top"] if d["v"] < 0][:2]
    drv = ", ".join(pro) + (f" (against: {', '.join(con)})" if con else "")
    return (f"{side} {pos['coin']} at {fmt_px(pos['entry'])}: calibrated win chance {pos['p']:.0%}, measured edge "
            f"{pos.get('expR', 0):+.2f}R, led by {pos['leadLabel']}. What pushed it: {drv}. Stop {fmt_px(pos['stop0'])} "
            f"(-1R, {pos['riskCash']:.2f} USDT at risk){' via a breakout order' if pos.get('via') == 'order' else ''}, target {fmt_px(pos['target'])} (+{pos['rr']:.1f}R; half banked at +1R, the rest trails toward +4R).")


def auto_review(tr):
    res = "won" if tr["pnl"] > 0 else "lost"
    return (f"{tr['exitReason']} after {tr['holdH']}h: {res} {abs(tr['pnl']):.2f} USDT ({tr['R']:+.2f}R). "
            f"Entered on {tr['p']:.0%} win chance led by {tr.get('leadLabel')}.")


def make_position(bot, model, coin, side, price, qty, risk_ps, atr_v, entry_t, entry_bar, meta):
    p = model["params"]; sgn = 1 if side == "long" else -1
    notional = qty * price
    fee = notional * p["feePct"]
    margin = notional / p.get("lev", 1.0)
    bot["cash"] -= margin + fee; bot["fees"] += fee; bot["tradeSeq"] += 1
    pos = {"id": f"T{bot['tradeSeq']:04d}", "coin": coin, "side": side, "status": "open", "entryTime": entry_t,
           "entryBar": entry_bar, "entry": price, "qty": qty, "notional": round(notional, 2), "notional0": round(notional, 2), "margin": margin,
           "stop": price - sgn * risk_ps, "stop0": price - sgn * risk_ps, "target": price + sgn * p["rr"] * risk_ps,
           "risk0": risk_ps, "riskCash": round(qty * risk_ps, 2), "atr": atr_v, "rr": p["rr"],
           "entryFee": round(fee, 6), "best": price, "last": price, "uPnl": round(-fee, 2), "uR": 0, "events": []}
    pos.update(meta)
    pos["thesis"] = auto_thesis(pos)
    bot["positions"].append(pos)
    return pos


def process_orders(bot, model, hist, now):
    """Fill resting breakout orders from the 15m bars since they were placed (in time order)."""
    p = model["params"]
    events = []
    for o in bot.get("orders", []):
        sgn = 1 if o["side"] == "long" else -1
        fill = None
        for b in hist.get(o["coin"], []):
            if b[0] < o["placedBar"] or b[0] >= o["expires"]: continue
            if (o["side"] == "long" and b[2] >= o["trigger"]) or (o["side"] == "short" and b[3] <= o["trigger"]):
                px = max(o["trigger"], b[1]) if o["side"] == "long" else min(o["trigger"], b[1])
                fill = (b[0], px * (1 + sgn * p["slipPct"]), b); break
        events.append((fill[0] if fill else 10 ** 15, o, fill))
    events.sort(key=lambda e: e[0])
    filled, cancelled = [], []
    for _, o, fill in events:
        if not fill:
            cancelled.append(o["id"]); continue
        t, px, b = fill
        open_n = len(bot["positions"])
        same = sum(1 for q in bot["positions"] if q["side"] == o["side"])
        if any(q["coin"] == o["coin"] for q in bot["positions"]) or open_n >= p["maxOpen"] or same >= p.get("maxSameSide", 3):
            cancelled.append(o["id"]); continue
        qty = min(o["qty"], bot["cash"] * 0.98 * p.get("lev", 1.0) / px)
        if qty * px < p.get("minNotional", 5):
            cancelled.append(o["id"]); continue
        meta = {k: o[k] for k in ("p", "minProb", "ens", "lead", "leadLabel", "scores", "top", "conf", "expR") if k in o}
        meta.update({"via": "order", "orderId": o["id"], "trigger": o["trigger"]})
        pos = make_position(bot, model, o["coin"], o["side"], px, qty, o["riskPs"], o["atr"], t + BAR // 2, t, meta)
        pos["events"].append({"t": t + BAR // 2, "e": f"Breakout order {o['id']} filled at {fmt_px(px)}"})
        # same-bar stop check (conservative: if the fill bar also touched the stop, it was stopped)
        if (o["side"] == "long" and b[3] <= pos["stop"]) or (o["side"] == "short" and b[2] >= pos["stop"]):
            pos["checkedTo"] = t - 1
        filled.append(pos)
    bot["orders"] = []
    return filled, cancelled


def place_orders(bot, model, hist, live, now, scan, scan_x, risk_mult, brake):
    """Rest a breakout order (buy-stop above the 4h high / sell-stop below the 4h low) on promising coins."""
    p = model["params"]
    if risk_mult == 0: return []
    held = {q["coin"] for q in bot["positions"]}
    slots = p["maxOpen"] - len(bot["positions"])
    placed = []
    eq = equity(bot, live)
    cands = []
    for it in scan:
        if "sides" not in it or it["coin"] in held: continue
        for side in (["long", "short"] if p["allowShort"] else ["long"]):
            sd = it["sides"][side]; sx = scan_x[(it["coin"], side)]
            if sd.get("evR", -1) < p.get("minEvR", 0.1) or sx["ens"] < p["minEdge"]: continue
            if is_blocked(bot, it["coin"], side): continue
            if sx["v"].get("emaTrend", 0) < -0.2 and sd.get("evR", 0) < p.get("strongEvR", 0.3): continue
            cands.append((sd["evR"], it, side, sd, sx))
    cands.sort(key=lambda c: -c[0])
    used = set()
    for pr, it, side, sd, sx in cands:
        if len(placed) >= max(0, slots) * 2 or it["coin"] in used: continue
        bars = hist[it["coin"]][-16:]
        sgn = 1 if side == "long" else -1
        trig = (max(b[2] for b in bars) * 1.001) if side == "long" else (min(b[3] for b in bars) * 0.999)
        if (trig - live[it["coin"]]) * sgn <= 0: continue
        risk_ps = p["atrStop"] * sx["a"]
        k = cost_R(p, trig, risk_ps)
        exp_R = sd["evR"]
        conf = min(1.5, max(0.5, 0.6 + exp_R * 1.5))
        qty = eq * risk_now(bot, p)[0] * conf * risk_mult * 0.8 / risk_ps
        qty = min(qty, eq * p["maxNotionalPct"] / trig)
        if p.get("fixedStake"): qty = p["fixedStake"] / trig
        if qty * trig < p.get("minNotional", 5): continue
        bot["orderSeq"] = bot.get("orderSeq", 0) + 1
        o = {"id": f"O{bot['orderSeq']:04d}", "coin": it["coin"], "side": side, "trigger": trig, "qty": qty,
             "riskPs": risk_ps, "atr": sx["a"], "placedAt": now, "placedBar": now // BAR * BAR,
             "expires": now + p["orderHours"] * H, "p": sd["p"], "minProb": p["minProb"], "ens": round(sx["ens"], 3),
             "lead": sx["lead"], "leadLabel": STRAT_LABEL[side].get(sx["lead"], "Learned pattern"), "scores": sx["s"], "top": sd["top"],
             "conf": round(conf, 2), "expR": round(exp_R, 2)}
        bot["orders"].append(o); placed.append(o["id"]); used.add(it["coin"])
        if it.get("decision") in (None, "No trade"):
            it["order"] = f"{'Buy' if side == 'long' else 'Sell'} breakout order at {fmt_px(trig)} for {p['orderHours']}h"
    return placed


def patterns(model):
    res = model["resolved"]
    out = {}
    def agg(keyf):
        d = {}
        for r in res:
            k = keyf(r)
            if k is None: continue
            a = d.setdefault(k, [0, 0, 0.0]); a[0] += 1; a[1] += r["y"]; a[2] += r["R"]
        return sorted([{"k": k, "n": v[0], "win": round(v[1] / v[0], 3), "avgR": round(v[2] / v[0], 3)}
                       for k, v in d.items() if v[0] >= 10], key=lambda x: -x["avgR"])
    def sess(r):
        hr = ((r["t"] // H) + 8) % 24
        return "Asia (08-16 SGT)" if 8 <= hr < 16 else "Europe (16-24 SGT)" if hr >= 16 else "US (00-08 SGT)"
    out["coinSide"] = agg(lambda r: f"{r['c']} {r['d']}")
    out["strategy"] = agg(lambda r: f"{STRAT_LABEL[r['d']].get(r.get('lead'), 'No clear setup')}" if r.get("lead") else None)
    out["session"] = agg(lambda r: f"{sess(r)} · {r['d']}")
    out["n"] = len(res)
    return out


def run(state, new, now):
    model = state.get("model") or default_model()
    if model.get("created") is None: model["created"] = now
    # one-time migration of v3 models saved before the minEdge change (the learned model, not the
    # strategy ensemble, is the main signal; the ensemble now only vetoes clearly opposed setups)
    # v4 migration: add new features / setups with neutral weights, keep everything learned so far
    for sd in SIDES:
        m = model["sides"][sd]
        for f in FEATS: m["w"].setdefault(f, 0.0)
        for k in STRATS: m["strat"].setdefault(k, {"w": 1.0, "n": 0, "wins": 0, "sumR": 0.0})
    model.setdefault("configs", {}); model.setdefault("configLog", [])
    if "mom96" in model["sides"]["long"]["w"] and not model.get("migr3b"):
        model["engineVersion"] = 3  # noqa
        model["params"]["minEdge"] = min(model["params"].get("minEdge", -0.6), -0.6)
        model["migr3b"] = True
    if model.get("engineVersion", 4) < 5:
        model["params"].update(V5_PARAMS)
        model.setdefault("lessons", []).append({"date": day_key(now), "text": "Engine v5: calibrated win chance and expected R, trend filter, bank half at +1R and let the rest run to +4R, 2x paper leverage so 3% risk is real, risk scales with the record."})
        model["lessons"] = model["lessons"][-60:]
    for k2, v2 in V5_PARAMS.items(): model["params"].setdefault(k2, v2)
    if not model.get("quick10"):
        # v5.1 (user, 15 Nov): fixed 10 USDT per trade, quick trades
        model["params"].update({"fixedStake": 10.0, "lev": 1.0, "maxOpen": 6, "maxSameSide": 4,
                                "atrStop": 1.5, "rr": 1.5, "timeStopH": 4, "hardStopH": 8,
                                "partialAtR": 1.0, "runnerR": 2.5, "orderHours": 2})
        model["quick10"] = True
        model.setdefault("lessons", []).append({"date": day_key(now), "text": "v5.1: fixed 10 USDT per trade, quick trades (1.5 ATR stop, 1.5R target, half banked at +1R, 4h time stop, 8h max, up to 6 open)."})
        model["lessons"] = model["lessons"][-60:]
    if not model.get("v6"):
        # v6 (user, 17 Nov): bigger trades sized by risk and confidence instead of a fixed 10 USDT,
        # stricter entries for accuracy, winners run further, learning journal
        model["params"].update({"fixedStake": None, "riskPct": 0.02, "lev": 2.0, "maxNotionalPct": 1.0,
                                "maxOpen": 4, "maxSameSide": 3, "minEvR": 0.20, "strongEvR": 0.35,
                                "beAfterR": 1.0, "partialAtR": 1.0, "trailAfterR": 1.5, "runnerR": 3.0})
        model["v6"] = True
        txt = ("v6: trades are now sized by risk (2% of equity at risk, scaled by confidence and the recent record, up to 1x equity per trade) "
               "instead of a fixed 10 USDT, so a typical win is worth about 1-3 USDT instead of a few cents. Entries need +0.20R measured edge "
               "after fees (was +0.10R), coin+side pairs that keep losing are benched, max 4 trades open, and the trailing stop waits until +1.5R.")
        model.setdefault("lessons", []).append({"date": day_key(now), "text": txt}); model["lessons"] = model["lessons"][-60:]
        jot(model, now, "upgrade", txt)
    model["engineVersion"] = 6
    bot = state.get("bot") or default_bot(now=now)
    hist = state.get("history") or {}
    hist, live, warns = merge_history(hist, new, now)
    p0 = model["params"].get("minProb"); cfg0 = (model.get("settings") or {}).get("active")
    learning = learn(model, hist, now)
    p = model["params"]
    if p0 is not None and p["minProb"] != p0:
        jot(model, now, "tuning", f"Win-chance bar moved {p0:.0%} → {p['minProb']:.0%}: "
            + ("recent ideas above the old bar were not paying after fees, so it got pickier." if p["minProb"] > p0
               else "ideas just below the bar have been paying, so it lets a few more through."))
    cfg1 = (learning.get("settings") or {}).get("active")
    if cfg0 and cfg1 and cfg1 != cfg0:
        jot(model, now, "tuning", f"Switched stop/target setting {cfg0} → {cfg1} (stop in ATRs x reward:risk): it measured better after fees on thousands of shadow ideas.")
    daily_review(bot, model, now)
    build_calib(model)

    scan, scan_x = [], {}
    btc_tr = None
    for coin in ["BTC"] + sorted(c for c in hist if c != "BTC"):
        bars = hist.get(coin, [])
        if coin not in live:
            if bars: scan.append({"coin": coin, "price": bars[-1][4], "decision": "No data", "why": "no fresh candles this run"})
            continue
        if len(bars) < MIN_BARS:
            scan.append({"coin": coin, "price": live[coin], "decision": "Warming up",
                         "why": f"needs {MIN_BARS} hours of history (has {len(bars)})"}); continue
        lb = bars + [[now, live[coin], live[coin], live[coin], live[coin], bars[-1][5]]]
        x, a, price = base_features(lb, None if coin == "BTC" else btc_tr)
        if coin == "BTC": btc_tr = {"emaTrend": x["emaTrend"], "ret24": x["_ret24"]}
        x.pop("_ret24", None)
        c24 = bars[-96][4] if len(bars) >= 96 else bars[0][4]
        item = {"coin": coin, "price": price, "chg24": round((price / c24 - 1) * 100, 2),
                "atrPct": round(a / price * 100, 2), "x": {k: round(v, 3) for k, v in x.items()}, "sides": {}}
        for side in SIDES:
            v, s = side_view(x, side)
            pr = predict(model, side, coin, v)
            ens = ensemble(model, side, s)
            lead = lead_strategy(model, side, s)
            pc, evr = cal(model, side, pr, cost_R(p, price, p["atrStop"] * a))
            item["sides"][side] = {"p": pc, "pRaw": round(pr, 4), "evR": evr, "ens": round(ens, 3), "lead": lead,
                                   "leadLabel": STRAT_LABEL[side].get(lead, "Learned pattern"), "scores": s,
                                   "top": top_drivers(model, side, coin, v)}
            scan_x[(coin, side)] = {"p": pc, "pRaw": pr, "evR": evr, "ens": ens, "v": v, "s": s, "lead": lead, "a": a}
        best = max(SIDES if p["allowShort"] else ["long"], key=lambda sd: item["sides"][sd]["evR"])
        item["best"] = best
        scan.append(item)

    if bot.get("btcStart") is None and "BTC" in live: bot["btcStart"] = live["BTC"]
    filled, cancelled = process_orders(bot, model, hist, now)
    closed = manage_positions(bot, model, hist, live, now, scan_x)

    eq = equity(bot, live)
    dd = (bot["peak"] - eq) / bot["peak"] if bot["peak"] else 0
    risk_mult, brake = 1.0, None
    if dd > 0.15: risk_mult, brake = 0.0, "Circuit breaker: drawdown above 15%, no new entries"
    elif dd > 0.08: risk_mult, brake = 0.5, "Drawdown above 8%: risk halved"

    risk_pct, risk_note = risk_now(bot, p)
    opened = []
    held = {pos["coin"] for pos in bot["positions"]}
    just_closed = {tr["coin"] for tr in closed}
    cands = sorted([it for it in scan if "sides" in it], key=lambda it: -it["sides"][it["best"]]["evR"])
    for it in cands:
        coin, side = it["coin"], it["best"]
        sd = it["sides"][side]; sx = scan_x[(coin, side)]
        reasons = []
        if coin in held:
            it["decision"], it["why"] = "Holding", "position already open"; continue
        exp_R = sd["evR"]
        if exp_R < p["minEvR"]: reasons.append(f"measured edge {exp_R:+.2f}R (needs +{p['minEvR']:.2f}R; calibrated win chance {sd['p']:.0%})")
        if sx["ens"] < p["minEdge"]: reasons.append(f"strategies clearly against it (ensemble {sx['ens']:+.2f})")
        if sx["v"].get("emaTrend", 0) < -0.2 and exp_R < p["strongEvR"]: reasons.append("against the 1h trend without a strong edge")
        if coin in just_closed: reasons.append("closed this run, cooling off")
        if is_blocked(bot, coin, side): reasons.append(f"{coin} {side}s are benched: they keep losing in live trading")
        if risk_mult == 0: reasons.append(brake)
        if len(bot["positions"]) >= p["maxOpen"]: reasons.append(f"already {p['maxOpen']} open positions")
        elif sum(1 for q in bot["positions"] if q["side"] == side) >= p.get("maxSameSide", 3):
            reasons.append(f"already {p.get('maxSameSide', 3)} {side} trades open (coins move together, so this limits one-way risk)")
        if reasons:
            it["decision"], it["why"] = "No trade", "; ".join(reasons); continue
        long = side == "long"; sgn = 1 if long else -1
        price = live[coin] * (1 + sgn * p["slipPct"])
        risk_ps = p["atrStop"] * sx["a"]
        conf = min(1.5, max(0.5, 0.6 + exp_R * 1.5))
        qty = eq * risk_pct * conf * risk_mult / risk_ps
        notional = min(qty * price, eq * p["maxNotionalPct"], bot["cash"] * 0.98 * p.get("lev", 1.0))
        if p.get("fixedStake"): notional = min(p["fixedStake"] * (1 if risk_mult else 0), bot["cash"] * 0.98)
        if notional < p.get("minNotional", 5):
            it["decision"], it["why"] = "No trade", "not enough free cash"; continue
        meta = {"p": sd["p"], "pRaw": sd.get("pRaw"), "minProb": p["minProb"], "ens": round(sx["ens"], 3), "lead": sx["lead"],
                "leadLabel": STRAT_LABEL[side].get(sx["lead"], "Learned pattern"), "scores": sx["s"], "top": sd["top"],
                "conf": round(conf, 2), "expR": round(exp_R, 2), "via": "market"}
        pos = make_position(bot, model, coin, side, price, notional / price, risk_ps, sx["a"], now, hist[coin][-1][0], meta)
        held.add(coin); opened.append(pos)
        it["decision"] = "BUY" if long else "SELL SHORT"
        it["why"] = f"{sd['p']:.0%} calibrated win chance, measured edge {exp_R:+.2f}R, led by {sd['leadLabel']}"

    placed = place_orders(bot, model, hist, live, now, scan, scan_x, risk_mult, brake)
    bench = sorted(k for k in bot.get("byCS", {}) if is_blocked(bot, *k.split(":")))
    old = set(bot.get("benched", []))
    for k in bench:
        if k not in old: jot(model, now, "bench", f"Benched {k.replace(':', ' ')}s: they have lost too often in real trades, so it stops taking them until it re-learns.")
    for k in old - set(bench):
        jot(model, now, "bench", f"Un-benched {k.replace(':', ' ')}s: recent results recovered.")
    bot["benched"] = bench
    eq = equity(bot, live)
    bot["equity"] = round(eq, 2)
    bot["peak"] = max(bot["peak"], eq)
    bot["maxDD"] = round(max(bot["maxDD"], (bot["peak"] - eq) / bot["peak"] * 100), 2)
    bot["curve"] = (bot["curve"] + [[now, round(eq, 2), live.get("BTC")]])[-1000:]
    bot["lastRunAt"] = now; bot["runs"] += 1; bot["brake"] = brake
    bot["dailyTarget"] = p.get("dailyTarget", 100.0); bot["today"] = bot["daily"].get(day_key(now), 0.0)
    bot["riskNote"] = risk_note; bot["engineVersion"] = 6
    bot["readiness"] = readiness(bot, now)
    for g in bot.get("goals", []):
        if eq >= g and str(g) not in bot["goalHits"]:
            bot["goalHits"][str(g)] = now

    model["thresholdHist"] = (model["thresholdHist"] + [[now, p["minProb"]]])[-200:]
    model["calibration"] = calibration(model)
    model["settings"] = learning.get("settings")
    model["patterns"] = patterns(model)
    model["learning"] = (model["learning"] + [{
        "t": now, "newIdeas": learning["newIdeas"], "resolved": learning["resolved"], "wins": learning["wins"],
        "bySide": learning["bySide"], "minProb": p["minProb"],
        "strat": {sd: {k: model["sides"][sd]["strat"][k]["w"] for k in STRATS} for sd in SIDES}}])[-60:]

    opened = filled + opened
    return {"bot": bot, "model": model, "history": hist, "opened": opened, "closed": closed, "scan": scan,
            "orders": {"placed": placed, "cancelled": cancelled, "filled": [f["id"] for f in filled]},
            "learning": learning, "warnings": warns, "live": live,
            "risk": {"drawdownPct": round(dd * 100, 2), "brake": brake}}


def daily_review(bot, model, now):
    """Once per SGT day: write yesterday's report card into the journal."""
    today = day_key(now)
    last = bot.get("reviewDay")
    bot["reviewDay"] = today
    if not last or last == today: return
    pnl = bot.get("daily", {}).get(last, 0.0)
    tgt = model["params"].get("dailyTarget", 100.0)
    rec = bot.get("recent", [])[-20:]
    wr = (sum(1 for x in rec if x > 0) / len(rec)) if rec else None
    good = sorted(((k, sum(v)) for k, v in bot.get("byCS", {}).items() if len(v) >= 3), key=lambda t: -t[1])
    parts = [f"Report card for {last}: {pnl:+.2f} USDT ({pnl / tgt * 100:.0f}% of the {tgt:.0f} USDT daily target)."]
    if wr is not None: parts.append(f"Last {len(rec)} trades won {wr:.0%}.")
    if good:
        parts.append(f"Best pair so far: {good[0][0].replace(':', ' ')} ({good[0][1]:+.1f}R).")
        if good[-1][1] < 0: parts.append(f"Worst: {good[-1][0].replace(':', ' ')} ({good[-1][1]:+.1f}R).")
    for sd in SIDES:
        st = model["sides"][sd]["strat"]
        top = max(st, key=lambda k: st[k]["w"]); low = min(st, key=lambda k: st[k]["w"])
        parts.append(f"{sd.title()}s: trusts {STRAT_LABEL[sd].get(top, top)} most ({st[top]['w']:.2f}), {STRAT_LABEL[sd].get(low, low)} least ({st[low]['w']:.2f}).")
    jot(model, now, "daily", " ".join(parts))


def top_drivers(model, side, coin, v):
    w = model["sides"][side]["w"]
    con = sorted(((f, w.get(f, 0) * v.get(f, 0)) for f in FEATS), key=lambda t: -abs(t[1]))
    return [{"f": f, "label": FEAT_LABEL[f], "v": round(val, 3)} for f, val in con[:4]]


def calibration(model):
    out = []
    for lo, hi in [(0, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, 1.01)]:
        sel = [r for r in model["resolved"] if lo <= r["p"] < hi]
        if sel:
            out.append({"bucket": f"{int(lo*100)}-{min(100, int(hi*100))}%", "n": len(sel),
                        "pred": round(sum(r["p"] for r in sel) / len(sel), 3),
                        "actual": round(sum(r["y"] for r in sel) / len(sel), 3),
                        "avgR": round(sum(r["R"] for r in sel) / len(sel), 3)})
    return out


def readiness(bot, now):
    """Live-trading gate: the user's 500 USDT goal, backed by checks that it was not luck."""
    days = (now - bot["startedAt"]) / (24 * H) if bot.get("startedAt") else 0
    n = bot["closedN"]
    pf = bot["grossWin"] / bot["grossLoss"] if bot["grossLoss"] > 0 else (99 if bot["grossWin"] > 0 else 0)
    eq = bot["equity"]
    checks = [
        {"k": "goal", "label": "Equity reaches 500 USDT (5x the start)", "ok": eq >= 500, "val": f"{eq:.2f}"},
        {"k": "trades", "label": "At least 40 closed trades", "ok": n >= 40, "val": f"{n} trades"},
        {"k": "pf", "label": "Profit factor 1.3 or better", "ok": n >= 10 and pf >= 1.3, "val": f"{pf:.2f}" if n else "-"},
        {"k": "dd", "label": "Never fell more than 35% from a peak", "ok": bot["maxDD"] < 35 and n >= 10, "val": f"{bot['maxDD']:.1f}%"},
        {"k": "days", "label": "At least 7 days of trading", "ok": days >= 7, "val": f"{days:.1f} days"}]
    return {"ready": all(c["ok"] for c in checks), "passed": sum(c["ok"] for c in checks), "checks": checks}


# ---------------------------------------------------------------- db doc helpers
def load_doc(path):
    if not os.path.exists(path): return None
    d = json.load(open(path))
    if isinstance(d, dict) and isinstance(d.get("data"), dict) and ("version" in d or "exists" in d):
        d = d["data"]  # unwrap {exists, version, data}
    return d


def cmd_state(dirp, out):
    """DOCS dir holds bot.json, model.json, candles_<COIN>.json ({bars}), pending_<COIN>.json ({items})."""
    bot = load_doc(os.path.join(dirp, "bot.json"))
    model = load_doc(os.path.join(dirp, "model.json"))
    hist, pend = {}, []
    for c in COINS:
        cd = load_doc(os.path.join(dirp, f"candles_{c}.json"))
        if cd and cd.get("bars"): hist[c] = cd["bars"]
        pd = load_doc(os.path.join(dirp, f"pending_{c}.json"))
        if pd and pd.get("items"): pend += pd["items"]
    if model is not None: model["pending"] = pend
    json.dump({"bot": bot, "model": model, "history": hist}, open(out, "w"))
    print(json.dumps({"bot": bot is not None, "model": model is not None,
                      "bars": {c: len(v) for c, v in hist.items()},
                      "lastBar": {c: v[-1][0] for c, v in hist.items()}, "pending": len(pend)}))


def cmd_export(rep_path, dirp, now):
    rep = json.load(open(rep_path))
    os.makedirs(dirp, exist_ok=True)
    man = {}
    def put(path, data):
        fn = path.replace("/", "_") + ".json"
        json.dump(data, open(os.path.join(dirp, fn), "w"), allow_nan=False, separators=(",", ":"))
        man[path] = fn
    model = dict(rep["model"]); pend = model.pop("pending", [])
    put("desk/bot", rep["bot"])
    put("desk/model", model)
    put("desk/market", {"t": now, "scan": rep["scan"], "warnings": rep["warnings"], "risk": rep["risk"],
                        "minProb": rep["model"]["params"]["minProb"], "live": rep["live"]})
    for c, bars in rep["history"].items():
        put(f"candles/{c}", {"bars": bars, "updatedAt": now})
    by = {}
    for it in pend: by.setdefault(it["c"], []).append(it)
    for c in rep["history"]:
        put(f"pending/{c}", {"items": by.get(c, []), "updatedAt": now})
    opened_ids = {t["id"] for t in rep["opened"]}
    for tr in rep["opened"] + rep["closed"] + [ps for ps in rep["bot"]["positions"] if ps["id"] not in opened_ids]:
        put(f"trades/{tr['id']}", tr)
    rid = dt.datetime.fromtimestamp((now + 8 * H) / 1000, dt.timezone.utc).strftime("%Y%m%d-%H%M")
    b = rep["bot"]; L = rep["learning"]
    put(f"runs/{rid}", {"id": rid, "t": now, "equity": b["equity"], "opened": [t["id"] for t in rep["opened"]],
                        "closed": [[t["id"], t["pnl"]] for t in rep["closed"]], "open": len(b["positions"]),
                        "newIdeas": L["newIdeas"], "resolved": L["resolved"], "wins": L["wins"],
                        "minProb": rep["model"]["params"]["minProb"], "warnings": rep["warnings"],
                        "orders": rep.get("orders", {}), "summary": "", "marketNote": ""})
    json.dump(man, open(os.path.join(dirp, "manifest.json"), "w"), indent=1)
    print(json.dumps({"docs": len(man), "runId": rid, "files": man}, indent=1))


# ---------------------------------------------------------------- self test
def synth(n, seed, drift=0.0003, vol=0.008, start=100.0, t0=0):
    rnd = random.Random(seed); bars, px = [], start
    for i in range(n):
        o = px; c = o * math.exp(rnd.gauss(drift, vol))
        h = max(o, c) * (1 + abs(rnd.gauss(0, vol / 2))); l = min(o, c) * (1 - abs(rnd.gauss(0, vol / 2)))
        bars.append([t0 + i * BAR, o, h, l, c, 1000 * (1 + abs(rnd.gauss(0, 0.5)))]); px = c
    return bars


def selftest():
    import io, contextlib, shutil, tempfile
    t0 = 1_790_000_000_000 // H * H
    coins = {"BTC": synth(700, 1, start=70000, t0=t0, vol=0.004), "PEPE": synth(700, 2, drift=-0.0006, start=0.00000506, t0=t0, vol=0.006),
             "WIF": synth(700, 3, drift=0.0008, start=0.16, t0=t0, vol=0.007), "SOL": synth(700, 4, start=86, t0=t0, vol=0.005)}
    now = t0 + 300 * BAR + 5 * 60000
    rep = run({}, {c: b[:301] for c, b in coins.items()}, now)
    assert rep["bot"]["startBalance"] == 100
    assert rep["learning"]["resolved"] > 0
    nfilled = 0
    for k in range(1, 24):
        now = t0 + (300 + 16 * k) * BAR + 5 * 60000
        st = json.loads(json.dumps({"bot": rep["bot"], "model": rep["model"], "history": rep["history"]}))
        rep = run(st, {c: b[300 + 16 * k - 20: 300 + 16 * k + 1] for c, b in coins.items()}, now)
        nfilled += len(rep["orders"]["filled"])
    bot = rep["bot"]
    assert abs(equity(bot, rep["live"]) - bot["equity"]) < 0.05
    assert bot["closedN"] == bot["wins"] + bot["losses"]
    _, _, w = merge_history({}, {"BTC": coins["BTC"][:10]}, now)
    assert any("stale" in x for x in w)
    json.dumps(rep, allow_nan=False)
    tmp = tempfile.mkdtemp()
    json.dump(rep, open(f"{tmp}/rep.json", "w"))
    with contextlib.redirect_stdout(io.StringIO()):
        cmd_export(f"{tmp}/rep.json", f"{tmp}/out", now)
    os.makedirs(f"{tmp}/docs")
    shutil.copy(f"{tmp}/out/desk_bot.json", f"{tmp}/docs/bot.json")
    shutil.copy(f"{tmp}/out/desk_model.json", f"{tmp}/docs/model.json")
    for c in coins:
        shutil.copy(f"{tmp}/out/candles_{c}.json", f"{tmp}/docs/candles_{c}.json")
        shutil.copy(f"{tmp}/out/pending_{c}.json", f"{tmp}/docs/pending_{c}.json")
    with contextlib.redirect_stdout(io.StringIO()):
        cmd_state(f"{tmp}/docs", f"{tmp}/state.json")
    st = json.load(open(f"{tmp}/state.json"))
    assert len(st["model"]["pending"]) == len(rep["model"]["pending"])
    assert len(st["history"]["BTC"]) == len(rep["history"]["BTC"])
    assert rep["model"]["patterns"]["n"] > 0
    print(json.dumps({"ok": True, "orderFills": nfilled, "samples": rep["model"]["samples"], "trades": bot["closedN"],
                      "bySide": bot["bySide"], "open": len(bot["positions"]), "equity": bot["equity"],
                      "minProb": rep["model"]["params"]["minProb"],
                      "strat": {s: {k: rep["model"]["sides"][s]["strat"][k]["w"] for k in STRATS} for s in SIDES}}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--state"); ap.add_argument("--candles"); ap.add_argument("--now", type=int)
    ap.add_argument("--out", default="report.json"); ap.add_argument("--dir"); ap.add_argument("--report")
    a = ap.parse_args()
    if a.cmd == "selftest": return selftest()
    if a.cmd == "state": return cmd_state(a.dir, a.out)
    if a.cmd == "export": return cmd_export(a.report, a.dir, a.now)
    if a.cmd == "run":
        state = json.load(open(a.state)) if a.state else {}
        new = json.load(open(a.candles)) if a.candles else {}
        rep = run(state, new, a.now)
        json.dump(rep, open(a.out, "w"), allow_nan=False)
        b = rep["bot"]
        print(json.dumps({"equity": b["equity"], "cash": round(b["cash"], 2), "open": len(b["positions"]),
                          "opened": [(o["id"], o["coin"], o["side"], o["p"]) for o in rep["opened"]],
                          "closed": [(c["id"], c["coin"], c["side"], c["exitReason"], c["pnl"]) for c in rep["closed"]],
                          "learning": {k: v for k, v in rep["learning"].items() if k != "threshold"},
                          "threshold": rep["learning"]["threshold"].get("note") or rep["learning"]["threshold"].get("minProb"),
                          "scan": [(s["coin"], s.get("best"), s.get("sides", {}).get(s.get("best"), {}).get("p"),
                                    s.get("decision"), s.get("why")) for s in rep["scan"]],
                          "warnings": rep["warnings"]}, indent=1))
        return
    print("unknown command"); sys.exit(1)


if __name__ == "__main__":
    main()
