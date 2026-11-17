"""Turns raw data into the numbers the models look at. Same code for live trading, backfill and replay.

Key domain fact (read from each market): these markets settle on Chainlink's 60-second TWAP at the close
versus the TWAP at the open ("price to beat"). In the last 60 seconds part of the result is already locked in,
so the fair probability is computed for the *average* of the remaining path, not the spot price.
"""
import math
from datetime import datetime, timezone

from util import norm_cdf

SCHEMA = 1
NAMES = [
    "t", "tau", "d_open", "z_fair", "p_fair", "twap_frac", "vol_cl_300", "vol_bn_60", "vol_bn_300", "vol_bn_900",
    "vol_ratio", "ret_bn_10", "ret_bn_30", "ret_bn_60", "ret_bn_180", "lead_5", "flow_30", "flow_120", "volu_60",
    "book_imb", "eth_ret_60", "eth_ret_300", "funding", "premium", "oi_chg", "liq_imb", "fut_flow",
    "pm_mid_up", "pm_spread", "pm_imb", "hour_sin", "hour_cos", "prev_outcome", "cl_age", "basis",
]
IDX = {n: i for i, n in enumerate(NAMES)}


def _std(xs):
    n = len(xs)
    if n < 10:
        return None
    m = sum(xs) / n
    return math.sqrt(max(sum((x - m) ** 2 for x in xs) / (n - 1), 0.0))


def _rets(prices):
    return [math.log(b / a) for a, b in zip(prices, prices[1:]) if a > 0 and b > 0]


def _closes(series, ts, back):
    """Closes for seconds [ts-back, ts) from a dict ts->value (value may be a tuple; close at [3])."""
    out = []
    for s in range(int(ts) - back, int(ts)):
        v = series.get(s)
        if v is not None:
            out.append(v[3] if isinstance(v, tuple) else v)
    return out


def _price_at(series, s, max_back=5):
    for k in range(max_back + 1):
        v = series.get(int(s) - k)
        if v is not None:
            return v[3] if isinstance(v, tuple) else v
    return None


def fair_prob(price, open_ref, sigma, ts, end, twap_secs, twap_obs_avg, twap_obs_n):
    """P(TWAP_close >= open) for a driftless random walk with per-second log vol sigma."""
    if not price or not open_ref or not sigma or sigma <= 0:
        return None, None
    r = max(end - ts, 0.0)
    L = float(twap_secs or 0)
    if L <= 0:
        mean, var = price, (price * sigma) ** 2 * max(r, 1e-6)
    else:
        a = end - L
        if ts <= a:
            mean, var = price, (price * sigma) ** 2 * ((a - ts) + L / 3.0)
        else:
            k = min(twap_obs_n, L - r) if twap_obs_n else 0
            obs = twap_obs_avg if twap_obs_avg else price
            k = max(k, 0.0)
            # unobserved seconds inside the TWAP window but already in the past are filled with the current price
            mean = (k * obs + (L - k) * price) / L
            var = (price * sigma) ** 2 * (r ** 3) / (3.0 * L * L)
    sd = math.sqrt(max(var, 1e-12))
    z = (mean - open_ref) / sd
    z = max(min(z, 12.0), -12.0)
    return z, norm_cdf(z)


def build(ts, start, end, twap_secs, open_ref, cl, bn, eth, ext, book, prev_outcome, coinbase, backfill=False):
    """cl, bn, eth: dicts second->value (bn values are (o,h,l,c,v,taker_buy)). ext: kind->(ts,value)."""
    f = [None] * len(NAMES)
    t = ts - start
    f[IDX["t"]] = t
    f[IDX["tau"]] = end - ts

    bn300 = _closes(bn, ts, 300)
    bn900 = _closes(bn, ts, 900)
    bn60 = _closes(bn, ts, 60)
    vbn60, vbn300, vbn900 = _std(_rets(bn60)), _std(_rets(bn300)), _std(_rets(bn900))
    f[IDX["vol_bn_60"]], f[IDX["vol_bn_300"]], f[IDX["vol_bn_900"]] = vbn60, vbn300, vbn900
    if vbn60 and vbn900:
        f[IDX["vol_ratio"]] = vbn60 / vbn900

    # reference price: Chainlink when live, Binance as a proxy for backfill or an outage
    if cl:
        cl_now_ts = max((s for s in range(int(ts) - 15, int(ts) + 1) if s in cl), default=None)
    else:
        cl_now_ts = None
    if cl_now_ts is not None:
        ref_series, price = cl, cl[cl_now_ts]
        f[IDX["cl_age"]] = ts - cl_now_ts
    else:
        ref_series, price = bn, _price_at(bn, ts)
        f[IDX["cl_age"]] = None if not backfill else 0.0
    cl300 = _closes(ref_series, ts, 300)
    vcl = _std(_rets(cl300)) if cl_now_ts is not None else None
    f[IDX["vol_cl_300"]] = vcl
    sigma = vcl or vbn300
    if sigma and vbn300:
        sigma = max(sigma, 0.5 * vbn300)  # Chainlink is smoothed; do not let it claim near-zero risk

    if price and open_ref:
        f[IDX["d_open"]] = math.log(price / open_ref)
    obs = []
    if twap_secs and ts > end - twap_secs:
        obs = _closes(ref_series, ts + 1, int(ts - (end - twap_secs)) + 1)
    f[IDX["twap_frac"]] = (len(obs) / twap_secs) if twap_secs else 0.0
    z, p = fair_prob(price, open_ref, sigma, ts, end, twap_secs, (sum(obs) / len(obs)) if obs else None, len(obs))
    f[IDX["z_fair"]], f[IDX["p_fair"]] = z, p

    def ret(series, back):
        a, b = _price_at(series, ts - back), _price_at(series, ts)
        return math.log(b / a) if a and b else None

    for back in (10, 30, 60, 180):
        f[IDX[f"ret_bn_{back}"]] = ret(bn, back)
    if cl_now_ts is not None:
        rb, rc = ret(bn, 5), ret(cl, 5)
        if rb is not None and rc is not None:
            f[IDX["lead_5"]] = rb - rc

    def flow(back):
        v = tb = 0.0
        for s in range(int(ts) - back, int(ts)):
            k = bn.get(s)
            if k:
                v += k[4]
                tb += k[5]
        return (tb / v - 0.5) if v > 0 else None

    f[IDX["flow_30"]], f[IDX["flow_120"]] = flow(30), flow(120)
    v60 = sum(bn[s][4] for s in range(int(ts) - 60, int(ts)) if s in bn)
    v900 = sum(bn[s][4] for s in range(int(ts) - 900, int(ts)) if s in bn)
    if v900 > 0:
        f[IDX["volu_60"]] = v60 / (v900 / 15.0)

    if not backfill:
        if book is not None and isinstance(book, dict) and "binance" in book:
            bq, aq = book["binance"]
            if bq + aq > 0:
                f[IDX["book_imb"]] = (bq - aq) / (bq + aq)
        f[IDX["eth_ret_60"]] = ret(eth, 60) if eth else None
        f[IDX["eth_ret_300"]] = ret(eth, 300) if eth else None
        if ext:
            if "funding" in ext:
                f[IDX["funding"]] = ext["funding"][1]
            if "premium" in ext:
                f[IDX["premium"]] = ext["premium"][1]
            if "oi_chg" in ext:
                f[IDX["oi_chg"]] = ext["oi_chg"]
            if "liq_long_5m" in ext and "liq_short_5m" in ext:
                ll, ss = ext["liq_long_5m"][1], ext["liq_short_5m"][1]
                f[IDX["liq_imb"]] = (ss - ll) / (ss + ll + 1e5)
            if "fut_taker_buy" in ext and "fut_taker_sell" in ext:
                b, s = ext["fut_taker_buy"][1], ext["fut_taker_sell"][1]
                if b + s > 0:
                    f[IDX["fut_flow"]] = (b - s) / (b + s)
        if book is not None and isinstance(book, dict) and "pm" in book:
            pm = book["pm"]
            if pm.get("mid_up") is not None:
                f[IDX["pm_mid_up"]] = pm["mid_up"]
                f[IDX["pm_spread"]] = pm["spread_up"]
                f[IDX["pm_imb"]] = pm["imb"]
        if coinbase and price:
            f[IDX["basis"]] = math.log(price / coinbase)
    else:
        f[IDX["eth_ret_60"]] = ret(eth, 60) if eth else None
        f[IDX["eth_ret_300"]] = ret(eth, 300) if eth else None

    h = datetime.fromtimestamp(ts, timezone.utc)
    ang = 2 * math.pi * (h.hour + h.minute / 60) / 24
    f[IDX["hour_sin"]], f[IDX["hour_cos"]] = math.sin(ang), math.cos(ang)
    f[IDX["prev_outcome"]] = prev_outcome
    return [None if (v is not None and (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))) else
            (round(v, 8) if isinstance(v, float) else v) for v in f]


def get(f, name):
    return f[IDX[name]]
