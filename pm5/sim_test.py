#!/usr/bin/env python3
"""Offline end-to-end test: runs the real controller against a synthetic market with a fake clock.

Usage: python sim_test.py [hours]
Nothing here touches the network.
"""
import math
import os
import random
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
WORK = tempfile.mkdtemp(prefix="pm5sim-")
os.environ["PM5_WORK"] = WORK
os.environ.pop("GITHUB_REPOSITORY", None)

import time as _time  # noqa: E402

HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 6
T0 = 1_794_900_000 + 13  # a window boundary + 13s
START = T0 - 5 * 86400
N = int(5 * 86400 + HOURS * 3600 + 4000)
rng = np.random.default_rng(3)
sig = 8.5e-5 * np.exp(0.6 * np.sin(np.arange(N) / 7000.0))   # changing volatility regimes
lr = rng.standard_normal(N) * sig
lr += 0.15 * np.r_[0, lr[:-1]]                                 # a little momentum (a real, small edge)
price = 64000 * np.exp(np.cumsum(lr))
cl = price.copy()
for i in range(1, N):                                          # Chainlink: smoothed
    cl[i] = 0.7 * cl[i - 1] + 0.3 * price[i]


def idx(ts):
    return int(ts) - START


def twap(ts):
    i = idx(ts)
    return float(np.mean(cl[i - 59:i + 1]))


class Clock:
    t = float(T0)


def fake_time():
    return Clock.t


def fake_sleep(s):
    Clock.t += max(s, 0.05)


_time.time = fake_time
_time.sleep = fake_sleep

import config as C  # noqa: E402
import feeds  # noqa: E402
import market as MK  # noqa: E402
import backfill  # noqa: E402
import run as R  # noqa: E402
from util import FetchError  # noqa: E402

C.WORK = WORK
C.DB_PATH = os.path.join(WORK, "history.sqlite")
C.PUBLISH_DIR = os.path.join(WORK, "publish")


def gamma_market(start):
    from util import iso_utc
    return {"slug": MK.slug_for(start), "outcomes": '["Up", "Down"]', "clobTokenIds": f'["U{start}", "D{start}"]',
            "eventStartTime": iso_utc(start), "endDate": iso_utc(start + 300), "id": str(start), "conditionId": f"0x{start}",
            "question": f"Bitcoin Up or Down - sim {start}", "description": "TWAP sim", "feesEnabled": True,
            "feeSchedule": {"rate": 0.07, "exponent": 1}, "cryptoMarketConfig": {"twapEnabled": True, "twapLookbackSeconds": 60},
            "orderMinSize": 5, "orderPriceMinTickSize": 0.01, "acceptingOrders": True, "closed": False,
            "liquidityClob": 20000, "volumeNum": 5000}


def fake_fetch_market(start, closed=None):
    if start > Clock.t + 86400:
        return None
    m = gamma_market(start)
    if closed and Clock.t > start + 300 + 240:
        m["closed"] = True
        up = twap(start + 300) >= twap(start)
        m["outcomePrices"] = '["1", "0"]' if up else '["0", "1"]'
    return m


def true_prob(ts, start):
    from features import fair_prob
    i = idx(ts)
    s = float(np.std(np.diff(np.log(price[i - 300:i]))))
    end = start + 300
    obs = [cl[idx(x)] for x in range(end - 59, int(ts) + 1)] if ts > end - 60 else []
    z, p = fair_prob(cl[i], twap(start), s, ts, end, 60, (sum(obs) / len(obs)) if obs else None, len(obs))
    return p


def fake_books(m):
    if random.random() < 0.01:
        raise FetchError("sim: book timeout")
    ts = Clock.t
    p = true_prob(ts, m["start"]) or 0.5
    p = min(max(p + random.gauss(0, 0.04), 0.02), 0.98)  # the crowd is noisy
    mid = round(p, 2)
    up_bids = [(round(mid - 0.01 - k * 0.01, 2), 150 + 50 * k) for k in range(6) if mid - 0.01 - k * 0.01 > 0.005]
    up_asks = [(round(mid + 0.01 + k * 0.01, 2), 150 + 50 * k) for k in range(6) if mid + 0.01 + k * 0.01 < 0.995]
    dn_bids = [(round(1 - a, 2), s) for a, s in up_asks]
    dn_asks = [(round(1 - b, 2), s) for b, s in up_bids]
    return {"fetched": ts, "up": {"bids": up_bids, "asks": up_asks, "ts": ts}, "down": {"bids": dn_bids, "asks": dn_asks, "ts": ts}}


def fake_crypto_price(start):
    end = start + 300
    if Clock.t < start + 2:
        return {"open": None, "close": None, "completed": False}
    if Clock.t < end + 8:
        return {"open": twap(start), "close": None, "completed": False}
    return {"open": twap(start), "close": twap(end), "completed": True}


MK.fetch_market = fake_fetch_market
MK.fetch_books = fake_books
MK.crypto_price = fake_crypto_price


def fake_klines(sym, start, end):
    out = {}
    for ts in range(max(start, START + 1000), min(end, int(Clock.t))):
        i = idx(ts)
        p = price[i] * 1.0009 * (1 if sym == "BTCUSDT" else 0.05)
        out[ts] = (p, p * 1.0001, p * 0.9999, p, 1.0 + abs(lr[i]) * 2e4, 0.5 + 0.5 * math.tanh(lr[i] * 5e3))
    return out


backfill._klines = lambda a, b: fake_klines('BTCUSDT', a, b)
backfill._outcomes = lambda starts: {x: ('UP' if twap(x + 300) >= twap(x) else 'DOWN') for x in starts if x + 300 < Clock.t}


class FakeCL:
    def __init__(self, health):
        self.h, self.reconnects, self.stop = health, 0, False
        self.prices = {}

    def start(self):
        pass

    def _sync(self):
        now = int(Clock.t)
        for ts in range(now - 300, now):
            self.prices.setdefault(ts, float(cl[idx(ts)]))
        self.h.ok("chainlink")

    def take_rows(self):
        self._sync()
        return []

    def series(self, since):
        self._sync()
        return sorted((t, p) for t, p in self.prices.items() if t >= since)

    def last(self):
        self._sync()
        t = max(self.prices)
        return t, self.prices[t]


class FakeSpot(feeds.Spot):
    def poll(self, now, backlog_s=900):
        last = next(reversed(self.btc)) if self.btc else int(now) - backlog_s
        for ts in range(last + 1, int(now)):
            i = idx(ts)
            p = price[i] * 1.0009
            v = (p, p * 1.0001, p * 0.9999, p, 1.0 + abs(lr[i]) * 2e4, 0.5 + 0.5 * math.tanh(lr[i] * 5e3))
            self.btc[ts] = v
            self.eth[ts] = p * 0.05
        while len(self.btc) > 4 * 3600:
            self.btc.popitem(last=False)
            self.eth.popitem(last=False)
        self.h.ok("binance_btc")
        self.book = (now, 5.0, 4.0, price[idx(now)], 0.01)
        self.coinbase = (now, price[idx(now)])
        self.okx = {"funding": (now, 1e-5), "premium": (now, -1e-4)}


R.ChainlinkStream = FakeCL
R.Spot = FakeSpot
MK.official_outcome = lambda s: None if Clock.t < s + 380 else ("UP" if twap(s + 300) >= twap(s) else "DOWN")
R.MK.official_outcome = MK.official_outcome

if __name__ == "__main__":
    random.seed(5)
    t_real = _time.perf_counter() if hasattr(_time, "perf_counter") else 0
    c = R.Controller(HOURS * 60)
    c.run()
    import json
    data = json.load(open(os.path.join(C.PUBLISH_DIR, "data.json")))
    ov = data["overview"]
    print("\n==== SIM RESULT ====")
    print("equity", round(ov["equity"], 2), "trades", ov["trades"], "win", ov["win_rate"], "brier", ov["brier"])
    print("lab", [(x["id"], x["status"], x["shadow"]["trades"], round(x["shadow"]["pnl"], 2)) for x in data["lab"]])
    print("experiments", len(data["journal"]["experiments"]), "trade analyses", len(data["journal"]["trades"]),
          "daily", len(data["journal"]["daily"]), "discoveries", len(data["journal"]["discoveries"]))
    print("integrity", data["journal"]["integrity"][-3:])
    print("research", json.dumps(data["performance"]["research"])[:800])
    print("sizes", {f: os.path.getsize(os.path.join(C.PUBLISH_DIR, f)) for f in os.listdir(C.PUBLISH_DIR)})
    print("loops", c.loop_n, "errors", c.loop_errors, "real secs", round(_time.perf_counter() - t_real, 1))
    print("WORK", WORK)
