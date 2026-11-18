"""PMX shared core: settings, clock, HTTP, paper account.

PMX runs two paper-trading desks on Polymarket:
  - the Weather desk (daily highest-temperature markets, forecast models vs market prices)
  - the Arbitrage desk (sets of outcomes that together cost less than they are guaranteed to pay)

PAPER TRADING ONLY. There is no wallet, no key and no code that sends an order to Polymarket.
Polymarket does not accept new orders from Singapore or from the United States (where GitHub runs this bot),
so a real-money mode is deliberately not part of this project.
"""
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(ROOT), "pm5"))  # after pmx/, so names never clash

import util as U  # noqa: E402  (pm5/util.py: http with retries, json helpers, SGT time)

PAPER_ONLY = True
WORK = os.environ.get("PMX_WORK", os.path.join(ROOT, "work"))
PUBLISH_DIR = os.environ.get("PMX_PUBLISH", os.path.join(WORK, "publish"))
BRANCH = "pmx-data"

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
ENSEMBLE = "https://ensemble-api.open-meteo.com/v1/ensemble"
AWC = "https://aviationweather.gov/api/data"

FetchError = U.FetchError

# ---------------------------------------------------------------- clock + http (swappable for the offline test)
_clock = {"fn": time.time}
_http = {"fn": U.http_json}


def now():
    return _clock["fn"]()


def http(url, data=None, timeout=10, retries=2):
    return _http["fn"](url, data=data, timeout=timeout, retries=retries)


# ---------------------------------------------------------------- settings
DEFAULTS = {
    "reset_token": "1",
    "weather_starting_balance": 1000.0,
    "arb_starting_balance": 1000.0,
    "max_stake_pct": 0.03,            # weather: most of the account one bucket bet may use
    "max_event_pct": 0.08,            # weather: most of the account one city/day may use
    "daily_loss_limit_pct": 0.10,
    "liquidity_take_pct": 0.6,        # only take this share of the size shown in the book
    "arb_max_set_pct": 0.15,          # arbitrage: most of the account one set may use
    "arb_max_days": 10,               # arbitrage: only lock money up for sets that resolve within this many days
    "max_challengers": 12,
    "min_trades_for_promotion": 40,
    # readiness gate ("target reached") - paper results a strategy must show
    "target_days": 14,
    "target_weather_trades": 150,
    "target_weather_roi": 0.08,
    "target_arb_sets": 40,
    "target_arb_profit": 20.0,
    "target_max_drawdown": 0.25,
}
LIMITS = {
    "weather_starting_balance": (10, 1_000_000), "arb_starting_balance": (10, 1_000_000),
    "max_stake_pct": (0.002, 0.10), "max_event_pct": (0.005, 0.3), "daily_loss_limit_pct": (0.01, 0.5),
    "liquidity_take_pct": (0.1, 1.0), "arb_max_set_pct": (0.01, 0.5), "arb_max_days": (0.5, 60),
    "max_challengers": (2, 30), "min_trades_for_promotion": (10, 1000), "target_days": (1, 365),
    "target_weather_trades": (10, 5000), "target_weather_roi": (0.0, 1.0), "target_arb_sets": (5, 5000),
    "target_arb_profit": (0, 1_000_000), "target_max_drawdown": (0.02, 0.9),
}


def _sanitize(raw):
    s = dict(DEFAULTS)
    for k, v in (raw or {}).items():
        if k not in DEFAULTS:
            continue
        if k == "reset_token":
            s[k] = str(v)
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        lo, hi = LIMITS[k]
        s[k] = min(max(v, lo), hi)
    for k in ("max_challengers", "min_trades_for_promotion", "target_weather_trades", "target_arb_sets"):
        s[k] = int(s[k])
    return s


def load_settings():
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        try:
            return _sanitize(http(f"https://raw.githubusercontent.com/{repo}/main/pmx/settings.json", retries=1)), "github"
        except Exception:  # noqa: BLE001
            pass
    try:
        with open(os.path.join(ROOT, "settings.json")) as f:
            return _sanitize(json.load(f)), "file"
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULTS), "defaults"


# ---------------------------------------------------------------- order-book maths
def fee_per_share(price, rate, exp=1.0):
    """Polymarket taker fee per share: rate x (p(1-p))^exp. Zero on fee-free markets."""
    if not rate:
        return 0.0
    return rate * (price * (1.0 - price)) ** exp


def walk_cost(asks, shares, take_pct, rate, exp, limit=1.0):
    """All-in cost (fees included) to buy exactly `shares` now, or None if the book is too thin."""
    need, cost, last = shares, 0.0, None
    for price, size in asks:
        if price > limit + 1e-9:
            break
        q = min(size * take_pct, need)
        if q <= 0:
            continue
        cost += q * (price + fee_per_share(price, rate, exp))
        need -= q
        last = price
        if need <= 1e-9:
            return {"cost": cost, "avg": cost / shares, "worst": last}
    return None


def walk_budget(asks, budget, take_pct, rate, exp, limit=1.0):
    """Buy as many shares as `budget` allows at prices <= limit. Shares to 2 decimals."""
    shares, spent, fees = 0.0, 0.0, 0.0
    for price, size in asks:
        if price > limit + 1e-9 or spent >= budget - 1e-9:
            break
        unit = price + fee_per_share(price, rate, exp)
        q = int(min(size * take_pct, (budget - spent) / unit) * 100) / 100.0
        if q <= 0:
            continue
        f = q * fee_per_share(price, rate, exp)
        shares += q
        spent += q * price + f
        fees += f
    return {"shares": round(shares, 2), "cost": round(spent, 5), "fee": round(fees, 5),
            "avg": (spent / shares) if shares else None}


def walk_sell(bids, shares, take_pct, rate, exp, limit=0.0):
    sold, gross, fees = 0.0, 0.0, 0.0
    for price, size in bids:
        if price < limit - 1e-9 or sold >= shares - 1e-9:
            break
        q = int(min(size * take_pct, shares - sold) * 100) / 100.0
        if q <= 0:
            continue
        sold += q
        gross += q * price
        fees += q * fee_per_share(price, rate, exp)
    return {"shares": round(sold, 2), "proceeds": round(gross - fees, 5), "fee": round(fees, 5),
            "avg": (gross / sold) if sold else None}


# ---------------------------------------------------------------- paper account
class Account:
    """A paper account. Positions hold legs; a leg is (market id, token, outcome label, shares, cost)."""

    def __init__(self, d=None, start=1000.0):
        d = d or {}
        self.start = float(d.get("start", start))
        self.cash = float(d.get("cash", self.start))
        self.positions = d.get("positions", [])
        self.closed = d.get("closed", [])
        self.curve = d.get("curve", [])
        self.day = d.get("day", {})
        self.created = d.get("created", now())
        self.peak = float(d.get("peak", self.start))
        self.max_dd = float(d.get("max_dd", 0.0))
        self.seq = int(d.get("seq", 0))

    def to_dict(self):
        return {"start": self.start, "cash": self.cash, "positions": self.positions, "closed": self.closed[-3000:],
                "curve": self.curve[-4000:], "day": self.day, "created": self.created, "peak": self.peak,
                "max_dd": self.max_dd, "seq": self.seq}

    def nid(self, prefix):
        self.seq += 1
        return f"{prefix}{self.seq}"

    def open(self, pos):
        self.cash -= pos["cost"]
        pos.setdefault("opened", now())
        self.positions.append(pos)
        self._today()["spent"] += pos["cost"]
        return pos

    def close(self, pos, proceeds, how, extra=None):
        self.positions = [p for p in self.positions if p["id"] != pos["id"]]
        self.cash += proceeds
        pnl = proceeds - pos["cost"]
        rec = dict(pos) | {"closed_at": now(), "proceeds": round(proceeds, 5), "pnl": round(pnl, 5), "how": how,
                           "won": pnl > 0} | (extra or {})
        self.closed.append(rec)
        self._today()["pnl"] += pnl
        return rec

    def _today(self):
        d = U.sgt_day(now())
        if self.day.get("d") != d:
            self.day = {"d": d, "pnl": 0.0, "spent": 0.0, "start_equity": self.equity_cost()}
        return self.day

    def equity_cost(self):
        return self.cash + sum(p["cost"] for p in self.positions)

    def equity(self, marks=None):
        marks = marks or {}
        v = self.cash
        for p in self.positions:
            v += marks.get(p["id"], p["cost"])
        return v

    def day_loss_hit(self, limit_pct):
        t = self._today()
        return t["pnl"] < -limit_pct * max(t.get("start_equity") or self.start, 1.0)

    def record(self, marks=None):
        eq = self.equity(marks)
        self.peak = max(self.peak, eq)
        if self.peak > 0:
            self.max_dd = max(self.max_dd, (self.peak - eq) / self.peak)
        t = now()
        if not self.curve or t - self.curve[-1][0] >= 600:
            self.curve.append([round(t), round(eq, 2)])
        return eq

    def stats(self, since=0):
        cl = [c for c in self.closed if c.get("closed_at", 0) >= since]
        pnl = sum(c["pnl"] for c in cl)
        staked = sum(c["cost"] for c in cl)
        wins = sum(1 for c in cl if c["pnl"] > 0)
        rets = [c["pnl"] / c["cost"] for c in cl if c["cost"] > 0]
        t = None
        if len(rets) >= 5:
            m = sum(rets) / len(rets)
            sd = math.sqrt(sum((r - m) ** 2 for r in rets) / (len(rets) - 1)) or 1e-9
            t = m / (sd / math.sqrt(len(rets)))
        return {"n": len(cl), "wins": wins, "win_rate": (wins / len(cl)) if cl else None, "pnl": round(pnl, 2),
                "staked": round(staked, 2), "roi": (pnl / staked) if staked else None, "t_stat": t}
