"""Paper Trading Engine. Simulates real Polymarket execution: walks the order book level by level,
only takes part of the displayed size, charges the market's taker fee, enforces the minimum order,
allows partial fills, and settles at $1 / $0. Selling early happens at the bids, never the midpoint.

PAPER ONLY: there is no code path here (or anywhere in pm5) that talks to an exchange's order endpoint.
"""
import time


def fee_per_share(price, rate, exp=1.0):
    """Polymarket taker fee per share: rate x (p x (1-p))^exp (crypto: rate 0.07, exp 1 -> 1.75c at 50c)."""
    return rate * (price * (1.0 - price)) ** exp


def round_fee(x):
    return round(x + 1e-12, 5)


def walk_buy(asks, budget, limit, take_pct, rate, exp):
    """Buy as many shares as `budget` (USDC, fees included) allows at prices <= limit."""
    fills, shares, spent, fees = [], 0.0, 0.0, 0.0
    for price, size in asks:
        if price > limit + 1e-9 or spent >= budget - 1e-9:
            break
        avail = size * take_pct
        unit = price + fee_per_share(price, rate, exp)
        q = min(avail, (budget - spent) / unit)
        q = int(q * 100) / 100.0  # shares to 2 decimals
        if q <= 0:
            continue
        fee = round_fee(q * fee_per_share(price, rate, exp))
        fills.append((price, q))
        shares += q
        spent += q * price + fee
        fees += fee
    avg = (spent - fees) / shares if shares else None
    return {"fills": fills, "shares": round(shares, 2), "cost": round(spent, 5), "fee": round(fees, 5), "avg": avg}


def walk_sell(bids, shares, limit, take_pct, rate, exp):
    fills, sold, gross, fees = [], 0.0, 0.0, 0.0
    for price, size in bids:
        if price < limit - 1e-9 or sold >= shares - 1e-9:
            break
        q = int(min(size * take_pct, shares - sold) * 100) / 100.0
        if q <= 0:
            continue
        fee = round_fee(q * fee_per_share(price, rate, exp))
        fills.append((price, q))
        sold += q
        gross += q * price
        fees += fee
    return {"fills": fills, "shares": round(sold, 2), "proceeds": round(gross - fees, 5), "fee": round(fees, 5),
            "avg": (gross / sold) if sold else None}


def quote_cost(asks, shares, rate, exp, take_pct):
    """Average all-in cost per share to buy `shares` now (None if the book cannot fill it)."""
    need, cost = shares, 0.0
    for price, size in asks:
        q = min(size * take_pct, need)
        cost += q * (price + fee_per_share(price, rate, exp))
        need -= q
        if need <= 1e-9:
            return cost / shares
    return None


class Account:
    """The real (champion) paper account: cash, open positions, history, risk limits."""

    def __init__(self, d=None, start_balance=1000.0):
        d = d or {}
        self.start_balance = d.get("start_balance", start_balance)
        self.cash = d.get("cash", start_balance)
        self.positions = d.get("positions", [])      # open
        self.closed = d.get("closed", [])            # all closed trades (kept for history)
        self.orders = d.get("orders", [])            # every simulated order (recent)
        self.rejected = d.get("rejected", [])        # rejected opportunities (recent)
        self.next_id = d.get("next_id", 1)
        self.peak = d.get("peak", start_balance)
        self.max_dd = d.get("max_dd", 0.0)
        self.day = d.get("day", {})                  # sgt-day -> {"pnl":..., "start_equity":...}
        self.loss_streak = d.get("loss_streak", 0)
        self.paused_until = d.get("paused_until", 0)
        self.equity_curve = d.get("equity_curve", [])  # [ts, equity] every few minutes
        self.archive = d.get("archive", {"trades": 0, "wins": 0, "pnl": 0.0, "fees": 0.0, "staked": 0.0})

    def trim(self, keep=5000):
        """Very old closed trades move into running totals (they stay in the SQLite history)."""
        if len(self.closed) > keep + 1000:
            old, self.closed = self.closed[:-keep], self.closed[-keep:]
            a = self.archive
            for t in old:
                a["trades"] += 1
                a["wins"] += t["pnl"] > 0
                a["pnl"] += t["pnl"]
                a["fees"] += t.get("fee", 0) + (t.get("exit_fee") or 0)
                a["staked"] += t["cost"]

    def to_dict(self):
        return {k: getattr(self, k) for k in ("start_balance", "cash", "positions", "closed", "orders", "rejected",
                                              "next_id", "peak", "max_dd", "day", "loss_streak", "paused_until",
                                              "equity_curve", "archive")} | {"orders": self.orders[-400:],
                                                                   "rejected": self.rejected[-400:],
                                                                   "equity_curve": self.equity_curve[-6000:]}

    def open_value(self, marks):
        """Mark open positions at the executable bid (what we could actually sell for)."""
        v = 0.0
        for p in self.positions:
            v += p["shares"] * marks.get((p["slug"], p["side"]), p["entry_avg"])
        return v

    def equity(self, marks=None):
        return self.cash + self.open_value(marks or {})

    def has_position(self, slug):
        return any(p["slug"] == slug for p in self.positions)

    def open_position(self, slug, start, side, res, meta):
        pid = f"T{self.next_id:05d}"
        self.next_id += 1
        self.cash -= res["cost"]
        pos = {"id": pid, "slug": slug, "start": start, "side": side, "shares": res["shares"], "cost": res["cost"],
               "fee": res["fee"], "entry_avg": res["avg"], "fills": res["fills"], "opened": time.time(),
               "status": "open"} | meta
        self.positions.append(pos)
        return pos

    def close_position(self, pos, proceeds, reason, exit_avg=None, exit_fee=0.0, outcome=None):
        self.positions = [p for p in self.positions if p["id"] != pos["id"]]
        self.cash += proceeds
        pnl = proceeds - pos["cost"]
        pos.update({"status": "closed", "closed": time.time(), "proceeds": round(proceeds, 5), "pnl": round(pnl, 5),
                    "exit_reason": reason, "exit_avg": exit_avg, "exit_fee": exit_fee, "outcome": outcome,
                    "roi": pnl / pos["cost"] if pos["cost"] else 0.0})
        self.closed.append(pos)
        self.loss_streak = self.loss_streak + 1 if pnl < 0 else 0
        return pos

    def track(self, ts, day, marks=None):
        eq = self.equity(marks)
        self.peak = max(self.peak, eq)
        if self.peak > 0:
            self.max_dd = max(self.max_dd, (self.peak - eq) / self.peak)
        d = self.day.setdefault(day, {"start_equity": eq})
        d["equity"] = eq
        d["pnl"] = eq - d["start_equity"]
        if not self.equity_curve or ts - self.equity_curve[-1][0] >= 300:
            self.equity_curve.append([int(ts), round(eq, 4)])
        return eq
