"""Strategies: when to trade, which side, how much, and when to get out.

A strategy pairs a model with trading rules. The bot does not hard-code one strategy: it keeps a champion
and a pool of challengers, all evaluated on the same markets (see learning.py).
"""
import copy
import random

import features as F
from paper import fee_per_share, quote_cost

DEFAULT_RULES = {
    "t_min": 30,          # earliest entry (seconds into the window)
    "t_max": 285,         # latest entry
    "min_edge": 0.04,     # model probability minus all-in cost per share
    "min_prob": 0.55,     # minimum model probability for the side bought
    "min_price": 0.07,    # avoid lottery tickets...
    "max_price": 0.93,    # ...and picking up pennies in front of a steamroller
    "vol_ratio_max": None,  # skip when short-term vol / 15-min vol is above this (None = no filter)
    "min_depth_usd": 20,  # need at least this much on the best levels
    "size_mult": 1.0,     # scales the Kelly stake
    "exit": "hold",       # hold to settlement, or "edge_exit": sell at the bid when the edge has gone
    "exit_buffer": 0.03,
}

MUTABLE = {
    "t_min": [10, 20, 30, 45, 60, 90, 120, 150, 180, 220, 240, 260],
    "t_max": [120, 180, 220, 250, 270, 285, 292],
    "min_edge": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
    "min_prob": [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 0.97],
    "min_price": [0.03, 0.05, 0.07, 0.1, 0.15],
    "max_price": [0.8, 0.85, 0.9, 0.93, 0.96, 0.98, 0.99],
    "vol_ratio_max": [None, 1.5, 2.0, 3.0],
    "min_depth_usd": [10, 20, 50, 100],
    "size_mult": [0.5, 0.75, 1.0, 1.25],
    "exit": ["hold", "edge_exit"],
    "exit_buffer": [0.02, 0.03, 0.05, 0.08],
}


class Strategy:
    def __init__(self, sid, model_id, rules, parent=None, note="", created=None, origin="seed"):
        self.id, self.model_id = sid, model_id
        self.rules = dict(DEFAULT_RULES) | dict(rules or {})
        self.parent, self.note, self.created, self.origin = parent, note, created, origin

    def to_dict(self):
        return {"id": self.id, "model_id": self.model_id, "rules": self.rules, "parent": self.parent,
                "note": self.note, "created": self.created, "origin": self.origin}

    @staticmethod
    def from_dict(d):
        return Strategy(d["id"], d["model_id"], d["rules"], d.get("parent"), d.get("note", ""), d.get("created"),
                        d.get("origin", "seed"))

    def describe(self):
        r = self.rules
        if r.get("observe"):
            return "stand aside: no trades until a strategy proves a real edge"
        s = (f"enter {r['t_min']}-{r['t_max']}s, edge >= {r['min_edge']:.2f}, prob >= {r['min_prob']:.2f}, "
             f"price {r['min_price']:.2f}-{r['max_price']:.2f}")
        if r["vol_ratio_max"]:
            s += f", skip if vol ratio > {r['vol_ratio_max']}"
        if r["exit"] == "edge_exit":
            s += f", early exit buffer {r['exit_buffer']:.2f}"
        return s

    # ------------------------------------------------------------------ decision
    def decide(self, f, p_up, book, mkt, bankroll, settings):
        """Return (order or None, reason). Order: side, prob, limit, budget, edge, cost_per_share."""
        r = self.rules
        if r.get("observe"):
            return None, "standing aside: no strategy has a proven edge over the market yet"
        t = F.get(f, "t")
        if p_up is None:
            return None, "model has no view"
        if t is None or t < r["t_min"] or t > r["t_max"]:
            return None, "outside entry window"
        vr = F.get(f, "vol_ratio")
        if r["vol_ratio_max"] and vr is not None and vr > r["vol_ratio_max"]:
            return None, f"volatility filter (ratio {vr:.1f})"
        best = None
        for side, q in (("up", p_up), ("down", 1 - p_up)):
            asks = book[side]["asks"]
            if not asks:
                continue
            ask = asks[0][0]
            if ask < r["min_price"] or ask > r["max_price"] or q < r["min_prob"]:
                continue
            depth = sum(p * s for p, s in asks[:3])
            if depth < r["min_depth_usd"]:
                continue
            c1 = ask + fee_per_share(ask, mkt["fee_rate"], mkt["fee_exp"])
            edge = q - c1
            if best is None or edge > best["edge"]:
                best = {"side": side, "prob": q, "ask": ask, "edge": edge, "cost1": c1}
        if best is None:
            return None, "no side passes price/probability/depth rules"
        if best["edge"] < r["min_edge"]:
            return None, f"edge too small ({best['edge']:+.3f})"
        # fractional Kelly on the all-in price, capped
        c = best["cost1"]
        kelly = max((best["prob"] - c) / (1 - c), 0.0)
        frac = min(settings["kelly_fraction"] * kelly * r["size_mult"], settings["max_stake_pct"])
        budget = bankroll * frac
        shares_est = budget / c if c > 0 else 0
        if shares_est < mkt.get("min_size", 5) or budget < 1.0:
            return None, "stake below the market minimum"
        # make sure the edge survives walking the book for the full size
        avg = quote_cost(book[best["side"]]["asks"], shares_est, mkt["fee_rate"], mkt["fee_exp"], settings["liquidity_take_pct"])
        if avg is None:
            return None, "not enough liquidity for the stake"
        if best["prob"] - avg < r["min_edge"] * 0.75:
            return None, f"edge disappears after slippage ({best['prob'] - avg:+.3f})"
        limit = min(best["ask"] + 0.02, r["max_price"])  # tolerate 2 ticks of movement before the fill
        return {"side": best["side"], "prob": best["prob"], "limit": round(limit, 2), "budget": round(budget, 2),
                "edge": best["edge"], "cost_per_share": avg}, "trade"

    def should_exit(self, pos_side, p_up, book, mkt):
        if self.rules["exit"] != "edge_exit" or p_up is None:
            return False
        q = p_up if pos_side == "up" else 1 - p_up
        bids = book[pos_side]["bids"]
        if not bids:
            return False
        bid = bids[0][0]
        net_bid = bid - fee_per_share(bid, mkt["fee_rate"], mkt["fee_exp"])
        return net_bid > q + self.rules["exit_buffer"]


def mutate(base, sid, model_id=None, keys=None, rng=random, note=""):
    s = copy.deepcopy(base.rules)
    keys = keys or rng.sample(list(MUTABLE), k=rng.choice([1, 1, 2]))
    changed = []
    for k in keys:
        if k == "exit_buffer" and s["exit"] == "hold":
            k = "exit"  # the buffer does nothing unless early exits are on
        opts = [v for v in MUTABLE[k] if v != s[k]]
        if opts:
            s[k] = rng.choice(opts)
            changed.append(f"{k} {base.rules[k]} -> {s[k]}")
    if s["t_min"] >= s["t_max"]:
        s["t_min"] = max(10, s["t_max"] - 60)
    return Strategy(sid, model_id or base.model_id, s, parent=base.id, note=note or "; ".join(changed),
                    origin="exploration"), changed
