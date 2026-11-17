"""Self-Learning Engine: observe -> predict -> evaluate -> analyse -> experiment -> test -> improve -> repeat.

Rules that keep it honest:
- Production (the champion) is never edited in place. Improvements are new versioned challengers.
- A challenger must (1) not lose to the champion when replayed on recorded markets it has never seen,
  then (2) beat the champion over a forward "shadow" period on live markets, by a margin that survives a
  bootstrap significance test corrected for the number of challengers being tried at once.
- If the champion's live record turns clearly negative, it is rolled back to the last validated version.
- The journal labels measured results and AI hypotheses separately.
"""
import collections
import math
import random
import time

import numpy as np

import features as F
import market as MK
import models as M
from paper import Account, fee_per_share, walk_buy, walk_sell
from strategy import Strategy, mutate
from util import clamp, money, sgt_day

STD_T = (30, 90, 150, 210, 270)   # checkpoints used to score probability forecasts
OFFICIAL_T = 150                   # the "official" once-per-market prediction horizon
SHADOW_BANKROLL = 1000.0           # fixed virtual bankroll so challengers are compared like for like
MAX_SHADOW_WINDOWS = 2000          # a challenger that cannot prove itself in ~7 days is retired


# ======================================================================= statistics
def brier(ps, ys):
    return float(np.mean([(p - y) ** 2 for p, y in zip(ps, ys)])) if ps else None


def logloss(p, y):
    p = clamp(p, 1e-4, 1 - 1e-4)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def boot_ci(diffs, alpha=0.05, n=2000, seed=11):
    """One-sided lower bound and upper bound of the mean via bootstrap."""
    d = np.asarray(diffs, dtype=float)
    if len(d) < 5:
        return None, None, float(d.mean()) if len(d) else 0.0
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)), float(d.mean())


def calibration(ps, ys, edges=None):
    edges = edges or [i / 10 for i in range(11)]
    out = []
    for lo, hi in zip(edges, edges[1:]):
        sel = [(p, y) for p, y in zip(ps, ys) if lo <= p < hi or (hi == 1.0 and p == 1.0)]
        if sel:
            out.append({"lo": lo, "hi": hi, "n": len(sel), "pred": float(np.mean([p for p, _ in sel])),
                        "actual": float(np.mean([y for _, y in sel]))})
    return out


def vol_regime(vr):
    if vr is None:
        return "unknown"
    return "calm" if vr < 0.8 else "normal" if vr < 1.4 else "volatile"


# ======================================================================= replay (historical test)
def replay(strategies, models_by_id, rows, settings, slip_ticks=1):
    """Replay strategies on recorded checkpoints (with books). Returns {sid: {start: pnl}} and trade counts.

    Conservative: every fill pays one extra tick versus the recorded book.
    """
    by_win = collections.OrderedDict()
    for slug, start, t, f, book, y in rows:
        if book is None:
            continue
        by_win.setdefault(start, []).append((t, f, book, y))
    cache = {}
    res = {s.id: {"pnl": {}, "trades": 0, "wins": 0} for s in strategies}
    mkt = {"fee_rate": 0.07, "fee_exp": 1.0, "min_size": 5}
    for start, cps in by_win.items():
        y = cps[0][3]
        for s in strategies:
            model = models_by_id.get(s.model_id)
            if model is None:
                continue
            pos = None
            for t, f, cb, _ in cps:
                key = (model.id, start, t)
                if key not in cache:
                    cache[key] = model.predict(f)
                p = cache[key]
                book = MK.expand_book(cb)
                if pos is None:
                    order, _ = s.decide(f, p, book, mkt, SHADOW_BANKROLL, settings)
                    if order:
                        asks = [(min(px + 0.01 * slip_ticks, 0.99), q) for px, q in book[order["side"]]["asks"]]
                        r = walk_buy(asks, order["budget"], order["limit"], settings["liquidity_take_pct"], 0.07, 1.0)
                        if r["shares"] >= 5:
                            pos = {"side": order["side"], "shares": r["shares"], "cost": r["cost"]}
                elif s.should_exit(pos["side"], p, book, mkt):
                    bids = [(max(px - 0.01 * slip_ticks, 0.01), q) for px, q in book[pos["side"]]["bids"]]
                    r = walk_sell(bids, pos["shares"], 0.01, settings["liquidity_take_pct"], 0.07, 1.0)
                    if r["shares"] > 0:
                        left = pos["shares"] - r["shares"]
                        win = (pos["side"] == "up") == (y == 1)
                        pnl = r["proceeds"] + (left if win else 0.0) - pos["cost"]
                        res[s.id]["pnl"][start] = pnl
                        res[s.id]["trades"] += 1
                        res[s.id]["wins"] += pnl > 0
                        pos = "done"
                if pos == "done":
                    break
            if isinstance(pos, dict):
                win = (pos["side"] == "up") == (y == 1)
                pnl = (pos["shares"] if win else 0.0) - pos["cost"]
                res[s.id]["pnl"][start] = pnl
                res[s.id]["trades"] += 1
                res[s.id]["wins"] += pnl > 0
    return res, list(by_win.keys())


# ======================================================================= the brain
class Brain:
    def __init__(self, state, db, journal, settings):
        self.s, self.db, self.j, self.cfg = state, db, journal, settings
        st = self.s
        st.setdefault("models", {})
        st.setdefault("strategies", {})
        st.setdefault("challengers", {})
        st.setdefault("ledgers", {})
        st.setdefault("windows", {})
        st.setdefault("pending", [])
        st.setdefault("champion_history", [])
        st.setdefault("next_ids", {"model": 1, "strategy": 1, "exp": 1})
        st.setdefault("last", {})
        st.setdefault("daily_stats", {})
        st.setdefault("missed", [])
        self.models = {k: M.Model.from_dict(v) for k, v in st["models"].items()}
        self.strats = {k: Strategy.from_dict(v) for k, v in st["strategies"].items()}
        self.account = Account(st.get("account"), settings["starting_balance"])
        self.rng = random.Random(int(time.time()))
        self._p_cache = {}
        self._shadow_book = None

    # ------------------------------------------------------------------ ids & registry
    def _nid(self, kind):
        n = self.s["next_ids"][kind]
        self.s["next_ids"][kind] = n + 1
        return n

    def add_model(self, kind, params, parent=None, note="", trained_until=None):
        mid = f"M{self._nid('model'):03d}"
        m = M.Model(mid, kind, params, parent, note, trained_until)
        self.models[mid] = m
        self.s["models"][mid] = m.to_dict()
        self.j.model_event(f"Model {mid} created", f"{m.describe()}. {note}".strip(), {"model": mid, "kind": kind})
        return m

    def add_strategy(self, model_id, rules, parent=None, note="", origin="seed", status="shadow", hypothesis=None,
                     replay=None, exp_id=None):
        sid = f"S{self._nid('strategy'):03d}"
        st = Strategy(sid, model_id, rules, parent, note, time.time(), origin)
        self.strats[sid] = st
        self.s["strategies"][sid] = st.to_dict()
        self.s["challengers"][sid] = {"status": status, "born": time.time(), "born_window": MK.window_start(time.time()),
                                      "origin": origin, "hypothesis": hypothesis, "replay": replay, "exp": exp_id}
        self.s["ledgers"][sid] = {"pnl": {}, "n_windows": 0, "trades": 0, "wins": 0, "open": {}, "recent": []}
        return st

    @property
    def champion(self):
        return self.strats.get(self.s.get("champion"))

    def active(self):
        out = []
        for sid, c in self.s["challengers"].items():
            if c["status"] in ("champion", "shadow", "backup") and sid in self.strats:
                out.append(self.strats[sid])
        return out

    # ------------------------------------------------------------------ bootstrap
    def bootstrap(self, labelled_rows):
        if self.s.get("champion"):
            return
        self.add_model("market", {}, note="Benchmark: what the crowd already prices in.")
        temp = M.fit_temperature([(f, y) for _, _, _, f, _, y in labelled_rows]) if labelled_rows else None
        fair = self.add_model("fair", {"temp": temp or 1.0},
                              note=(f"Temperature {temp} fitted on {len(labelled_rows)} backfilled checkpoints."
                                    if temp else "Untrained starting point (temperature 1.0)."))
        champ = self.add_strategy(fair.id, {}, note="Version 1: conservative TWAP fair-value trader.", origin="seed",
                                  status="champion")
        self.s["champion"] = champ.id
        self.s["champion_history"].append({"sid": champ.id, "model": fair.id, "since": time.time(), "reason": "initial version"})
        self.j.model_event(f"Champion {champ.id} (version 1) goes live in paper mode",
                           f"Model {fair.describe()}; rules: {champ.describe()}.", {"sid": champ.id})
        # a few structurally different starting challengers so learning starts on day one
        seeds = [
            ({"min_edge": 0.06, "t_min": 120}, "Waits for the second half of the window and demands a bigger edge."),
            ({"min_edge": 0.03, "t_min": 60, "exit": "edge_exit"}, "Smaller edge, but sells early at the bid when the edge reverses."),
            ({"t_min": 200, "t_max": 290, "min_prob": 0.7}, "Late, high-confidence entries only (TWAP partly locked in)."),
            ({"vol_ratio_max": 1.5}, "Skips markets where volatility is spiking."),
        ]
        for rules, why in seeds:
            self.add_strategy(fair.id, rules, parent=champ.id, note=why, origin="seed",
                              hypothesis=why)
        self._sync()

    # ------------------------------------------------------------------ live: every checkpoint
    def probs(self, f, start, t):
        out = {}
        in_use = {s.model_id for s in self.active()}
        for m in self.models.values():
            if m.kind != "market" and m.id not in in_use:
                continue
            key = (m.id, start, t)
            if key not in self._p_cache:
                try:
                    self._p_cache[key] = m.predict(f)
                except Exception:  # noqa: BLE001 - a broken model never stops the bot
                    self._p_cache[key] = None
            out[m.id] = self._p_cache[key]
        if len(self._p_cache) > 20000:
            self._p_cache.clear()
        return out

    def _model_in_use(self, mid):
        return any(s.model_id == mid for s in self.active())

    def window(self, start):
        return self.s["windows"].setdefault(str(start), {"preds": {}, "champ": None, "reject": None, "valid_cps": 0})

    def on_checkpoint(self, mkt, start, t, f, book, valid, trade_ok, why_invalid, refetch_book, now):
        w = self.window(start)
        w["slug"] = mkt["slug"] if mkt else MK.slug_for(start)
        P = self.probs(f, start, t)
        if valid:
            w["valid_cps"] += 1
        if t in STD_T and valid:
            for mid, p in P.items():
                if p is not None and mid in self.models and (self._model_in_use(mid) or self.models[mid].kind == "market"):
                    w["preds"].setdefault(mid, {})[str(t)] = round(p, 5)
        champ = self.champion
        events = {"champion_p": P.get(champ.model_id) if champ else None, "probs": P}
        if not mkt or book is None:
            return events

        # --- exits first (real account)
        for pos in [p for p in self.account.positions if p["start"] == start and trade_ok]:
            strat = self.strats.get(pos["strategy"]) or champ
            if strat and strat.should_exit(pos["side"], P.get(strat.model_id), book, mkt):
                fresh = refetch_book() or book
                r = walk_sell(fresh[pos["side"]]["bids"], pos["shares"], 0.01, self.cfg["liquidity_take_pct"],
                              mkt["fee_rate"], mkt["fee_exp"])
                if r["shares"] > 0:
                    self._early_exit(pos, r, mkt, f)

        # --- shadow ledgers (all active strategies, champion included)
        orders = []
        for s in self.active():
            led = self.s["ledgers"][s.id]
            sp = led["open"].get(str(start))
            p = P.get(s.model_id)
            if sp and sp.get("status") == "open":
                if trade_ok and s.should_exit(sp["side"], p, book, mkt):
                    orders.append(("exit", s, sp))
            elif not sp and trade_ok:
                order, reason = s.decide(f, p, book, mkt, SHADOW_BANKROLL, self.cfg)
                if order:
                    orders.append(("buy", s, order))
        fresh = None
        if orders:
            fresh = refetch_book()
            if fresh is None:
                self.j.integrity_event("book", "order book refresh failed; simulated orders cancelled this checkpoint")
                return events
            for kind, s, o in orders:
                led = self.s["ledgers"][s.id]
                if kind == "buy":
                    r = walk_buy(fresh[o["side"]]["asks"], o["budget"], o["limit"], self.cfg["liquidity_take_pct"],
                                 mkt["fee_rate"], mkt["fee_exp"])
                    if r["shares"] >= mkt.get("min_size", 5):
                        led["open"][str(start)] = {"status": "open", "side": o["side"], "shares": r["shares"],
                                                   "cost": r["cost"], "t": t, "prob": o["prob"], "edge": o["edge"]}
                    else:
                        led["open"][str(start)] = {"status": "nofill", "t": t}
                else:
                    r = walk_sell(fresh[o["side"]]["bids"], o["shares"], 0.01, self.cfg["liquidity_take_pct"],
                                  mkt["fee_rate"], mkt["fee_exp"])
                    if r["shares"] > 0:
                        o["exit_proceeds"] = o.get("exit_proceeds", 0) + r["proceeds"]
                        o["shares"] = round(o["shares"] - r["shares"], 2)
                        if o["shares"] <= 0.001:
                            o["status"] = "exited"

        # --- champion: real paper account
        if champ and trade_ok:
            p = P.get(champ.model_id)
            if not self.account.has_position(mkt["slug"]) and not w.get("champ_done"):
                bankroll = self.account.equity()
                order, reason = champ.decide(f, p, book, mkt, bankroll, self.cfg)
                if order:
                    blocked = self._risk_block(now)
                    if blocked:
                        self._reject(start, t, order, f"risk limit: {blocked}", missed=True)
                    else:
                        fb = fresh or refetch_book()
                        if fb is None:
                            self._reject(start, t, order, "order book refresh failed", missed=True)
                        else:
                            self._enter(mkt, start, t, f, p, order, fb, champ, now)
                            w["champ_done"] = True
                elif reason.startswith("edge too small") or reason.startswith("edge disappears"):
                    self._track_best_reject(start, t, f, p, book, mkt, reason)
        elif champ and not trade_ok and t % 30 == 0:
            w["invalid_reason"] = why_invalid
        return events

    def _risk_block(self, now):
        acct, cfg = self.account, self.cfg
        if now < acct.paused_until:
            return "cool-down after a losing streak"
        day = acct.day.get(sgt_day(now))
        if day and day.get("start_equity") and day.get("pnl", 0) < -cfg["daily_loss_limit_pct"] * day["start_equity"]:
            return "daily loss limit reached"
        if acct.loss_streak >= cfg["max_losses_in_a_row"]:
            acct.paused_until = now + 3600
            acct.loss_streak = 0
            self.j.activity(f"{cfg['max_losses_in_a_row']} losses in a row: new entries paused for 1 hour", kind="risk")
            return "cool-down after a losing streak"
        if acct.cash < 2:
            return "not enough paper cash"
        return None

    def _enter(self, mkt, start, t, f, p, order, book, champ, now):
        r = walk_buy(book[order["side"]]["asks"], min(order["budget"], self.account.cash), order["limit"],
                     self.cfg["liquidity_take_pct"], mkt["fee_rate"], mkt["fee_exp"])
        rec = {"ts": now, "slug": mkt["slug"], "t": t, "side": order["side"], "limit": order["limit"],
               "budget": order["budget"], "filled_shares": r["shares"], "avg": r["avg"], "fee": r["fee"],
               "fills": r["fills"], "strategy": champ.id}
        self.account.orders.append(rec)
        if r["shares"] < mkt.get("min_size", 5):
            rec["status"] = "not filled (price moved or book too thin)"
            self._reject(start, t, order, "order could not fill at the limit price", missed=True)
            self.j.activity(f"Order for {order['side'].upper()} not filled: price moved past the limit", kind="order")
            return
        partial = r["cost"] < order["budget"] * 0.97
        rec["status"] = "partial fill" if partial else "filled"
        pos = self.account.open_position(mkt["slug"], start, order["side"], r, {
            "t_entry": t, "prob": round(order["prob"], 4), "edge": round(order["edge"], 4), "model": champ.model_id,
            "strategy": champ.id, "title": mkt["title"], "partial": partial, "budget": order["budget"],
            "feat": {n: F.get(f, n) for n in ("d_open", "tau", "vol_ratio", "vol_bn_300", "vol_cl_300", "flow_30",
                                               "ret_bn_30", "pm_mid_up", "p_fair", "z_fair", "twap_frac")},
            "p_market": F.get(f, "pm_mid_up")})
        w = self.window(start)
        w["champ"] = {"t": t, "p_side": order["prob"], "side": order["side"], "trade": pos["id"]}
        self.j.activity(f"Simulated trade {pos['id']}: bought {r['shares']:.2f} {order['side'].upper()} @ "
                        f"${r['avg']:.3f} (model {order['prob']:.0%}, edge {order['edge']:+.3f}, fee ${r['fee']:.3f})"
                        + (" - partial fill" if partial else ""), kind="trade")

    def _early_exit(self, pos, r, mkt, f):
        left = round(pos["shares"] - r["shares"], 2)
        if left > 0.001:
            # partial exit: realise part, keep the rest open
            frac = r["shares"] / pos["shares"]
            part_cost = pos["cost"] * frac
            pos["cost"] -= part_cost
            pos["shares"] = left
            self.account.cash += r["proceeds"]
            pos.setdefault("partial_exits", []).append({"shares": r["shares"], "proceeds": r["proceeds"], "cost": part_cost})
            self.j.activity(f"{pos['id']}: sold {r['shares']:.2f} early at the bid (book too thin for all)", kind="trade")
            return
        prior = sum(x["proceeds"] - x["cost"] for x in pos.get("partial_exits", []))
        closed = self.account.close_position(pos, r["proceeds"], "early exit at bid", r["avg"], r["fee"])
        closed["pnl"] = round(closed["pnl"] + prior, 5)
        self.j.activity(f"{pos['id']} closed early at ${r['avg']:.3f} (edge gone): P&L {money(closed['pnl'])}", kind="trade")

    def _reject(self, start, t, order, reason, missed=False):
        rec = {"ts": time.time(), "start": start, "t": t, "side": order["side"], "prob": round(order["prob"], 4),
               "edge": round(order["edge"], 4), "reason": reason}
        self.account.rejected.append(rec)
        w = self.window(start)
        if missed:
            w["missed"] = rec

    def _track_best_reject(self, start, t, f, p, book, mkt, reason):
        w = self.window(start)
        side = "up" if p >= 0.5 else "down"
        q = p if side == "up" else 1 - p
        asks = book[side]["asks"]
        if not asks:
            return
        edge = q - asks[0][0] - fee_per_share(asks[0][0], mkt["fee_rate"], mkt["fee_exp"])
        if w.get("reject") is None or edge > w["reject"]["edge"]:
            w["reject"] = {"t": t, "side": side, "prob": round(q, 4), "ask": asks[0][0], "edge": round(edge, 4),
                           "reason": reason}

    # ------------------------------------------------------------------ resolution
    def on_resolution(self, start, outcome, open_p, close_p, src, path, our_twap):
        """outcome 'UP'/'DOWN'. path: [(ts, price)] Chainlink during the window."""
        w = self.window(start)
        if w.get("outcome") and w.get("outcome_src") == "official":
            return
        first = not w.get("outcome")
        corrected = w.get("outcome") and w["outcome"] != outcome
        w.update({"outcome": outcome, "outcome_src": src})
        for k, v in (("open", open_p), ("close", close_p), ("our_twap", our_twap)):
            if v is not None:
                w[k] = v
        y = 1 if outcome == "UP" else 0
        if corrected:
            self.j.integrity_event("resolution", f"window {start}: provisional result corrected to {outcome} by the official resolution")
            self._resettle(start, outcome)
            return
        if not first:
            return
        self.j.activity(f"Market resolved {outcome}: {MK.slug_for(start)}"
                        + (f" (open {open_p:,.2f} -> close {close_p:,.2f})" if open_p and close_p else ""), kind="resolve")
        # real positions
        for pos in [p for p in self.account.positions if p["start"] == start]:
            win = (pos["side"] == "up") == (y == 1)
            prior = sum(x["proceeds"] - x["cost"] for x in pos.get("partial_exits", []))
            closed = self.account.close_position(pos, pos["shares"] if win else 0.0, "settlement", 1.0 if win else 0.0,
                                                 0.0, outcome)
            closed["pnl"] = round(closed["pnl"] + prior, 5)
            self._analyse_trade(closed, path, open_p, w)
        # shadow ledgers
        for sid, led in self.s["ledgers"].items():
            c = self.s["challengers"].get(sid, {})
            if c.get("status") not in ("champion", "shadow", "backup"):
                continue
            if start < c.get("born_window", 0):
                continue
            led["n_windows"] += 1
            sp = led["open"].pop(str(start), None)
            if sp and sp.get("status") in ("open", "exited"):
                win = (sp["side"] == "up") == (y == 1)
                pnl = sp.get("exit_proceeds", 0.0) + (sp["shares"] if win else 0.0) - sp["cost"]
                led["pnl"][str(start)] = round(pnl, 5)
                led["trades"] += 1
                led["wins"] += pnl > 0
                led["recent"] = (led["recent"] + [[start, round(pnl, 3), sp["side"], round(sp["prob"], 3)]])[-60:]
        # prediction evaluation
        champ = self.champion
        if champ:
            p150 = w["preds"].get(champ.model_id, {}).get(str(OFFICIAL_T))
            w["official"] = None if p150 is None else {"p_up": p150, "model": champ.model_id,
                                                        "call": "UP" if p150 >= 0.55 else "DOWN" if p150 <= 0.45 else "NO CALL",
                                                        "correct": None if 0.45 < p150 < 0.55 else (p150 >= 0.55) == (y == 1)}
        # missed opportunities: a challenger traded and won while the champion stayed out
        if champ and not w.get("champ"):
            best = None
            for sid, led in self.s["ledgers"].items():
                pnl = led["pnl"].get(str(start))
                if sid != champ.id and pnl and pnl > 0 and (best is None or pnl > best[1]):
                    best = (sid, pnl)
            if best:
                self.s["missed"] = (self.s["missed"] + [{"start": start, "by": best[0], "pnl": round(best[1], 3),
                                                         "why_champion_passed": (w.get("reject") or {}).get("reason")
                                                         or (w.get("missed") or {}).get("reason") or "no edge by its rules"}])[-300:]
        if w.get("reject"):
            r = w["reject"]
            r["would_have_won"] = (r["side"] == "up") == (y == 1)
        self._update_day(start, w, y)
        self._prune_windows()

    def _resettle(self, start, outcome):
        y = 1 if outcome == "UP" else 0
        for pos in self.account.closed:
            if pos["start"] == start and pos["exit_reason"] == "settlement":
                win = (pos["side"] == "up") == (y == 1)
                new = pos["shares"] if win else 0.0
                self.account.cash += new - pos["proceeds"]
                pos["pnl"] = round(pos["pnl"] + new - pos["proceeds"], 5)
                pos["proceeds"], pos["outcome"] = new, outcome
                pos["corrected"] = True

    def _update_day(self, start, w, y):
        d = self.s["daily_stats"].setdefault(sgt_day(start + 300), {"markets": 0, "valid": 0, "calls": 0, "correct": 0,
                                                                    "brier_sum": 0.0, "brier_n": 0, "mkt_brier_sum": 0.0,
                                                                    "mkt_brier_n": 0, "pairs": 0})
        d["markets"] += 1
        if w.get("valid_cps", 0) > 0:
            d["valid"] += 1
        o = w.get("official")
        if o:
            d["brier_sum"] += (o["p_up"] - y) ** 2
            d["brier_n"] += 1
            if o["call"] != "NO CALL":
                d["calls"] += 1
                d["correct"] += bool(o["correct"])
            mk = next((m for m in self.models.values() if m.kind == "market"), None)
            pm = w["preds"].get(mk.id, {}).get(str(OFFICIAL_T)) if mk else None
            if pm is not None:
                d["mkt_brier_sum"] += (pm - y) ** 2
                d["mkt_brier_n"] += 1
                d["pairs"] += 1

    def _prune_windows(self):
        ws = self.s["windows"]
        if len(ws) > 2600:
            for k in sorted(ws, key=int)[: len(ws) - 2300]:
                ws.pop(k, None)
        cutoff = time.time() - 10 * 86400
        for led in self.s["ledgers"].values():
            if len(led["pnl"]) > 3000:
                led["pnl"] = {k: v for k, v in led["pnl"].items() if int(k) >= cutoff}

    # ------------------------------------------------------------------ step 4: analyse a trade
    def _analyse_trade(self, pos, path, open_p, w):
        win = pos["pnl"] > 0
        feat = pos.get("feat", {})
        tags, facts, ideas = [], [], []
        entry_ts = pos["start"] + pos["t_entry"]
        after = [p for ts, p in path if ts >= entry_ts]
        if open_p and len(after) > 3:
            signs = [1 if p >= open_p else -1 for p in after]
            crossed = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
            first, last = signs[0], signs[-1]
            if first != last:
                tags.append("reversal")
                facts.append(f"The Chainlink price was on the {'UP' if first > 0 else 'DOWN'} side of the price to beat at entry "
                             f"and finished on the other side ({crossed} crossings after entry).")
            elif crossed:
                facts.append(f"The price crossed the price to beat {crossed} times after entry but finished on the same side.")
        if len(after) > 10:
            rets = [math.log(b / a) for a, b in zip(after, after[1:]) if a > 0]
            if rets:
                rv = float(np.std(rets))
                ev = feat.get("vol_cl_300") or feat.get("vol_bn_300")
                if ev:
                    ratio = rv / ev
                    pos["vol_after_ratio"] = ratio
                    if ratio > 1.6:
                        tags.append("vol_spike")
                        facts.append(f"Volatility after entry was {ratio:.1f}x the level the model assumed.")
                    elif ratio < 0.5:
                        facts.append(f"Volatility after entry was only {ratio:.1f}x the assumed level.")
        if pos["t_entry"] >= 240:
            tags.append("late_entry")
        if pos["t_entry"] <= 45:
            tags.append("early_entry")
        if pos["entry_avg"] and pos["entry_avg"] >= 0.85:
            tags.append("expensive_price")
        if pos["entry_avg"] and pos["entry_avg"] <= 0.15:
            tags.append("cheap_price")
        if pos.get("partial"):
            tags.append("thin_book")
        fee_share = pos["fee"] / pos["cost"] if pos["cost"] else 0
        if pos["edge"] < 0.05:
            tags.append("thin_edge")
        vr = feat.get("vol_ratio")
        regime = vol_regime(vr)
        # calibration context: how often does the champion win at this confidence?
        band_lo = math.floor(pos["prob"] * 20) / 20
        same = [c for c in self.account.closed[-500:] if c.get("prob") is not None and band_lo <= c["prob"] < band_lo + 0.05
                and c["id"] != pos["id"]]
        if len(same) >= 15:
            hit = sum(1 for c in same if c["pnl"] > 0) / len(same)
            facts.append(f"At {band_lo:.0%}-{band_lo + 0.05:.0%} confidence the champion has won {hit:.0%} of {len(same)} trades.")
            if hit < band_lo - 0.08:
                tags.append("overconfident")
                ideas.append("Confidence at this level looks too high; test a lower temperature-scaled model or a higher minimum probability.")
        if not win:
            if "reversal" in tags:
                ideas.append("Test requiring a larger edge, or later entries, so a single reversal matters less.")
            if "vol_spike" in tags:
                ideas.append("Test a volatility filter that skips markets when short-term volatility is spiking.")
            if "late_entry" in tags:
                ideas.append("Test an earlier latest-entry time.")
            if "thin_edge" in tags:
                ideas.append("Fees took a large share of this edge; test a higher minimum edge.")
        if win:
            analysis = (f"Correct. The {pos['side'].upper()} side settled at $1. The model priced it at {pos['prob']:.0%} "
                        f"against an all-in cost of ${pos['cost'] / pos['shares']:.3f} per share.")
        else:
            analysis = (f"Incorrect. The {pos['side'].upper()} side settled at $0. The model priced it at {pos['prob']:.0%}; "
                        + ("the price reversed after entry." if "reversal" in tags else "the price never got where it needed to be."))
        entry = {"trade": pos["id"], "slug": pos["slug"], "title": pos.get("title"), "side": pos["side"].upper(),
                 "entry_price": round(pos["entry_avg"], 4), "shares": pos["shares"], "cost": round(pos["cost"], 4),
                 "fee": pos["fee"], "prob": pos["prob"], "edge": pos["edge"], "t_entry": pos["t_entry"],
                 "result": "Correct" if win else "Incorrect", "pnl": pos["pnl"], "exit": pos["exit_reason"],
                 "regime": regime, "tags": tags, "analysis": analysis, "facts": facts,
                 "learning_action": ("Added to the learning dataset; "
                                     + ("ideas queued for testing: " + " ".join(ideas) if ideas
                                        else "no change proposed from a single trade.")),
                 "fee_share": round(fee_share, 4), "model": pos.get("model"), "strategy": pos.get("strategy")}
        pos["tags"] = tags
        self.j.trade(entry)

    # ------------------------------------------------------------------ steps 5-7: experiment, test, improve
    def step(self, now, live_rows_fn):
        """Every few minutes: test challengers, promote, retire, roll back, propose new experiments."""
        self._check_rollback(now)
        champ = self.champion
        if not champ:
            return
        chal = [(sid, c) for sid, c in self.s["challengers"].items() if c["status"] == "shadow"]
        alpha = 0.05 / max(1, len(chal))
        champ_led = self.s["ledgers"][champ.id]
        for sid, c in chal:
            led = self.s["ledgers"][sid]
            n = led["n_windows"]
            if n < 20:
                continue
            starts = [int(k) for k in self.s["windows"] if int(k) >= c["born_window"] and self.s["windows"][k].get("outcome")]
            diffs = [led["pnl"].get(str(s), 0.0) - champ_led["pnl"].get(str(s), 0.0) for s in starts]
            lo, hi, mean = boot_ci(diffs, alpha=alpha)
            c["test"] = {"windows": len(diffs), "trades": led["trades"], "mean_diff": mean, "lo": lo, "hi": hi,
                         "pnl": sum(led["pnl"].get(str(s), 0.0) for s in starts),
                         "champ_pnl": sum(champ_led["pnl"].get(str(s), 0.0) for s in starts), "alpha": alpha,
                         "updated": now}
            ll = self._paired_logloss(self.strats[sid].model_id, champ.model_id, starts)
            c["test"]["logloss"] = ll
            enough = len(diffs) >= self.cfg["min_shadow_markets_for_promotion"] and led["trades"] >= self.cfg["min_shadow_trades_for_promotion"]
            model_ok = ll is None or ll["mean"] >= -0.002 or self.strats[sid].model_id == champ.model_id
            if enough and lo is not None and lo > 0 and c["test"]["pnl"] > 0 and model_ok:
                self._promote(sid, c, now)
                return
            if len(diffs) >= self.cfg["min_shadow_markets_for_promotion"] and hi is not None and hi < 0:
                self._retire(sid, c, "clearly worse than the champion (upper confidence bound below zero)")
            elif n > MAX_SHADOW_WINDOWS:
                self._retire(sid, c, "could not prove an improvement within the test period")
        self._propose(now, live_rows_fn)
        self._sync()

    def _paired_logloss(self, mid_a, mid_b, starts):
        diffs = []
        for s in starts:
            pr = self.s["windows"][str(s)]
            y = 1 if pr.get("outcome") == "UP" else 0
            a, b = pr["preds"].get(mid_a, {}), pr["preds"].get(mid_b, {})
            common = [t for t in a if t in b]
            if common:
                diffs.append(np.mean([logloss(b[t], y) - logloss(a[t], y) for t in common]))
        if len(diffs) < 20:
            return None
        lo, hi, mean = boot_ci(diffs, alpha=0.05)
        return {"n": len(diffs), "mean": mean, "lo": lo, "hi": hi}

    def _promote(self, sid, c, now):
        old = self.champion
        new = self.strats[sid]
        t = c["test"]
        self.s["challengers"][old.id]["status"] = "backup"
        prev_backup = [k for k, v in self.s["challengers"].items() if v["status"] == "backup" and k != old.id]
        for k in prev_backup:
            self.s["challengers"][k]["status"] = "retired"
        c["status"] = "champion"
        c["promoted"] = now
        self.s["champion"] = sid
        hist = self.s["champion_history"]
        if hist:
            hist[-1]["until"] = now
        hist.append({"sid": sid, "model": new.model_id, "since": now, "reason": "won its forward test", "test": t,
                     "version": len(hist) + 1})
        body = (f"{sid} beat {old.id} over {t['windows']} live markets it had never seen: shadow P&L {money(t['pnl'])} vs "
                f"{money(t['champ_pnl'])} (mean {money(t['mean_diff'])}/market, lower bound {money(t['lo'])} at "
                f"{(1 - t['alpha']):.1%} confidence). New rules: {new.describe()}; model: {self.models[new.model_id].describe()}.")
        self.j.model_event(f"Version {len(hist)}: {sid} promoted to champion", body, {"sid": sid, "old": old.id})
        self.j.experiment({"exp": c.get("exp"), "strategy": sid, "stage": "accepted", "title": f"{sid} accepted",
                           "result": body, "evidence": "measured"})
        self.j.activity(f"Model upgrade: {sid} replaces {old.id} as champion (kept {old.id} as backup)", kind="model")

    def _retire(self, sid, c, reason):
        c["status"] = "retired"
        c["retired_reason"] = reason
        t = c.get("test") or {}
        self.j.experiment({"exp": c.get("exp"), "strategy": sid, "stage": "rejected", "title": f"{sid} rejected",
                           "result": f"{reason}. Shadow P&L {money(t.get('pnl', 0))} vs champion {money(t.get('champ_pnl', 0))} "
                                     f"over {t.get('windows', 0)} markets.", "evidence": "measured"})

    def _check_rollback(self, now):
        champ = self.champion
        if not champ:
            return
        hist = self.s["champion_history"]
        since = hist[-1]["since"] if hist else 0
        trades = [c for c in self.account.closed if c.get("strategy") == champ.id and c.get("closed", 0) >= since]
        backup = next((k for k, v in self.s["challengers"].items() if v["status"] == "backup"), None)
        if len(trades) < 40 or not backup:
            return
        pnls = [t["pnl"] for t in trades]
        lo, hi, mean = boot_ci(pnls, alpha=0.10)
        if hi is not None and hi < 0:
            bled = self.s["ledgers"][backup]
            recent = [bled["pnl"].get(str(int(t["start"])), 0.0) for t in trades]
            if sum(recent) > sum(pnls):
                self.s["challengers"][champ.id]["status"] = "retired"
                self.s["challengers"][champ.id]["retired_reason"] = "rolled back: live performance deteriorated"
                self.s["challengers"][backup]["status"] = "champion"
                self.s["champion"] = backup
                hist[-1]["until"] = now
                hist.append({"sid": backup, "model": self.strats[backup].model_id, "since": now,
                             "reason": "rollback to previous validated version", "version": len(hist) + 1})
                body = (f"{champ.id} lost ${-sum(pnls):.2f} over {len(pnls)} trades since promotion "
                        f"(90% upper bound of mean P&L/trade ${hi:+.3f} < 0). The previous version {backup} did better "
                        f"on the same markets ({money(sum(recent))}), so it is back in charge.")
                self.j.model_event(f"Rollback: {backup} restored as champion", body, {"from": champ.id, "to": backup})
                self.j.activity(f"Performance deteriorated: rolled back from {champ.id} to {backup}", kind="model")

    # ------------------------------------------------------------------ proposals
    def _propose(self, now, live_rows_fn):
        last = self.s["last"].get("propose", 0)
        if now - last < 1800:
            return
        self.s["last"]["propose"] = now
        n_active = sum(1 for c in self.s["challengers"].values() if c["status"] == "shadow")
        room = self.cfg["max_challengers"] - n_active
        if room <= 0:
            return
        champ = self.champion
        ideas = self._ideas_from_errors()
        made = 0
        rows = None
        for key, val, why in ideas:
            if made >= room or made >= 2:
                break
            if self._already_tested(champ, key, val):
                continue
            rules = dict(champ.rules) | {key: val}
            rows = rows if rows is not None else live_rows_fn(3)
            self._launch(champ, champ.model_id, rules, why, "error analysis", rows, f"{key} {champ.rules.get(key)} -> {val}")
            made += 1
        while made < min(room, 2):
            cand, changed = mutate(champ, "tmp", rng=self.rng)
            if not changed or self._already_tested(champ, None, None, cand.rules):
                made += 1
                continue
            rows = rows if rows is not None else live_rows_fn(3)
            self._launch(champ, champ.model_id, cand.rules, "Exploration: try a nearby variation of the champion's rules.",
                         "exploration", rows, "; ".join(changed))
            made += 1

    def _already_tested(self, champ, key, val, rules=None):
        target = rules or (dict(champ.rules) | {key: val})
        for sid, st in self.strats.items():
            if st.model_id == champ.model_id and st.rules == target and sid != champ.id:
                return True
        return target == champ.rules

    def _ideas_from_errors(self):
        """Compare tags on recent losing vs winning champion trades; over-represented tags become hypotheses."""
        recent = [e for e in self.j.trades[-150:] if e.get("strategy") == self.s.get("champion")]
        if len(recent) < 12:
            return []
        losses = [e for e in recent if e["pnl"] <= 0]
        wins = [e for e in recent if e["pnl"] > 0]
        champ = self.champion.rules
        mapping = {
            "vol_spike": ("vol_ratio_max", 1.5 if champ.get("vol_ratio_max") in (None, 2.0, 3.0) else 2.0,
                          "Losses cluster in volatility spikes: hypothesis - skipping spiking markets improves P&L."),
            "late_entry": ("t_max", max(150, champ["t_max"] - 40), "Late entries lose more often: hypothesis - stop entering earlier."),
            "early_entry": ("t_min", min(200, champ["t_min"] + 40), "Early entries lose more often: hypothesis - wait longer before entering."),
            "reversal": ("min_edge", round(champ["min_edge"] + 0.02, 2), "Reversals after entry cause most losses: hypothesis - demand a larger edge."),
            "thin_edge": ("min_edge", round(champ["min_edge"] + 0.02, 2), "Thin-edge trades lose to fees: hypothesis - raise the minimum edge."),
            "overconfident": ("min_prob", round(min(champ["min_prob"] + 0.05, 0.85), 2), "Confidence is too high: hypothesis - require a higher model probability."),
            "expensive_price": ("max_price", round(max(champ["max_price"] - 0.05, 0.7), 2), "Expensive entries lose too much when wrong: hypothesis - cap the entry price lower."),
            "cheap_price": ("min_price", round(min(champ["min_price"] + 0.05, 0.3), 2), "Cheap long shots rarely pay: hypothesis - raise the minimum price."),
            "thin_book": ("min_depth_usd", min(champ["min_depth_usd"] * 2, 200), "Thin books hurt fills: hypothesis - require more depth."),
        }
        out = []
        for tag, (key, val, why) in mapping.items():
            lf = sum(tag in e["tags"] for e in losses) / max(len(losses), 1)
            wf = sum(tag in e["tags"] for e in wins) / max(len(wins), 1)
            if lf >= 0.25 and lf > wf + 0.15:
                out.append((lf - wf, key, val, f"{why} (measured: tag on {lf:.0%} of losses vs {wf:.0%} of wins, "
                                                 f"n={len(losses)}/{len(wins)})"))
        out.sort(reverse=True)
        return [(k, v, why) for _, k, v, why in out]

    def _launch(self, champ, model_id, rules, why, origin, rows, change):
        exp_id = f"E{self._nid('exp'):03d}"
        cand = Strategy("candidate", model_id, rules)
        # Stage 1: historical replay on recorded markets, against the champion replayed the same way
        rep = None
        if rows:
            res, wins = replay([champ, cand], self.models, rows, self.cfg)
            if len(wins) >= 60:
                cp, xp = res[champ.id], res["candidate"]
                diffs = [xp["pnl"].get(s, 0.0) - cp["pnl"].get(s, 0.0) for s in wins]
                lo, hi, mean = boot_ci(diffs, alpha=0.1)
                rep = {"windows": len(wins), "cand_pnl": sum(xp["pnl"].values()), "champ_pnl": sum(cp["pnl"].values()),
                       "cand_trades": xp["trades"], "champ_trades": cp["trades"], "mean_diff": mean, "hi": hi}
                if hi is not None and hi < 0:
                    self.j.experiment({"exp": exp_id, "stage": "rejected in replay", "title": f"{exp_id}: {change}",
                                       "goal": why, "change": change,
                                       "test": f"Replayed on {len(wins)} recorded markets (1 extra tick of slippage on every fill).",
                                       "result": f"Rejected before going live: replay P&L {money(rep['cand_pnl'])} vs champion "
                                                 f"{money(rep['champ_pnl'])} (clearly worse).", "evidence": "measured",
                                       "origin": origin})
                    self.strats.pop("candidate", None)
                    return None
        st = self.add_strategy(model_id, rules, parent=champ.id, note=change, origin=origin, hypothesis=why, replay=rep,
                               exp_id=exp_id)
        self.j.experiment({"exp": exp_id, "strategy": st.id, "stage": "testing", "title": f"{exp_id}: {change}",
                           "goal": why, "why": why, "change": change, "origin": origin,
                           "test": ((f"Replay on {rep['windows']} recorded markets: {money(rep['cand_pnl'])} vs champion "
                                     f"{money(rep['champ_pnl'])}. " if rep else "Not enough recorded markets for a replay yet. ")
                                    + f"Now running in shadow on live markets; needs {self.cfg['min_shadow_markets_for_promotion']} "
                                      f"markets and {self.cfg['min_shadow_trades_for_promotion']} trades, and a significant win."),
                           "evidence": "hypothesis"})
        self.j.activity(f"Testing potential improvement {exp_id} as {st.id}: {change}", kind="experiment")
        return st

    # ------------------------------------------------------------------ model research (retraining)
    def research(self, now, all_rows):
        """Train candidate models on older data, judge them on the most recent unseen day."""
        if len(all_rows) < 3000:
            self.j.activity(f"Model research skipped: {len(all_rows)} labelled checkpoints (need 3,000)", kind="research")
            return
        champ = self.champion
        starts = sorted({r[1] for r in all_rows})
        cut = starts[int(len(starts) * 0.8)] if len(starts) < 288 * 5 else max(starts) - 86400
        train = [(f, y) for _, s, _, f, _, y in all_rows if s < cut]
        hold = [r for r in all_rows if r[1] >= cut]
        live_train = [(f, y) for f, y in train if F.get(f, "pm_mid_up") is not None]
        cands = []
        temp = M.fit_temperature(train)
        if temp:
            cands.append(("fair", {"temp": temp}, f"fair value re-fitted (temperature {temp})"))
        for fset in ("core", "flow", "regime"):
            for l2 in (3.0, 30.0):
                p = M.fit_logit(train, fset, l2=l2)
                if p:
                    cands.append(("logit", p, f"logistic [{fset}] L2={l2}"))
        if len(live_train) >= 3000:
            p = M.fit_blend(live_train)
            if p:
                cands.append(("blend", p, "fair value + market blend"))
            p = M.fit_logit(live_train, "core", l2=10.0, use_market=True)
            if p:
                cands.append(("logit", p, "logistic [core + market]"))
        g = M.fit_gbm(train, "all")
        if g:
            cands.append(("gbm", g, "LightGBM [all features]"))
        champ_model = self.models[champ.model_id]
        by_win = collections.defaultdict(list)
        for _, s, t, f, _, y in hold:
            if t in STD_T:
                by_win[s].append((f, y))
        alpha = 0.05 / max(1, len(cands))
        results = []
        for kind, params, note in cands:
            m = M.Model("cand", kind, params)
            diffs, ll_c, ll_x = [], [], []
            for s, cps in by_win.items():
                a = [(m.predict(f), champ_model.predict(f), y) for f, y in cps]
                a = [(x, c, y) for x, c, y in a if x is not None and c is not None]
                if a:
                    lx = np.mean([logloss(x, y) for x, _, y in a])
                    lc = np.mean([logloss(c, y) for _, c, y in a])
                    diffs.append(lc - lx)
                    ll_x.append(lx)
                    ll_c.append(lc)
            if len(diffs) < 50:
                continue
            lo, hi, mean = boot_ci(diffs, alpha=alpha)
            results.append((mean, lo, kind, params, note, float(np.mean(ll_x)), float(np.mean(ll_c)), len(diffs)))
        results.sort(key=lambda r: r[0], reverse=True)
        summary = [{"model": r[4], "holdout_logloss": round(r[5], 4), "champion_logloss": round(r[6], 4),
                    "improvement": round(r[0], 4), "lower_bound": None if r[1] is None else round(r[1], 4), "markets": r[7]}
                   for r in results]
        self.s["last"]["research_result"] = {"ts": now, "train_rows": len(train), "holdout_markets": len(by_win),
                                             "candidates": summary}
        winners = [r for r in results if r[1] is not None and r[1] > 0][:2]
        if not winners:
            self.j.experiment({"exp": f"E{self._nid('exp'):03d}", "stage": "rejected in validation",
                               "title": "Model research: no candidate beat the champion's model",
                               "goal": "Find a more accurate probability model.",
                               "change": f"Trained {len(results)} candidate models on {len(train):,} checkpoints.",
                               "test": f"Scored on the most recent {len(by_win)} unseen markets (log loss, paired, bootstrap).",
                               "result": "Best: " + (f"{results[0][4]} ({results[0][0]:+.4f} log loss/market, not significant)"
                                                     if results else "none could be evaluated."),
                               "evidence": "measured", "origin": "model research"})
            return
        rows_after = [r for r in all_rows if r[1] >= cut]
        room = self.cfg["max_challengers"] - sum(1 for c in self.s["challengers"].values() if c["status"] == "shadow")
        if room <= 0:
            # make space: retire the oldest exploration challenger, new evidence-backed models take priority
            old = sorted(((c["born"], sid) for sid, c in self.s["challengers"].items()
                          if c["status"] == "shadow" and c.get("origin") == "exploration"))
            if old:
                self._retire(old[0][1], self.s["challengers"][old[0][1]], "made room for a validated model candidate")
                room = 1
        for mean, lo, kind, params, note, lx, lc, n in winners[:max(room, 0)]:
            m = self.add_model(kind, params, parent=champ.model_id, note=f"{note}; holdout log loss {lx:.4f} vs {lc:.4f} "
                                                                         f"on {n} unseen markets", trained_until=cut)
            why = (f"Measured on {n} unseen markets: log loss {lx:.4f} vs champion model {lc:.4f} "
                   f"(improvement {mean:+.4f}, lower bound {lo:+.4f}). Hypothesis: better probabilities -> better trades.")
            self._launch(champ, m.id, dict(champ.rules), why, "model research", rows_after, f"new model {m.id}: {note}")
        self._sync()

    # ------------------------------------------------------------------ persistence
    def _sync(self):
        self.s["models"] = {k: v.to_dict() for k, v in self.models.items()}
        self.s["strategies"] = {k: v.to_dict() for k, v in self.strats.items() if k != "candidate"}
        self.s["account"] = self.account.to_dict()
        self._gc_models()

    def _gc_models(self):
        """Drop retired gbm blobs that nothing uses (they are large); keep their history entries."""
        used = {s.model_id for sid, s in self.strats.items() if self.s["challengers"].get(sid, {}).get("status") in
                ("champion", "shadow", "backup")}
        for mid, m in list(self.models.items()):
            if m.kind == "gbm" and mid not in used and m.params.get("model_b64"):
                m.params = {k: v for k, v in m.params.items() if k != "model_b64"} | {"archived": True}
                self.s["models"][mid] = m.to_dict()

    def save(self):
        self._sync()
        return self.s
