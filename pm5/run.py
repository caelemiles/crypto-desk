#!/usr/bin/env python3
"""Central controller. Coordinates the seven components in one continuous loop:

  Market Intelligence (market.py)  ->  Data Intelligence (feeds.py)  ->  Prediction Engine (features/models)
  -> Strategy Development (strategy/learning)  ->  Trading Engine (paper.py)  ->  Learning Engine (learning.py)
  -> Learning Journal (journal.py)

It runs for PM5_MINUTES (default ~5h40m, a GitHub Actions job lasts up to 6h), saves everything, and exits;
the next scheduled job restores the memory and carries on.

PAPER TRADING ONLY. No keys, no wallet, no order endpoints.
"""
import collections
import json
import math
import os
import signal
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backfill  # noqa: E402
import patterns  # noqa: E402
import config as C  # noqa: E402
import features as F  # noqa: E402
import market as MK  # noqa: E402
import report  # noqa: E402
from db import DB  # noqa: E402
from feeds import ChainlinkStream, Health, Spot  # noqa: E402
from journal import Journal  # noqa: E402
from learning import OFFICIAL_T, Brain  # noqa: E402
from paper import Account  # noqa: E402
from publish import Publisher  # noqa: E402
from util import FetchError, sgt_day  # noqa: E402

assert C.PAPER_ONLY, "This bot only paper-trades."

EPOCH = "launch-1"  # changing this starts the AI from a clean memory (the historical database is kept)

STOP = False


class Tee:
    """Keeps the last lines of output so they can be published as log.txt (for remote diagnosis)."""

    def __init__(self, stream, n=500):
        self.stream, self.lines, self.buf = stream, collections.deque(maxlen=n), ""

    def write(self, s):
        self.stream.write(s)
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.lines.append(time.strftime("%H:%M:%S ", time.gmtime()) + line)
        return len(s)

    def flush(self):
        self.stream.flush()


LOG = Tee(sys.stdout)


def _stop(*_):
    global STOP
    STOP = True


class Controller:
    def __init__(self, minutes):
        self.t0 = time.time()
        self.deadline = self.t0 + minutes * 60
        self.settings, self.settings_src = C.load_settings()
        self.pub = Publisher(C.PUBLISH_DIR, os.environ.get("GITHUB_REPOSITORY"), os.environ.get("PM5_TOKEN"), branch=C.A["branch"])
        self.pub.init()
        state = self.pub.read("state.json", {}) or {}
        if state and state.get("epoch") != EPOCH:
            print(f"memory epoch {state.get('epoch')} != {EPOCH}: starting from a clean memory", flush=True)
            state = {}
        state["epoch"] = EPOCH
        self.db = DB(C.DB_PATH)
        self.j = Journal(state.get("journal"))
        self.h = Health()
        self._apply_reset(state)
        self.brain = Brain(state, self.db, self.j, self.settings)
        self.cl = ChainlinkStream(self.h)
        self.spot = Spot(self.h)
        self.life = MK.Lifecycle(self.h, self.db, self.j)
        self.done_cp = set()
        self.flags = collections.defaultdict(set)   # per-window activity lines already written
        self.last_book = None
        self.last_book_ts = 0
        self.marks = {}
        self.live = {}
        self.oi_hist = collections.deque(maxlen=60)
        self.t = collections.defaultdict(float)     # timers
        self.loop_n = 0
        self.loop_errors = 0
        self.run_id = os.environ.get("GITHUB_RUN_ID", "local")
        self.pending = set(int(x) for x in state.get("pending", []))
        self.pub.write("live.json", {"generated": time.time(), "paper_only": True, "starting": True,
                                     "run": {"id": self.run_id, "started": self.t0, "ends": self.deadline},
                                     "activity": self.j.feed[-40:]})
        self.pub.push("pm5 starting")

    # ------------------------------------------------------------------ setup
    def _apply_reset(self, state):
        token = self.settings["reset_token"]
        if state.get("account") and state.get("reset_token") != token:
            old = Account(state["account"])
            self.j.activity(f"Paper account reset (reset_token changed). Previous equity ${old.cash:,.2f}; "
                            f"learning data and models are kept.", kind="risk")
            state["account"] = None
        if not state.get("account"):
            state["account"] = Account(None, self.settings["starting_balance"]).to_dict()
        state["reset_token"] = token

    def start(self):
        self.j.activity(f"Controller started (run {self.run_id}); settings from {self.settings_src}; paper trading only",
                        kind="system")
        self.cl.start()
        now = time.time()
        try:
            old = backfill.oldest_history(self.db)
            if old is None:
                backfill.run(self.db, self.j, now - 3 * 86400, now, budget_s=480)
            else:
                backfill.run(self.db, self.j, now - 8 * 3600, now, budget_s=180, label="Filling the gap of")
                if old > now - backfill.MAX_DAYS * 86400:
                    backfill.run(self.db, self.j, old - 86400, old, budget_s=240, label="Extending history with")
        except Exception as e:  # noqa: BLE001
            self.j.integrity_event("backfill", f"backfill failed: {type(e).__name__}: {e}")
            traceback.print_exc()
        if not self.brain.s.get("champion"):
            rows = self.db.labelled_checkpoints(F.SCHEMA)
            self.brain.bootstrap(rows)
        if time.time() - self.brain.s["last"].get("research", 0) > 3 * 3600:
            self.research()
        # warm up spot history so volatility features are ready at once
        self.spot.poll(time.time(), backlog_s=960)

    # ------------------------------------------------------------------ main loop
    def run(self):
        self.start()
        while not STOP and time.time() < self.deadline:
            t_loop = time.time()
            try:
                self.tick(t_loop)
            except Exception as e:  # noqa: BLE001 - never die on one bad tick
                self.loop_errors += 1
                self.j.integrity_event("loop", f"{type(e).__name__}: {e}")
                traceback.print_exc()
                time.sleep(2)
            self.loop_n += 1
            # aim for one tick per second, slightly after the second boundary
            time.sleep(max(0.05, 1.0 - (time.time() - t_loop)))
        self.shutdown()

    def tick(self, now):
        self.spot.poll(now)
        self._store_rows()
        S = MK.window_start(now)
        t = now - S
        m = self.life.ensure(S, now)
        if t > 200:
            self.life.ensure(S + C.WINDOW, now)
        if m:
            self.pending.add(S)
            self.life.refresh_meta(S, now)
            self.life.get_open(S, now, self.cl)
            self._book(m, now)
        cp = int(t // C.CHECK_EVERY * C.CHECK_EVERY)
        if 5 <= cp <= 295 and (S, cp) not in self.done_cp:
            self.done_cp.add((S, cp))
            self.checkpoint(m, S, cp, now)
        self._resolutions(now)
        self._periodic(now)

    def _store_rows(self):
        rows = self.cl.take_rows()
        if rows:
            self.db.many("INSERT OR IGNORE INTO chainlink VALUES(?,?)", rows)
        b, e, x = self.spot.take_rows()
        self.db.many("INSERT OR IGNORE INTO btc1s VALUES(?,?,?,?,?,?,?)", b)
        self.db.many("INSERT OR IGNORE INTO eth1s VALUES(?,?)", e)
        self.db.many("INSERT OR IGNORE INTO ext VALUES(?,?,?)", x)
        for ts, kind, v in x:
            if kind == "oi_usd":
                self.oi_hist.append((ts, v))

    def _book(self, m, now):
        try:
            b = MK.fetch_books(m)
            self.last_book, self.last_book_ts = b, now
            self.h.ok("clob")
            ub, ubs, ua, uas, umid = MK.top(b, "up")
            db_, dbs, da, das, dmid = MK.top(b, "down")
            self.marks[(m["slug"], "up")] = ub
            self.marks[(m["slug"], "down")] = db_
            if now - self.t["snap"] >= 2:
                self.t["snap"] = now
                self.db.x("INSERT INTO pm_snap VALUES(?,?,?,?,?,?,?,?,?,?)", (now, m["slug"], ub, ua, db_, da, ubs, uas, dbs, das))
            if umid is not None and dmid is not None and abs(umid + dmid - 1) > 0.08:
                self.j.integrity_event("book", f"UP and DOWN mids add to {umid + dmid:.2f} (expected ~1.00)")
        except (FetchError, ValueError, KeyError, TypeError) as e:
            self.h.bad("clob", e)

    def _refetch(self, m):
        cache = {}

        def go():
            if "b" not in cache:
                try:
                    cache["b"] = MK.fetch_books(m)
                except (FetchError, ValueError, KeyError, TypeError) as e:
                    self.h.bad("clob", e)
                    cache["b"] = None
            return cache["b"]
        return go

    # ------------------------------------------------------------------ one decision checkpoint
    def checkpoint(self, m, S, cp, now):
        E = S + C.WINDOW
        open_ref = self.life.open_price.get(S)
        cl_series = dict(self.cl.series(int(now) - 1000))
        coin = self.spot.coinbase[1] if self.spot.coinbase and now - self.spot.coinbase[0] < 20 else None
        ext = {k: v for k, v in self.spot.okx.items()}
        if len(self.oi_hist) >= 2:
            old = [v for ts, v in self.oi_hist if ts <= now - 900]
            if old:
                ext["oi_chg"] = self.oi_hist[-1][1] / old[-1] - 1
        bookinfo = {}
        if self.spot.book and now - self.spot.book[0] < 10:
            bookinfo["binance"] = (self.spot.book[1], self.spot.book[2])
        book = self.last_book if (m and now - self.last_book_ts < 3) else None
        if book:
            ub, ubs, ua, uas, umid = MK.top(book, "up")
            up_b = sum(q for _, q in book["up"]["bids"][:3])
            up_a = sum(q for _, q in book["up"]["asks"][:3])
            bookinfo["pm"] = {"mid_up": umid, "spread_up": (ua - ub) if umid is not None else None,
                              "imb": (up_b - up_a) / (up_b + up_a) if up_b + up_a > 0 else None}
        prev = self.db.one("SELECT outcome FROM markets WHERE start=? AND outcome IS NOT NULL", (S - C.WINDOW,))
        prev_o = 0 if not prev else (1 if prev[0] == "UP" else -1)
        twap = m["twap_secs"] if m else C.TWAP_SECONDS
        ts = S + cp
        f = F.build(ts, S, E, twap, open_ref, cl_series, self.spot.btc, self.spot.eth, ext, bookinfo, prev_o, coin)
        # data quality gate
        problems = []
        if not m:
            problems.append("market not found")
        if open_ref is None:
            problems.append("price to beat not published yet")
        if self.h.age("chainlink") > 5:
            problems.append(f"Chainlink feed stale ({self.h.age('chainlink'):.0f}s)")
        if self.h.age("binance_btc") > 8:
            problems.append("Binance feed stale")
        if book is None:
            problems.append("no fresh order book")
        if F.get(f, "p_fair") is None:
            problems.append("fair value unavailable")
        valid = not problems
        tradable, why = self.life.tradable(m, now)
        trade_ok = valid and tradable
        if valid or (open_ref and F.get(f, "p_fair") is not None):
            self.db.put_checkpoint(m["slug"] if m else MK.slug_for(S), cp, ts, F.SCHEMA, f,
                                   MK.compact_book(book) if book else None, valid)
        if problems and cp % 60 == 0:
            self.j.integrity_event("data", f"{MK.slug_for(S)} t={cp}s: " + "; ".join(problems))
        ev = self.brain.on_checkpoint(m, S, cp, f, book, valid, trade_ok, "; ".join(problems) or why,
                                      self._refetch(m) if m else (lambda: None), now)
        self._narrate(m, S, cp, f, ev, problems)
        self._live_block(m, S, cp, f, ev, book, open_ref, problems, why if not trade_ok else "")

    def _narrate(self, m, S, cp, f, ev, problems):
        fl = self.flags[S]
        if not m:
            return
        if "collect" not in fl:
            fl.add("collect")
            self.j.activity("Collecting market data (order book, Chainlink, Binance, futures)", kind="data")
        if cp >= 60 and "imb" not in fl:
            fl.add("imb")
            imb = F.get(f, "pm_imb")
            bimb = F.get(f, "book_imb")
            self.j.activity("Analysing order-book imbalance: Polymarket "
                            + (f"{imb:+.2f}" if imb is not None else "n/a") + ", Binance "
                            + (f"{bimb:+.2f}" if bimb is not None else "n/a"), kind="analysis")
        if cp >= OFFICIAL_T and "pred" not in fl:
            fl.add("pred")
            p = ev.get("champion_p")
            mid = F.get(f, "pm_mid_up")
            if p is not None:
                call = "UP" if p >= 0.55 else "DOWN" if p <= 0.45 else "NO CALL"
                self.j.activity(f"Generating prediction: {call} (P(up) {p:.0%}"
                                + (f", market {mid:.0%}" if mid is not None else "") + ")", kind="prediction")
            elif problems:
                self.j.activity("Prediction skipped: " + "; ".join(problems), kind="prediction")
        if cp == 295 and "end" not in fl:
            fl.add("end")
            w = self.brain.window(S)
            if not w.get("champ"):
                r = w.get("missed") or w.get("reject")
                self.j.activity("Opportunity rejected: " + (r["reason"] if r else "no side had enough edge after fees"),
                                kind="decision")
        if len(self.flags) > 50:
            for k in sorted(self.flags)[:-20]:
                self.flags.pop(k, None)
                self.done_cp = {x for x in self.done_cp if x[0] >= k}

    def _live_block(self, m, S, cp, f, ev, book, open_ref, problems, why):
        champ = self.brain.champion
        cl_t, cl_p = self.cl.last()
        mk = {}
        if book:
            for side in ("up", "down"):
                bb, bs, ba, az, mid = MK.top(book, side)
                mk[side] = {"bid": bb, "ask": ba, "mid": mid, "bid_size": bs, "ask_size": az,
                            "depth_bid": [[p, q] for p, q in book[side]["bids"][:8]],
                            "depth_ask": [[p, q] for p, q in book[side]["asks"][:8]]}
        w = self.brain.window(S)
        models = []
        for mid_, p in ev.get("probs", {}).items():
            mo = self.brain.models.get(mid_)
            if mo and p is not None:
                models.append({"id": mid_, "desc": mo.describe(), "p_up": p, "kind": mo.kind})
        self.live = {
            "ts": time.time(), "market": {"title": m["title"] if m else None, "slug": MK.slug_for(S), "start": S,
                                          "end": S + C.WINDOW, "t": cp, "open_ref": open_ref, "cl_price": cl_p,
                                          "cl_ts": cl_t, "d_bp": (F.get(f, "d_open") or 0) * 1e4 if F.get(f, "d_open") is not None else None,
                                          "twap_frac": F.get(f, "twap_frac"), "z": F.get(f, "z_fair"),
                                          "vol_ratio": F.get(f, "vol_ratio"), "book": mk,
                                          "liquidity": m.get("liquidity") if m else None, "volume": m.get("volume") if m else None,
                                          "fee_rate": m.get("fee_rate") if m else None,
                                          "resolution": (m.get("description") or "")[:400] if m else None,
                                          "problems": problems, "not_tradable": why},
            "champion": {"id": champ.id if champ else None, "model": champ.model_id if champ else None,
                         "p_up": ev.get("champion_p"), "decision": w.get("champ"), "best_reject": w.get("reject"),
                         "missed": w.get("missed")},
            "models": models,
            "features": {n: F.get(f, n) for n in F.NAMES},
        }

    # ------------------------------------------------------------------ resolution
    def _resolutions(self, now):
        """Settle on Polymarket's official resolution (usually ~80s after the close). If it is late, settle
        provisionally on the AI's own Chainlink TWAP comparison and correct it when the official result lands."""
        if now - self.t["resolve"] < 10:
            return
        self.t["resolve"] = now
        for S in sorted(self.pending):
            E = S + C.WINDOW
            if now < E + 20:
                continue
            w = self.brain.window(S)
            if w.get("outcome_src") == "official":
                self.pending.discard(S)
                continue
            if now > E + 6 * 3600:
                self.pending.discard(S)
                self.j.integrity_event("resolution", f"{MK.slug_for(S)}: no official result after 6 hours")
                continue
            path = [(ts, p) for ts, p in self.cl.series(S - 60) if ts <= E]
            twap_pts = [p for ts, p in path if E - 60 < ts <= E]
            our_twap = sum(twap_pts) / len(twap_pts) if len(twap_pts) >= 40 else None
            open_ref = self.life.open_price.get(S)
            our = None if (our_twap is None or open_ref is None) else ("UP" if our_twap >= open_ref else "DOWN")
            w["our_call"] = our
            try:
                off = MK.official_outcome(S)
                self.h.ok("resolution")
            except FetchError as e:
                self.h.bad("resolution", e)
                off = None
            if off:
                self.db.set_result(MK.slug_for(S), open_ref, our_twap, off, "official")
                self.db.set_official(MK.slug_for(S), off)
                if our and our != off:
                    self.j.integrity_event("resolution", f"{MK.slug_for(S)}: own TWAP said {our}, official {off} "
                                                         f"(margin {(our_twap / open_ref - 1) * 1e4:+.2f} bp)")
                self.brain.on_resolution(S, off, open_ref, our_twap, "official", path, our_twap)
                self.pending.discard(S)
                self._after_resolution(S, w)
                try:
                    patterns.explain(self.brain, S, off, open_ref, our_twap, path, self._window_stats(S))
                    self.j.activity(f"Explained why the market went {off} (see Market explanations)", kind="learning")
                except Exception as e:  # noqa: BLE001
                    self.j.integrity_event("explain", f"{type(e).__name__}: {e}")
            elif our and not w.get("outcome") and now > E + 900:
                self.db.set_result(MK.slug_for(S), open_ref, our_twap, our, "own TWAP (provisional)")
                self.brain.on_resolution(S, our, open_ref, our_twap, "own TWAP (provisional)", path, our_twap)
                self.j.activity(f"Official result late for {MK.slug_for(S)}: settled provisionally on the AI's own TWAP",
                                kind="resolve")
                self._after_resolution(S, w)

    def _window_stats(self, S):
        """Order flow, the other coin, volatility and liquidations during one window (for its explanation)."""
        E = S + C.WINDOW
        bn = self.spot.btc
        v = tb = 0.0
        rets, pre = [], []
        for s in range(S, E):
            k, k0 = bn.get(s), bn.get(s - 1)
            if k:
                v += k[4]
                tb += k[5]
                if k0 and k0[3] > 0:
                    rets.append(math.log(k[3] / k0[3]))
        for s in range(S - 900, S):
            k, k0 = bn.get(s), bn.get(s - 1)
            if k and k0 and k0[3] > 0:
                pre.append(math.log(k[3] / k0[3]))
        out = {"flow": (tb / v - 0.5) if v > 0 else None}
        if len(rets) > 30 and len(pre) > 100:
            sd = lambda xs: (sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5  # noqa: E731
            if sd(pre) > 0:
                out["vol_ratio"] = sd(rets) / sd(pre)
        o0 = next((self.spot.eth.get(s) for s in range(S, S - 5, -1) if self.spot.eth.get(s)), None)
        o1 = next((self.spot.eth.get(s) for s in range(E, E - 5, -1) if self.spot.eth.get(s)), None)
        if o0 and o1:
            out["other_bp"] = (o1 / o0 - 1) * 1e4
        liq = {k: v for k, v in self.db.q("SELECT kind, value FROM ext WHERE ts>=? AND ts<=? AND kind IN "
                                          "('liq_long_5m','liq_short_5m') ORDER BY ts", (E - 30, E + 90))}
        if liq:
            tot = liq.get("liq_long_5m", 0) + liq.get("liq_short_5m", 0)
            if tot > 0:
                out["liq_usd"] = tot
                out["liq_short_share"] = liq.get("liq_short_5m", 0) / tot
        return out

    def _after_resolution(self, S, w):
        self.j.activity("Comparing prediction with actual outcome", kind="learning")
        o = w.get("official")
        if o and o["call"] != "NO CALL":
            self.j.activity(f"Prediction {o['call']} at {OFFICIAL_T}s was {'correct' if o['correct'] else 'incorrect'}",
                            kind="learning")
        self.j.activity("Updating learning dataset", kind="learning")
        for pos in [p for p in self.brain.account.closed[-3:] if p.get("start") == S]:
            self.db.x("INSERT OR REPLACE INTO trades VALUES(?,?,?)", (pos["id"], pos["slug"], json.dumps(pos)))

    # ------------------------------------------------------------------ periodic jobs
    def _periodic(self, now):
        if now - self.t["commit"] >= 5:
            self.t["commit"] = now
            self.db.commit()
            self.brain.account.track(now, sgt_day(now), self.marks)
        # heavy jobs run early in a window, before most strategies may enter, so they never block an exit
        t_in = now % C.WINDOW
        early = 7 <= t_in <= 25
        if now - self.t["step"] >= 300 and early:
            self.t["step"] = now
            self.brain.step(now, self.live_rows)
        if now - self.brain.s["last"].get("research", 0) >= 3 * 3600 and early:
            self.research()
        if now - self.brain.s["last"].get("mine", 0) >= 3600 and early:
            self.brain.s["last"]["mine"] = now
            try:
                res = patterns.mine(self.brain, self.db, now)
                n_trad = sum(1 for x in res if x["status"] == "tradable")
                self.j.activity(f"Pattern search: tested {len(res)} signal/time combinations, {n_trad} beat the market "
                                f"price by enough to trade", kind="research")
                patterns.launch_patterns(self.brain, self.live_rows)
            except Exception as e:  # noqa: BLE001
                self.j.integrity_event("patterns", f"{type(e).__name__}: {e}")
                traceback.print_exc()
        day = sgt_day(now)
        last_day = self.brain.s["last"].get("report_day")
        if last_day is None:
            self.brain.s["last"]["report_day"] = day
        elif day != last_day:
            from datetime import datetime
            from util import SGT
            d0 = datetime.strptime(last_day, "%Y-%m-%d").replace(tzinfo=SGT).timestamp()
            report.daily_report(self.brain, last_day, d0, d0 + 86400)
            self.brain.s["last"]["report_day"] = day
            self.db.prune(now)
        if now - self.brain.s["last"].get("discoveries", 0) >= 7200:
            self.brain.s["last"]["discoveries"] = now
            report.discoveries(self.brain, self.db)
        if now - self.t["settings"] >= 600:
            self.t["settings"] = now
            s, src = C.load_settings()
            if s != self.settings:
                changed = [k for k in s if s[k] != self.settings.get(k)]
                self.settings.clear()
                self.settings.update(s)
                self.j.activity(f"Settings updated: {', '.join(changed)}", kind="system")
                if "reset_token" in changed:
                    self.brain.account = Account(None, s["starting_balance"])
                    self.brain.s["reset_token"] = s["reset_token"]
                    self.j.activity(f"Paper account reset to ${s['starting_balance']:,.2f} (learning kept)", kind="risk")
        if now - self.t["pub_live"] >= 90:
            self.t["pub_live"] = now
            full = now - self.t["pub_full"] >= 600
            self.publish(full)

    def research(self):
        now = time.time()
        self.brain.s["last"]["research"] = now
        self.j.activity("Testing potential model improvements (retraining candidates on recorded data)", kind="research")
        try:
            rows = self.db.labelled_checkpoints(F.SCHEMA, since=now - 21 * 86400)
            self.brain.research(now, rows)
        except Exception as e:  # noqa: BLE001
            self.j.integrity_event("research", f"{type(e).__name__}: {e}")
            traceback.print_exc()

    def live_rows(self, days):
        return self.db.labelled_checkpoints(F.SCHEMA, since=time.time() - days * 86400, live_only=True)

    # ------------------------------------------------------------------ publishing
    def snapshot_state(self):
        self.brain.account.trim()
        st = self.brain.save()
        st["journal"] = self.j.to_dict()
        st["pending"] = sorted(self.pending)
        st["saved"] = time.time()
        return st

    def market_data(self, now):
        rows = self.db.q("SELECT ts, o, h, l, c, v FROM btc1s WHERE ts>=? ORDER BY ts", (int(now) - 6 * 3600,))
        candles = {}
        for ts, o, h, l, c, v in rows:
            k = ts // 60 * 60
            if k not in candles:
                candles[k] = [k, o, h, l, c, v]
            else:
                x = candles[k]
                x[2], x[3], x[4], x[5] = max(x[2], h), min(x[3], l), c, x[5] + v
        cl = self.db.q("SELECT ts, price FROM chainlink WHERE ts>=? ORDER BY ts", (int(now) - 1800,))
        snaps = self.db.q("SELECT ts, slug, up_bid, up_ask FROM pm_snap WHERE ts>=? ORDER BY ts", (now - 1800,))
        outs = self.db.q("SELECT start, open_price, close_price, outcome FROM markets WHERE outcome IS NOT NULL AND start>=? "
                         "ORDER BY start DESC LIMIT 48", (int(now) - 6 * 3600,))
        ext = {}
        for kind in ("funding", "oi_usd", "liq_long_5m", "liq_short_5m", "premium"):
            r = self.db.one("SELECT ts, value FROM ext WHERE kind=? ORDER BY ts DESC LIMIT 1", (kind,))
            if r:
                ext[kind] = r[1]
        eth = self.db.q("SELECT ts, c FROM eth1s WHERE ts>=? AND ts % 60 = 0 ORDER BY ts", (int(now) - 6 * 3600,))
        return {"btc_1m": list(candles.values())[-360:], "chainlink": cl[::2], "pm_mid": [[round(ts), s, (b + a) / 2]
                                                                                       for ts, s, b, a in snaps[::3] if b and a],
                "outcomes": [{"start": s, "open": o, "close": c, "outcome": x} for s, o, c, x in outs],
                "ext": ext, "eth_1m": eth}

    def publish(self, full):
        now = time.time()
        ov = report.overview(self.brain, self.marks, now)
        acct = self.brain.account
        live = {"generated": now, "paper_only": True, "asset": C.A["name"], "asset_long": C.A["long"], "other": C.A["other"], "run": {"id": self.run_id, "started": self.t0, "loops": self.loop_n,
                                                               "errors": self.loop_errors, "ends": self.deadline,
                                                               "chainlink_reconnects": self.cl.reconnects},
                "health": self.h.summary(), "overview": ov, "live": self.live,
                "activity": self.j.feed[-80:], "open": acct.positions,
                "recent_trades": acct.closed[-10:], "settings": self.settings}
        self.pub.write("live.json", live)
        with open(os.path.join(self.pub.dir, "log.txt"), "w") as fh:
            fh.write("\n".join(LOG.lines) + "\n")
        if full:
            self.t["pub_full"] = now
            data = live | {
                "activity": self.j.feed[-400:],
                "market_data": self.market_data(now),
                "trades": {"closed": acct.closed[-400:], "orders": acct.orders[-150:], "rejected": acct.rejected[-150:],
                           "missed": self.brain.s["missed"][-150:]},
                "equity_curve": acct.equity_curve[-3000:],
                "patterns": self.brain.s.get("patterns"), "win_rate_table": patterns.win_rate_note(self.settings["target_win_rate"]),
                "journal": {"markets": self.j.markets[-200:], "trades": self.j.trades[-300:], "daily": self.j.daily[-120:], "experiments": self.j.experiments[-300:],
                            "discoveries": self.j.discoveries[-120:], "models": self.j.models[-200:],
                            "integrity": self.j.integrity[-120:]},
                "lab": report.lab(self.brain), "performance": report.performance(self.brain),
                "history": report.history(self.brain),
            }
            self.pub.write("data.json", data)
            self.pub.write("state.json", self.snapshot_state())
        self.pub.push(f"pm5 {'full' if full else 'live'} {int(now)}",
                      squash_day=sgt_day(now) if full else None)

    def shutdown(self):
        self.j.activity("Saving memory and handing over to the next run", kind="system")
        try:
            self.db.commit()
            self.publish(full=True)
        finally:
            self.cl.stop = True
            self.db.close()


def main():
    sys.stdout = LOG
    sys.stderr = LOG
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    minutes = float(os.environ.get("PM5_MINUTES", "340"))
    Controller(minutes).run()


if __name__ == "__main__":
    main()
