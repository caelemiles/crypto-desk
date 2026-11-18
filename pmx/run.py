#!/usr/bin/env python3
"""PMX controller: runs the Weather desk and the Arbitrage desk in one loop on GitHub Actions.

  every ~1 min   arbitrage scan (rotating through the active events, biggest first), new weather events noticed
  every ~5 min   weather scan (observations, forecasts, probabilities, decisions)
  every ~10 min  settlement checks; dashboard data and memory saved to the pmx-data branch
  every ~3 h     learning: calibration review, strategy versions promoted / retired / born

It runs for PMX_MINUTES (default ~5h40m), saves, and exits; the next job carries on from the saved memory.
PAPER TRADING ONLY. No keys, no wallet, no order endpoints.
"""
import os
import signal
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import core as K  # noqa: E402
import gamma as G  # noqa: E402
import readiness as R  # noqa: E402
from adesk import ArbDesk  # noqa: E402
from journal import Journal  # noqa: E402  (pm5/journal.py)
from publish import Publisher  # noqa: E402  (pm5/publish.py)
from wdesk import WeatherDesk  # noqa: E402

assert K.PAPER_ONLY
EPOCH = "pmx-3"
STOP = False
EVERY = {"arb": 60, "weather": 300, "wdiscover": 1800, "slugs": 3 * 3600, "resolve": 600, "learn": 3 * 3600,
         "pub_live": 120, "pub_full": 600, "settings": 600}


def _stop(*_):
    global STOP
    STOP = True


class Controller:
    def __init__(self, minutes):
        self.t0 = K.now()
        self.deadline = self.t0 + minutes * 60
        self.settings, self.src = K.load_settings()
        self.pub = Publisher(K.PUBLISH_DIR, os.environ.get("GITHUB_REPOSITORY"), os.environ.get("PMX_TOKEN"), branch=K.BRANCH)
        self.pub.init()
        st = self.pub.read("state.json", {}) or {}
        if st and st.get("epoch") != EPOCH:
            st = {}
        self.j = Journal(st.get("journal"))
        if st.get("reset_token") not in (None, self.settings["reset_token"]):
            self.j.activity("Paper accounts reset (reset_token changed); everything learned is kept", kind="risk")
            for k in ("weather", "arb"):
                if st.get(k):
                    st[k]["account"] = None
        self.w = WeatherDesk(st.get("weather"), self.j, self.settings)
        self.a = ArbDesk(st.get("arb"), self.j, self.settings)
        self.t = dict(st.get("timers", {}))
        self.t.pop("pub_live", None)
        self.t.pop("pub_full", None)
        self.reached = st.get("reached", {})
        self.loops, self.errors = 0, 0
        self.run_id = os.environ.get("GITHUB_RUN_ID", "local")

    def due(self, k):
        n = K.now()
        if n - self.t.get(k, 0) >= EVERY[k]:
            self.t[k] = n
            return True
        return False

    def run(self):
        self.j.activity(f"PMX started (run {self.run_id}); settings from {self.src}; paper trading only", kind="system")
        while not STOP and K.now() < self.deadline:
            t = K.now()
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001
                self.errors += 1
                self.j.integrity_event("loop", f"{type(e).__name__}: {e}")
                traceback.print_exc()
            self.loops += 1
            if self.sleep(max(1.0, 5.0 - (K.now() - t))):
                break
        self.shutdown()

    def sleep(self, s):
        time.sleep(s)
        return False

    def tick(self):
        if self.due("settings"):
            new, _ = K.load_settings()
            self.settings.update(new)
        if self.due("arb"):
            evs = self.fetch_events()
            self.w.discover(evs)
            self.a.scan(evs)
            self.a.mark()
        if self.due("wdiscover"):
            try:
                self.w.discover(G.events({"tag_slug": "weather", "closed": "false"}, pages=25)[0])
            except K.FetchError as e:
                self.j.integrity_event("gamma", f"weather tag: {e}")
        if self.due("slugs"):
            self.w.discover_by_slug()
        if self.due("weather"):
            self.w.scan()
        if self.due("resolve"):
            self.w.resolve()
            self.a.resolve()
        if self.due("learn"):
            self.w.learn()
            self.a.learn()
        self.w.acct.record(self.w.marks)
        self.a.acct.record(self.a.marks)
        self._gate()
        if self.due("pub_full"):
            self.publish(full=True)
        elif self.due("pub_live"):
            self.publish(full=False)

    def fetch_events(self):
        base = {"closed": "false", "order": "volume24hr", "ascending": "false"}
        evs = []
        try:
            top, _ = G.events(base, pages=3)                                    # the busiest events, every minute
            more, cur = G.events(base, pages=5, cursor=self.a.cursor or None)   # plus a rotating slice of the rest
            self.a.cursor = cur
            evs = top + more
        except K.FetchError as e:
            self.a.cursor = None
            self.j.integrity_event("gamma", str(e))
        uniq = {}
        for e in evs:
            uniq[e["slug"]] = e
        return list(uniq.values())

    def _gate(self):
        for name, fn, desk in (("weather", R.weather, self.w), ("arb", R.arb, self.a)):
            g = fn(desk, self.settings)
            if g["reached"] and not self.reached.get(name):
                self.reached[name] = K.now()
                self.j.discovery(f"{name.title()} desk reached its paper target",
                                 "All readiness checks passed. Nothing switches to real money: Polymarket does not take "
                                 "new orders from Singapore or the US.", "measured", key=f"target-{name}")
            elif not g["reached"] and self.reached.get(name):
                self.reached.pop(name)

    # ------------------------------------------------------------------ publishing
    def overview(self, desk):
        a = desk.acct
        eq = a.equity(desk.marks)
        return {"equity": round(eq, 2), "cash": round(a.cash, 2), "start": a.start, "pnl": round(eq - a.start, 2),
                "open": len(a.positions), "max_dd": round(a.max_dd, 4), "all": a.stats(),
                "d1": a.stats(K.now() - 86400), "d7": a.stats(K.now() - 7 * 86400), "champion": desk.lab.champion,
                "champion_params": desk.lab.champ()["params"], "size_factor": desk.lab.size_factor()}

    def snapshot(self):
        return {"epoch": EPOCH, "reset_token": self.settings["reset_token"], "journal": self.j.to_dict(),
                "weather": self.w.to_dict(), "arb": self.a.to_dict(), "timers": self.t, "reached": self.reached}

    def publish(self, full):
        now = K.now()
        live = {"generated": now, "paper_only": True, "real_trading": R.REAL_TRADING,
                "run": {"id": self.run_id, "started": self.t0, "ends": self.deadline, "loops": self.loops, "errors": self.errors},
                "weather": {"overview": self.overview(self.w), "readiness": R.weather(self.w, self.settings),
                            "open": self.w.acct.positions, "live": self.w.live[:30], "counters": self.w.counters},
                "arb": {"overview": self.overview(self.a), "readiness": R.arb(self.a, self.settings),
                        "open": self.a.acct.positions, "live": self.a.live, "stats": self.a.stats,
                        "slip": self.a.learned_slip()},
                "activity": self.j.feed[-120:], "settings": self.settings}
        self.pub.write("live.json", live)
        if full:
            data = live | {
                "weather": live["weather"] | {"closed": self.w.acct.closed[-400:], "curve": self.w.acct.curve[-3000:],
                                              "lab": self.w.lab.table(), "calibration": self.w.calibration_table(),
                                              "brier": self.w.last.get("brier", []), "blocked": self.w.blocked,
                                              "samples": self.w.samples,
                                              "tracked": len(self.w.events)},
                "arb": live["arb"] | {"closed": self.a.acct.closed[-400:], "curve": self.a.acct.curve[-3000:],
                                      "lab": self.a.lab.table(), "lifetimes": self.a.lifetimes[-300:],
                                      "slips": self.a.slips[-300:]},
                "journal": {"trades": self.j.trades[-300:], "experiments": self.j.experiments[-300:],
                            "discoveries": self.j.discoveries[-120:], "markets": self.j.markets[-200:],
                            "integrity": self.j.integrity[-120:]},
                "activity": self.j.feed[-400:],
            }
            self.pub.write("data.json", data)
            self.pub.write("state.json", self.snapshot())
        self.pub.push(f"pmx {'full' if full else 'live'} {int(now)}", squash_day=K.U.sgt_day(now) if full else None)

    def shutdown(self):
        self.j.activity("Saving memory and handing over to the next run", kind="system")
        self.publish(full=True)


def main():
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    Controller(float(os.environ.get("PMX_MINUTES", "340"))).run()


if __name__ == "__main__":
    main()
