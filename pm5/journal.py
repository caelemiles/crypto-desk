"""Learning Journal. Everything the AI does or learns is written here and shown on the dashboard.

Every entry is tagged with its evidence level:
- "measured"   : computed from recorded outcomes, with the sample size shown
- "hypothesis" : an idea the AI generated that has not been proven yet
- "event"      : something that happened (a market opened, a trade filled)
"""
import time

from util import sgt_hm

LIMITS = {"markets": 600, "feed": 600, "trades": 1500, "experiments": 800, "discoveries": 400, "models": 400, "integrity": 300}


class Journal:
    def __init__(self, d=None):
        d = d or {}
        self.feed = d.get("activity", [])
        self.trades = d.get("trades", [])          # individual trade analyses
        self.daily = d.get("daily", [])            # daily learning reports (all kept)
        self.experiments = d.get("experiments", [])  # strategy development journal
        self.discoveries = d.get("discoveries", [])  # measured findings and hypotheses
        self.models = d.get("models", [])          # model / champion history events
        self.integrity = d.get("integrity", [])    # data problems seen and handled
        self.markets = d.get("markets", [])        # a measured explanation of every market
        self.seq = d.get("seq", 0)

    def to_dict(self):
        out = {"seq": self.seq, "daily": self.daily[-400:]}
        for k, n in LIMITS.items():
            out["activity" if k == "feed" else k] = getattr(self, k)[-n:]
        return out

    def _push(self, bucket, entry):
        self.seq += 1
        entry = {"n": self.seq, "ts": round(time.time(), 1)} | entry
        lst = getattr(self, bucket)
        lst.append(entry)
        if len(lst) > LIMITS.get(bucket, 10_000) * 1.2:
            del lst[: len(lst) - LIMITS.get(bucket, 10_000)]
        return entry

    def activity(self, text, kind="info"):
        e = self._push("feed", {"text": text, "kind": kind, "time": sgt_hm(time.time())})
        print(f"[{e['time']}] {text}", flush=True)
        return e

    def market_entry(self, entry):
        return self._push("markets", entry)

    def trade(self, entry):
        return self._push("trades", entry)

    def experiment(self, entry):
        return self._push("experiments", entry)

    def discovery(self, title, body, evidence, data=None, key=None):
        if key:  # one live version per key: update it in place instead of piling up near-duplicates
            for e in reversed(self.discoveries):
                if e.get("key") == key:
                    if e.get("body") != body:
                        e.update({"title": title, "body": body, "evidence": evidence, "data": data or {},
                                  "ts": round(time.time(), 1), "updates": e.get("updates", 0) + 1})
                    return e
        return self._push("discoveries", {"title": title, "body": body, "evidence": evidence, "data": data or {},
                                          "key": key})

    def model_event(self, title, body, data=None):
        return self._push("models", {"title": title, "body": body, "data": data or {}})

    def integrity_event(self, kind, detail):
        return self._push("integrity", {"kind": kind, "detail": str(detail)[:300]})

    def daily_report(self, report):
        self.daily = [d for d in self.daily if d.get("day") != report["day"]]
        self.seq += 1
        self.daily.append({"n": self.seq, "ts": round(time.time(), 1)} | report)
