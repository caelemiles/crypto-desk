#!/usr/bin/env python3
"""Offline test: runs the real PMX controller on a simulated Polymarket + weather world for N simulated days.

    python pmx/sim_test.py 6

The simulated crowd misprices some temperature buckets and leaves occasional arbitrage gaps, so a working bot
should learn station biases, trade the mispricing profitably and lock in the gaps. Nothing touches the network.
"""
import json
import math
import os
import random
import sys
import tempfile
import urllib.parse
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.pop("GITHUB_REPOSITORY", None)
os.environ["PMX_WORK"] = tempfile.mkdtemp()
os.environ["PMX_PUBLISH"] = os.path.join(os.environ["PMX_WORK"], "pub")

import core as K  # noqa: E402

RND = random.Random(5)
CLOCK = {"t": datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp()}
CITIES = [  # slug, name, icao, unit, tz offset s, lat, lon, typical max, station bias of the models
    ("nyc", "NYC", "KLGA", "F", -5 * 3600, 40.78, -73.88, 52, 2.0),
    ("london", "London", "EGLC", "C", 0, 51.5, 0.05, 11, -0.8),
    ("seoul", "Seoul", "RKSI", "C", 9 * 3600, 37.46, 126.44, 9, 0.0),
]
MONTH = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
         "november", "december"]


class World:
    def __init__(self):
        self.events = {}     # slug -> raw gamma event
        self.books = {}      # token -> (fair prob, noise)
        self.truth = {}      # (icao, date) -> true max in unit
        self.mid = 0
        self.tok = 0

    def token(self):
        self.tok += 1
        return f"tok{self.tok}"

    def market(self, q, group, end, neg=True, fees=False):
        self.mid += 1
        y, n = self.token(), self.token()
        return {"id": str(self.mid), "slug": f"m{self.mid}", "question": q, "groupItemTitle": group,
                "conditionId": f"c{self.mid}", "outcomes": '["Yes","No"]', "clobTokenIds": json.dumps([y, n]),
                "outcomePrices": '["0.5","0.5"]', "closed": False, "active": True, "acceptingOrders": True,
                "negRisk": neg, "feesEnabled": fees, "orderMinSize": 5, "orderPriceMinTickSize": 0.01,
                "endDate": datetime.fromtimestamp(end, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "description": "", "_yes": y, "_no": n}

    def true_max(self, c, date):
        k = (c[2], date)
        if k not in self.truth:
            self.truth[k] = c[7] + RND.gauss(0, 3 if c[3] == "F" else 1.7)
        return self.truth[k]

    def ensure_weather(self):
        now = CLOCK["t"]
        for c in CITIES:
            for dd in range(0, 3):
                d = datetime.fromtimestamp(now + c[4], timezone.utc) + timedelta(days=dd)
                date = d.strftime("%Y-%m-%d")
                slug = f"highest-temperature-in-{c[0]}-on-{MONTH[d.month-1]}-{d.day}-{d.year}"
                if slug in self.events:
                    continue
                tm = self.true_max(c, date)
                end = datetime(d.year, d.month, d.day, 12, tzinfo=timezone.utc).timestamp() - c[4] + 12 * 3600
                mks, base = [], round(c[7])
                u = c[3]
                step = 2 if u == "F" else 1
                lo0 = base - 5 * step
                labels = [(None, lo0 - 1)] + [(lo0 + i * step, lo0 + i * step + step - 1) for i in range(10)] + [(lo0 + 10 * step, None)]
                for lo, hi in labels:
                    if lo is None:
                        q = f"Will the highest temperature in {c[1]} be {hi}°{u} or below on {MONTH[d.month-1].title()} {d.day}?"
                    elif hi is None:
                        q = f"Will the highest temperature in {c[1]} be {lo}°{u} or higher on {MONTH[d.month-1].title()} {d.day}?"
                    elif lo == hi:
                        q = f"Will the highest temperature in {c[1]} be {lo}°{u} on {MONTH[d.month-1].title()} {d.day}?"
                    else:
                        q = f"Will the highest temperature in {c[1]} be between {lo}-{hi}°{u} on {MONTH[d.month-1].title()} {d.day}?"
                    m = self.market(q, "", end)
                    m["_bucket"] = (lo, hi)
                    mks.append(m)
                # the crowd: a forecast with the station bias unknown to it and a fat-tailed error
                crowd = tm + c[8] + RND.gauss(0, 2.5 if u == "F" else 1.4)
                sig = (3.5 if u == "F" else 1.9) + dd
                ps = []
                for m in mks:
                    lo, hi = m["_bucket"]
                    a = -1e9 if lo is None else lo - 0.5
                    b = 1e9 if hi is None else hi + 0.5
                    ps.append(max(0.004, _phi((b - crowd) / sig) - _phi((a - crowd) / sig)))
                s = sum(ps)
                for m, p in zip(mks, ps):
                    self.books[m["_yes"]] = p / s
                    self.books[m["_no"]] = 1 - p / s
                self.events[slug] = {"id": slug, "slug": slug, "title": f"Highest temperature in {c[1]} on {MONTH[d.month-1].title()} {d.day}?",
                                     "description": f"Resolves to the highest temperature recorded at the station, "
                                                    f"https://www.wunderground.com/history/daily/xx/{c[0]}/{c[2]}.",
                                     "negRisk": True, "endDate": mks[0]["endDate"], "closed": False, "active": True,
                                     "markets": mks, "tags": [{"slug": "weather"}], "_city": c, "_date": date,
                                     "_end": end}

    def ensure_arb(self):
        now = CLOCK["t"]
        live = [e for e in self.events.values() if e.get("_arb") and not e["closed"]]
        while len(live) < 6:
            n = RND.choice([3, 4, 6])
            end = now + RND.uniform(0.5, 5) * 86400
            slug = f"arb-{self.mid}"
            mks = [self.market(f"Will option {i} win?", f"Option {i}", end) for i in range(n)]
            ps = [RND.random() + 0.2 for _ in mks]
            s = sum(ps)
            gap = RND.choice([0.0, 0.0, 0.0, -0.09, 0.04])   # sometimes the YESes add up to less than $1
            for m, p in zip(mks, ps):
                self.books[m["_yes"]] = p / s + gap / n
                self.books[m["_no"]] = 1 - p / s - gap / n
            ev = {"id": slug, "slug": slug, "title": f"Who wins contest {slug}?", "description": "", "negRisk": True,
                  "endDate": mks[0]["endDate"], "closed": False, "active": True, "markets": mks,
                  "tags": [{"slug": "sports"}], "_arb": True, "_end": end, "_win": RND.randrange(n), "_ps": [p / s for p in ps]}
            self.events[slug] = ev
            live.append(ev)
        lad = [e for e in self.events.values() if e.get("_lad") and not e["closed"]]
        if len(lad) < 2:
            end = now + 86400
            slug = f"btc-ladder-{self.mid}"
            spot = 100000 + RND.gauss(0, 2000)
            mks = []
            for k in (96000, 98000, 100000, 102000, 104000):
                m = self.market(f"Will the price of Bitcoin be above ${k:,} on November 9?", f"{k:,}", end, neg=False)
                p = 1 - _phi((k - spot) / 2500)
                self.books[m["_yes"]] = p
                m["_k"] = k
                mks.append(m)
            for i, m in enumerate(mks[:-1]):   # sometimes a lower level is priced below the next one up
                if RND.random() < 0.3:
                    self.books[m["_yes"]] = self.books[mks[i + 1]["_yes"]] - 0.04
            for m in mks:
                self.books[m["_no"]] = 1 - self.books[m["_yes"]]
            ev = {"id": slug, "slug": slug, "title": "Bitcoin above ___ on November 9?", "description": "", "negRisk": False,
                  "endDate": mks[0]["endDate"], "closed": False, "active": True, "markets": mks, "tags": [{"slug": "crypto"}],
                  "_lad": True, "_end": end, "_final": spot + RND.gauss(0, 1500)}
            self.events[slug] = ev

    def settle(self):
        now = CLOCK["t"]
        for e in self.events.values():
            if e["closed"] or now < e["_end"] + 3600:
                continue
            if e.get("_arb"):
                win = e["_win"]
                for i, m in enumerate(e["markets"]):
                    m["outcomePrices"] = '["1","0"]' if i == win else '["0","1"]'
                    m["closed"] = True
            elif e.get("_lad"):
                for m in e["markets"]:
                    m["outcomePrices"] = '["1","0"]' if e["_final"] > m["_k"] else '["0","1"]'
                    m["closed"] = True
            else:
                c = e["_city"]
                rep = round(self.true_max(c, e["_date"]))
                for m in e["markets"]:
                    lo, hi = m["_bucket"]
                    ok = (lo is None or rep >= lo) and (hi is None or rep <= hi)
                    m["outcomePrices"] = '["1","0"]' if ok else '["0","1"]'
                    m["closed"] = True
            e["closed"] = True

    def book(self, tok):
        p = self.books.get(tok)
        if p is None:
            return None
        p = min(0.98, max(0.01, p + RND.gauss(0, 0.004)))
        bid, ask = max(0.001, round(p - 0.01, 3)), min(0.999, round(p + 0.01, 3))
        return {"asset_id": tok, "timestamp": str(int(CLOCK["t"] * 1000)),
                "bids": [{"price": str(round(bid - i * 0.01, 3)), "size": str(RND.randint(20, 300))} for i in range(4) if bid - i * 0.01 > 0],
                "asks": [{"price": str(round(ask + i * 0.01, 3)), "size": str(RND.randint(20, 300))} for i in range(4) if ask + i * 0.01 < 1]}

    def public(self, e):
        out = {k: v for k, v in e.items() if not k.startswith("_")}
        mks = []
        for m in e["markets"]:
            mm = {k: v for k, v in m.items() if not k.startswith("_")}
            p = self.books.get(m["_yes"], 0.5)
            if not m["closed"]:
                mm["bestAsk"] = round(min(0.999, p + 0.01), 3)
                mm["bestBid"] = round(max(0.001, p - 0.01), 3)
            mks.append(mm)
        out["markets"] = mks
        return out


W = World()


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def obs_series(c, hours):
    out = []
    now = CLOCK["t"]
    for h in range(hours, -1, -1):
        ts = int(now // 3600 * 3600 - h * 3600)
        lt = datetime.fromtimestamp(ts + c[4], timezone.utc)
        tm = W.true_max(c, lt.strftime("%Y-%m-%d"))
        tm_c = (tm - 32) / 1.8 if c[3] == "F" else tm
        v = tm_c - 6 * (1 - (1 + math.cos((lt.hour - 15) / 24 * 2 * math.pi)) / 2)
        out.append({"icaoId": c[2], "obsTime": ts, "temp": round(v, 1), "lat": c[5], "lon": c[6], "rawOb": ""})
    return out


def hourly(c, unit_f, ensemble, models):
    now = CLOCK["t"]
    start = datetime.fromtimestamp(now + c[4], timezone.utc).replace(hour=0, minute=0, second=0) - timedelta(days=1)
    times = [(start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:00") for h in range(24 * 5)]

    def series(bias, noise):
        vals = []
        for t in times:
            date, hr = t[:10], int(t[11:13])
            tm = W.true_max(c, date)
            if c[3] == "C" and unit_f:
                tm = tm * 1.8 + 32
            lead = max(0, (datetime.strptime(date, "%Y-%m-%d") - datetime.strptime(times[24][:10], "%Y-%m-%d")).days)
            amp = 10 if unit_f else 6
            vals.append(round(tm + bias + noise * (1 + 0.5 * lead) - amp * (1 - (1 + math.cos((hr - 15) / 24 * 2 * math.pi)) / 2), 2))
        return vals
    scale = 1.8 if unit_f else 1.0
    b = c[8] * (1 if c[3] == "F" else 1) * (scale if c[3] == "C" else 1)
    h = {"time": times}
    if ensemble:
        base = RND.gauss(0, 1.0 * scale)
        h["temperature_2m"] = series(-b, base)
        for i in range(1, 11):
            h[f"temperature_2m_member{i:02d}"] = series(-b, base + RND.gauss(0, 0.9 * scale))
    else:
        for mdl in models:
            h[f"temperature_2m_{mdl}"] = series(-b + (0.5 if mdl.startswith("icon") else 0), RND.gauss(0, 1.1 * scale))
    return {"utc_offset_seconds": c[4], "timezone": "x", "hourly": h}


def fake_http(url, data=None, timeout=10, retries=2):
    u = urllib.parse.urlparse(url)
    q = dict(urllib.parse.parse_qsl(u.query))
    if "raw.githubusercontent" in url:
        raise K.FetchError("offline")
    if u.netloc.startswith("gamma"):
        W.ensure_weather()
        W.ensure_arb()
        W.settle()
        if "slug" in q:
            e = W.events.get(q["slug"])
            return [W.public(e)] if e else []
        evs = [e for e in W.events.values() if q.get("closed") != "false" or not e["closed"]]
        if q.get("tag_slug"):
            evs = [e for e in evs if any(t["slug"] == q["tag_slug"] for t in e["tags"])]
        off, lim = int(q.get("offset", 0)), int(q.get("limit", 100))
        return [W.public(e) for e in evs[off:off + lim]]
    if u.netloc.startswith("clob"):
        return [b for b in (W.book(x["token_id"]) for x in data) if b]
    if "aviationweather" in u.netloc:
        c = next(c for c in CITIES if c[2] == q["ids"])
        return obs_series(c, int(q.get("hours", 24)))
    if "open-meteo" in u.netloc:
        lat = float(q["latitude"])
        c = min(CITIES, key=lambda c: abs(c[5] - lat))
        unit_f = q.get("temperature_unit") == "fahrenheit"
        return hourly(c, unit_f, "ensemble" in u.netloc, q.get("models", "").split(","))
    raise K.FetchError(f"no fake for {url}")


def main(days):
    K._clock["fn"] = lambda: CLOCK["t"]
    K._http["fn"] = fake_http
    import run
    ctl = run.Controller(minutes=days * 1440)

    def fast_sleep(s):
        CLOCK["t"] += 60
        return False
    ctl.sleep = fast_sleep
    steps = int(days * 1440)
    for i in range(steps):
        try:
            ctl.tick()
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            raise
        CLOCK["t"] += 60
        if i % 1440 == 0:
            w, a = ctl.w.acct, ctl.a.acct
            print(f"day {i // 1440}: weather eq {w.equity(ctl.w.marks):.2f} trades {len(w.closed)} open {len(w.positions)} | "
                  f"arb eq {a.equity(ctl.a.marks):.2f} sets {len(a.closed)} open {len(a.positions)}", flush=True)
    ctl.shutdown()
    w, a = ctl.w, ctl.a
    print("\nWEATHER", w.acct.stats(), "equity", round(w.acct.equity(w.marks), 2))
    print("champion", w.lab.champion, w.lab.champ()["params"])
    for r in w.calibration_table():
        print("  cal", r["station"], "bias1", r["bias1"], "n1", r["n1"], "sigma1", r["sigma1"], "rounding", r["rounding"])
    print("  brier", w.last.get("brier"))
    print("ARB", a.acct.stats(), "equity", round(a.acct.equity(a.marks), 2), "slip", a.learned_slip())
    print("  kinds", a.stats["by_kind"], "lifetimes", len(a.lifetimes))
    print("journal: trades", len(ctl.j.trades), "experiments", len(ctl.j.experiments), "discoveries", len(ctl.j.discoveries))
    for d in ctl.j.discoveries[-6:]:
        print("  *", d["title"], "-", d["body"][:160])
    import readiness as R
    print("readiness weather", [(c["label"], c["ok"], c["value"]) for c in R.weather(w, ctl.settings)["checks"]])
    print("readiness arb", [(c["label"], c["ok"], c["value"]) for c in R.arb(a, ctl.settings)["checks"]])
    pub = os.environ["PMX_PUBLISH"]
    for f in ("live.json", "data.json", "state.json"):
        print(f, os.path.getsize(os.path.join(pub, f)), "bytes")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 3)
