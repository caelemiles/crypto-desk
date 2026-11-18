"""Weather desk: paper-trades Polymarket's daily highest-temperature markets.

Each scan, for every tracked city/day:
  1. read what the airport has measured so far (METAR) and the latest forecast models;
  2. turn them into a probability for every temperature bucket, using what it has learned about that station;
  3. every strategy version decides, in the shadows, whether a bucket (YES) or its opposite (NO) is cheap enough;
  4. the champion's decisions are executed in the paper account against the real order book.
When a day settles, it learns: each model's error for that station, the station's bias, how much extra
uncertainty to allow, how Polymarket rounds, and whether its probabilities beat the market's.
"""
import core as K
import gamma as G
import weather as W
from lab import Lab

SEED = {"min_edge": 0.08, "min_prob": 0.05, "max_price": 0.65, "sides": "both", "max_lead": 1, "kelly": 0.2,
        "exit_edge": 0.12, "max_spread_c": 3.0, "trust": 0.85, "city_filter": True}
SPACE = {"min_edge": [0.03, 0.05, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25], "min_prob": [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.45],
         "max_price": [0.12, 0.2, 0.3, 0.45, 0.65, 0.8, 0.92], "sides": ["yes", "both", "no"], "max_lead": [0, 1, 2],
         "kelly": [0.08, 0.12, 0.2, 0.3, 0.45], "exit_edge": [0.06, 0.12, 0.2, 0.35, 2.0],
         "max_spread_c": [1.5, 2.0, 3.0, 4.0, 6.0, 99.0], "trust": [0.5, 0.65, 0.75, 0.85, 0.95, 1.0],
         "city_filter": [True, False]}

CITY_SLUGS = ["nyc", "london", "chicago", "miami", "dallas", "atlanta", "seattle", "toronto", "seoul", "tokyo", "paris",
              "ankara", "wellington", "buenos-aires", "shanghai", "austin", "los-angeles", "san-francisco", "denver",
              "houston", "phoenix", "boston", "philadelphia", "sao-paulo", "hong-kong", "munich", "madrid", "milan",
              "lucknow", "singapore", "sydney", "mexico-city", "beijing", "istanbul", "moscow", "warsaw", "taipei",
              "jakarta", "manila", "dubai", "riyadh", "tel-aviv", "las-vegas", "minneapolis", "washington-dc"]


def f2(x):
    return f"{x:.2f}" if x is not None else "?"


class WeatherDesk:
    name = "weather"

    def __init__(self, d, journal, settings):
        d = d or {}
        self.j, self.s = journal, settings
        self.acct = K.Account(d.get("account"), settings["weather_starting_balance"])
        self.lab = Lab(d.get("lab"), "weather", SEED, SPACE, journal, settings)
        self.cal = W.Calibration(d.get("cal"))
        self.events = d.get("events", {})        # slug -> tracked event
        self.skipped = d.get("skipped", {})      # slug -> reason (so each is reported once)
        self.stations = d.get("stations", {})    # icao -> {lat, lon, tz, obs: {date: max in station unit}}
        self.blocked = d.get("blocked", {})      # station -> reason (learned: model worse than the market there)
        self.exited = d.get("exited", {})        # "vid|key" -> ts: don't buy back what a version just sold
        self.last = d.get("last", {})
        self.fc = {}                             # in-memory forecast cache: icao -> (ts, data)
        self.marks = {}
        self.live = []                           # latest per-event view for the dashboard
        self.counters = d.get("counters", {"scans": 0, "decisions": 0})

    def to_dict(self):
        return {"account": self.acct.to_dict(), "lab": self.lab.to_dict(), "cal": self.cal.to_dict(),
                "events": self.events, "skipped": dict(list(self.skipped.items())[-400:]), "stations": self.stations,
                "blocked": self.blocked, "exited": {k: v for k, v in self.exited.items() if K.now() - v < 3 * 86400},
                "last": self.last, "counters": self.counters}

    # ------------------------------------------------------------------ discovery
    def discover(self, evs):
        n = 0
        for ev in evs:
            if not W.is_weather(ev) or ev["closed"] or ev["slug"] in self.events:
                continue
            info, why = W.parse_weather_event(ev)
            if not info:
                if ev["slug"] not in self.skipped:
                    self.skipped[ev["slug"]] = why
                    self.j.activity(f"Skipping {ev['title']}: {why}", kind="skip")
                continue
            info["buckets"] = [{k: v for k, v in b.items() if k != "mk"} | {
                "yes": b["mk"]["yes_token"], "no": b["mk"]["no_token"], "fee": b["mk"]["fee_rate"],
                "fee_exp": b["mk"]["fee_exp"], "min": b["mk"]["min_size"]} for b in info["buckets"]]
            info |= {"added": K.now(), "snaps": {}, "brier": {}, "status": "open", "checked": 0}
            self.events[info["slug"]] = info
            n += 1
            self.j.activity(f"Tracking {info['title']} — settles on {info['station']} airport, "
                            f"{len(info['buckets'])} buckets in °{info['unit']}", kind="market")
        return n

    def discover_by_slug(self):
        """Fallback discovery: try the standard slugs for known cities for the next few days."""
        found = []
        for d in W.next_dates(K.now(), 2)[1:]:
            tail = f"on-{d.strftime('%B').lower()}-{d.day}-{d.year}"
            for c in CITY_SLUGS:
                slug = f"highest-temperature-in-{c}-{tail}"
                if slug in self.events or slug in self.skipped:
                    continue
                try:
                    ev = G.event_by_slug(slug)
                except K.FetchError:
                    continue
                if ev:
                    found.append(ev)
                else:
                    self.skipped[slug] = "no such market"
        return self.discover(found)

    # ------------------------------------------------------------------ data
    def _station(self, icao, unit):
        st = self.stations.setdefault(icao, {"obs": {}, "unit": unit})
        t = K.now()
        if t - st.get("metar_ts", 0) > 600:
            try:
                obs, lat, lon = W.metar(icao)
                st["metar_ts"] = t
                if lat is not None:
                    st["lat"], st["lon"] = lat, lon
                tz = st.get("tz")
                if tz is not None:
                    for ts, c in obs:
                        day = W.local_day(ts, tz)
                        v = c * 1.8 + 32 if unit == "F" else c
                        if v > st["obs"].get(day, -1e9):
                            st["obs"][day] = round(v, 2)
                    for day in sorted(st["obs"])[:-6]:
                        st["obs"].pop(day, None)
            except K.FetchError as e:
                self.j.integrity_event("metar", f"{icao}: {e}")
        if st.get("lat") is None:
            return None, None
        c = self.fc.get(icao)
        if not c or t - c[0] > 3600:   # hourly: models update every 6-12 h, and the free API has a daily limit
            try:
                data = W.forecasts(st["lat"], st["lon"], unit)
                if data["det"] or data["ens"]:
                    self.fc[icao] = (t, data)
                    if data.get("tz_offset") is not None:
                        first = st.get("tz") is None
                        st["tz"] = data["tz_offset"]
                        if first:
                            st["metar_ts"] = 0    # now the local day is known, re-read observations
            except K.FetchError as e:
                self.j.integrity_event("forecast", f"{icao}: {e}")
        c = self.fc.get(icao)
        return st, (c[1] if c else None)

    # ------------------------------------------------------------------ scanning and deciding
    def scan(self):
        t = K.now()
        self.counters["scans"] += 1
        todo = [e for e in self.events.values() if e["status"] == "open"]
        ready = []
        for e in todo:
            st, fc = self._station(e["station"], e["unit"])
            if not st or not fc or st.get("tz") is None:
                continue
            if W.lead_of(e["date"], t, st["tz"]) < 0:
                e["status"] = "waiting"          # the day is over locally: wait for Polymarket to settle it
                self.j.activity(f"{e['city']} {e['date']} is over locally (max seen {f2(st['obs'].get(e['date']))}°"
                                f"{e['unit']}); waiting for settlement", kind="info")
                continue
            obs = st["obs"].get(e["date"]) if W.lead_of(e["date"], t, st["tz"]) == 0 else None
            fs = W.forecast_state(fc, e["date"], t, obs, self.cal, e["station"], e["unit"])
            if fs:
                ready.append((e, fs))
        if not ready:
            self.live = []
            return
        toks = [x for e, _ in ready for b in e["buckets"] for x in (b["yes"], b["no"])]
        books = G.books(toks)
        self.live = []
        for e, fs in ready:
            probs = W.bucket_probs(fs, e["buckets"])
            self._snapshot(e, fs, probs, books)
            self._decide(e, fs, probs, books)
            self.live.append(self._view(e, fs, probs, books))
        self._marks(books)

    def _snapshot(self, e, fs, probs, books):
        lead = fs["lead"]
        if lead not in e["snaps"]:
            e["snaps"][lead] = {"center": round(fs["center"], 3), "models": {m: round(v, 2) for m, v in fs["models"].items()},
                                "ens_var": round(fs["ens_var"], 3), "ts": K.now(), "bias": round(fs["bias"], 3),
                                "sigma": round(fs["sigma"], 3)}
        if lead not in e["brier"]:
            mids = []
            for b in e["buckets"]:
                bb, ba = G.best(books.get(b["yes"]))
                mids.append(round((bb + ba) / 2, 4) if books.get(b["yes"]) else None)
            e["brier"][lead] = {"p": [round(p, 4) for p in probs], "mkt": mids}

    def _decide(self, e, fs, probs, books):
        st_blocked = e["station"] in self.blocked
        B = self.s["weather_starting_balance"]
        champ = self.lab.champion
        sf = self.lab.size_factor()
        event_spent = sum(p["cost"] for p in self.acct.positions if p.get("event") == e["slug"])
        for b, p in zip(e["buckets"], probs):
            ybook, nbook = books.get(b["yes"]), books.get(b["no"])
            if not ybook:
                continue
            yb, ya = G.best(ybook)
            mid = (yb + ya) / 2
            sides = [("YES", p, mid, ybook, b["yes"])]
            if nbook:
                sides.append(("NO", 1 - p, 1 - mid, nbook, b["no"]))
            for side, q_model, mkt, book, tok in sides:
                key = f"{e['slug']}|{b['market']}|{side}"
                bid, ask = G.best(book)
                for v in self.lab.active():
                    P = v["params"]
                    q = P["trust"] * q_model + (1 - P["trust"]) * mkt
                    vid = v["id"]
                    if self.lab.holds(vid, key):
                        pos = v["open"][key]
                        if bid - q > P["exit_edge"] and bid > 0:
                            r = K.walk_sell(book["bids"], pos["legs"][0]["shares"], self.s["liquidity_take_pct"],
                                            b["fee"], b["fee_exp"], limit=q + P["exit_edge"] / 2)
                            if r["shares"] >= pos["legs"][0]["shares"] - 1e-6:
                                self.lab.vclose(vid, key, r["proceeds"], "sold early")
                                self.exited[f"{vid}|{key}"] = K.now()
                                if vid == champ:
                                    self._champ_exit(key, book, b, q, P)
                        continue
                    if f"{vid}|{key}" in self.exited:
                        continue
                    ok, why = self._rules(P, side, q, ask, fs, st_blocked)
                    if not ok:
                        continue
                    self.counters["decisions"] += 1
                    kelly = max(0.0, (q - ask) / (1 - ask)) * P["kelly"]
                    stake = min(kelly * B, self.s["max_stake_pct"] * B)
                    limit = min(P["max_price"], q - P["min_edge"])
                    r = K.walk_budget(book["asks"], stake, self.s["liquidity_take_pct"], b["fee"], b["fee_exp"], limit)
                    if r["shares"] < b["min"]:
                        continue
                    meta = {"event": e["slug"], "city": e["city"], "date": e["date"], "bucket": b["label"], "side": side,
                            "p": round(q, 4), "p_model": round(q_model, 4), "price": round(r["avg"], 4),
                            "lead": fs["lead_n"], "mkt": round(mkt, 4)}
                    self.lab.vopen(vid, key, r["cost"], [{"market": b["market"], "side": side, "shares": r["shares"]}], meta)
                    if vid == champ:
                        self._champ_enter(e, b, side, tok, book, q, q_model, P, fs, sf, event_spent, key)
                        event_spent = sum(p["cost"] for p in self.acct.positions if p.get("event") == e["slug"])

    def _rules(self, P, side, q, ask, fs, blocked):
        if P["city_filter"] and blocked:
            return False, "station blocked"
        if fs["lead_n"] > P["max_lead"]:
            return False, "too far ahead"
        if fs["spread_c"] > P["max_spread_c"]:
            return False, "models disagree too much"
        if P["sides"] != "both" and P["sides"].upper() != side:
            return False, "side not traded"
        if ask >= 0.995 or ask > P["max_price"]:
            return False, "too expensive"
        if q < P["min_prob"]:
            return False, "too unlikely"
        if q - ask < P["min_edge"]:
            return False, "edge too small"
        return True, ""

    def _champ_enter(self, e, b, side, tok, book, q, q_model, P, fs, sf, event_spent, key):
        a = self.acct
        if a.day_loss_hit(self.s["daily_loss_limit_pct"]):
            return
        if any(p["key"] == key for p in a.positions):
            return
        eq = a.equity_cost()
        bid, ask = G.best(book)
        kelly = max(0.0, (q - ask) / (1 - ask)) * P["kelly"] * sf
        stake = min(kelly * eq, self.s["max_stake_pct"] * eq, self.s["max_event_pct"] * eq - event_spent, a.cash * 0.95)
        if stake < 1:
            return
        limit = min(P["max_price"], q - P["min_edge"])
        r = K.walk_budget(book["asks"], stake, self.s["liquidity_take_pct"], b["fee"], b["fee_exp"], limit)
        if r["shares"] < b["min"]:
            return
        pos = {"id": a.nid("W"), "key": key, "desk": "weather", "event": e["slug"], "title": e["title"], "city": e["city"],
               "date": e["date"], "station": e["station"], "bucket": b["label"], "side": side, "shares": r["shares"],
               "cost": r["cost"], "fee": r["fee"], "price": round(r["avg"], 4), "p": round(q, 4), "p_model": round(q_model, 4),
               "lead": fs["lead_n"], "obs_max": fs["obs_max"], "center": round(fs["center"] + fs["bias"], 2),
               "sigma": round(fs["sigma"], 2), "strategy": self.lab.champion,
               "legs": [{"market": b["market"], "token": tok, "side": side, "shares": r["shares"]}]}
        a.open(pos)
        self.j.activity(f"BUY {side} {e['city']} {e['date']} {b['label']}: {r['shares']:.0f} shares at {r['avg']*100:.1f}¢ "
                        f"(${r['cost']:.2f}). Model {q*100:.0f}% vs price {ask*100:.0f}¢; forecast max "
                        f"{fs['center'] + fs['bias']:.1f}°{e['unit']}" + (f", already {fs['obs_max']:.1f} observed" if fs["obs_max"] is not None else ""),
                        kind="trade")

    def _champ_exit(self, key, book, b, q, P):
        for pos in list(self.acct.positions):
            if pos.get("key") != key:
                continue
            r = K.walk_sell(book["bids"], pos["shares"], self.s["liquidity_take_pct"], b["fee"], b["fee_exp"],
                            limit=q + P["exit_edge"] / 2)
            if r["shares"] >= pos["shares"] - 1e-6:
                rec = self.acct.close(pos, r["proceeds"], "sold early", {"exit_price": r["avg"], "exit_p": round(q, 4)})
                self.j.activity(f"SELL {pos['side']} {pos['city']} {pos['bucket']} early at {r['avg']*100:.1f}¢ "
                                f"(model now {q*100:.0f}%): {K.U.money(rec['pnl'])}", kind="trade")
                self._analyse(rec, None)

    def _marks(self, books):
        for p in self.acct.positions:
            bk = books.get(p["legs"][0]["token"])
            if bk:
                self.marks[p["id"]] = p["shares"] * G.best(bk)[0]

    def _view(self, e, fs, probs, books):
        rows = []
        for b, p in zip(e["buckets"], probs):
            bb, ba = G.best(books.get(b["yes"]))
            rows.append({"label": b["label"], "p": round(p, 4), "bid": bb, "ask": ba})
        return {"slug": e["slug"], "city": e["city"], "date": e["date"], "station": e["station"], "unit": e["unit"],
                "lead": fs["lead_n"], "forecast": round(fs["center"] + fs["bias"], 2), "bias": round(fs["bias"], 2),
                "sigma": round(fs["sigma"], 2), "obs_max": fs["obs_max"], "models": {m: round(v, 1) for m, v in fs["models"].items()},
                "blocked": self.blocked.get(e["station"]), "buckets": rows}

    # ------------------------------------------------------------------ settlement and learning
    def resolve(self):
        t = K.now()
        for e in list(self.events.values()):
            if e["status"] == "done":
                if t - e.get("done_at", t) > 3 * 86400:
                    self.events.pop(e["slug"], None)
                continue
            st = self.stations.get(e["station"], {})
            over = e["status"] == "waiting" or (e.get("end") and t > e["end"] + 3600)
            if not over or t - e.get("checked", 0) < 900:
                continue
            e["checked"] = t
            try:
                ev = G.event_by_slug(e["slug"])
            except K.FetchError:
                continue
            if not ev:
                continue
            res = {m["id"]: G.resolved_yes(m) for m in ev["markets"]}
            winners = [i for i, b in enumerate(e["buckets"]) if res.get(b["market"]) is True]
            if len(winners) != 1 or any(res.get(b["market"]) is None for b in e["buckets"]):
                if t - (e.get("end") or t) > 5 * 86400:
                    e["status"] = "done"
                    e["done_at"] = t
                    self.j.integrity_event("settlement", f"{e['slug']} did not settle cleanly after 5 days")
                continue
            self._settle(e, res, winners[0], st.get("obs", {}).get(e["date"]))

    def _settle(self, e, res, wi, metar_max):
        t = K.now()
        e["status"], e["done_at"] = "done", t
        win = e["buckets"][wi]
        self.lab.settle(res)
        for pos in list(self.acct.positions):
            if pos.get("event") != e["slug"]:
                continue
            leg = pos["legs"][0]
            yes = res.get(leg["market"])
            payout = leg["shares"] if (leg["side"] == "YES") == yes else 0.0
            rec = self.acct.close(pos, payout, "settled", {"winner": win["label"], "actual": metar_max})
            self._analyse(rec, e)
        actual, rounding = W.actual_from(e["buckets"], wi, metar_max, self.cal, e["station"])
        lines = []
        if actual is not None:
            for lead, snap in e["snaps"].items():
                r = self.cal.learn(e["station"], e["unit"], lead, actual, snap, rounding)
                lines.append(f"lead {lead}d: forecast {snap['center']:.1f}, actual {actual:.1f} (off by {r['r']:+.1f})")
        # Brier: were our probabilities better than the market's?
        for lead, bz in e["brier"].items():
            for i, (p, m) in enumerate(zip(bz["p"], bz["mkt"])):
                if m is not None:
                    y = 1 if i == wi else 0
                    self.cal.brier.append([round(t), e["station"], lead, round((p - y) ** 2, 5), round((m - y) ** 2, 5)])
        st = self.cal.station(e["station"])
        self.j.market_entry({"kind": "weather", "title": f"{e['city']} {e['date']}: {win['label']} won",
                             "body": f"Settled on {e['station']}; METAR max {f2(metar_max)}°{e['unit']}. " + "; ".join(lines)
                                     + f". Learned bias now {self.cal.bias(e['station'], '1'):+.2f}° (next-day), "
                                     f"rounding looks like '{self.cal.rounding(e['station'])}' "
                                     f"({st['round']} round / {st['floor']} truncate).", "evidence": "measured"})

    def _analyse(self, rec, e):
        won = rec["pnl"] > 0
        if rec["how"] == "sold early":
            why = f"sold early at {rec.get('exit_price', 0)*100:.0f}¢ when the market moved past the model"
        elif won:
            why = f"{rec['winner']} won as forecast"
        else:
            why = (f"{rec['winner']} won instead; actual max {f2(rec.get('actual'))}° vs forecast {rec['center']}°"
                   f" ± {rec['sigma']}")
        self.j.trade({"desk": "weather", "id": rec["id"], "title": f"{rec['side']} {rec['city']} {rec['date']} {rec['bucket']}",
                      "pnl": rec["pnl"], "won": won, "body": f"Bought at {rec['price']*100:.1f}¢ with the model at "
                      f"{rec['p']*100:.0f}% ({rec['lead']} day(s) ahead, strategy {rec['strategy']}). {why}. "
                      f"Result {K.U.money(rec['pnl'])}.", "evidence": "measured"})

    def learn(self):
        """Every few hours: compare our accuracy with the market's per station, evolve the strategy versions."""
        t = K.now()
        by = {}
        for ts, stn, lead, ours, mkt in self.cal.brier:
            if t - ts < 60 * 86400:
                by.setdefault(stn, []).append((ours, mkt))
        summary = []
        for stn, xs in by.items():
            n = len(xs)
            o = sum(x[0] for x in xs) / n
            m = sum(x[1] for x in xs) / n
            summary.append({"station": stn, "n": n, "ours": round(o, 4), "market": round(m, 4)})
            if n >= 120 and o > m + 0.004 and stn not in self.blocked:
                self.blocked[stn] = f"our Brier {o:.4f} worse than market {m:.4f} over {n} buckets"
                self.j.discovery(f"Market beats the model at {stn}", self.blocked[stn] + "; versions with the city "
                                 "filter stop trading it", "measured", key=f"block-{stn}")
            elif stn in self.blocked and n >= 60 and o < m - 0.001:
                self.blocked.pop(stn, None)
                self.j.discovery(f"Model beats the market again at {stn}", f"our Brier {o:.4f} vs market {m:.4f} over "
                                 f"{n} buckets; unblocked", "measured", key=f"block-{stn}")
        self.last["brier"] = sorted(summary, key=lambda x: -x["n"])
        tot = [x for v in by.values() for x in v]
        if len(tot) >= 30:
            o = sum(x[0] for x in tot) / len(tot)
            m = sum(x[1] for x in tot) / len(tot)
            self.j.discovery("Forecast accuracy vs the market", f"Over {len(tot)} buckets in 60 days our Brier score is "
                             f"{o:.4f} and the market's is {m:.4f} (lower is better): the model is "
                             f"{'better' if o < m else 'worse'} than the crowd.", "measured", key="brier-all")
        for kind, text, evidence in self.lab.evolve():
            self.j.experiment({"desk": "weather", "kind": kind, "body": text, "evidence": evidence})

    def calibration_table(self):
        rows = []
        for stn, st in self.cal.st.items():
            row = {"station": stn, "unit": st.get("unit"), "rounding": self.cal.rounding(stn)}
            for lead in W.LEADS:
                n = len(st["samples"].get(lead, []))
                row[f"n{lead}"] = n
                row[f"bias{lead}"] = round(self.cal.bias(stn, lead), 2)
                row[f"sigma{lead}"] = round(self.cal.sigma(stn, lead, st.get("unit") or "F"), 2)
            row["weights"] = {m: round(w, 2) for m, w in self.cal.model_weights(stn, "1").items()}
            rows.append(row)
        return sorted(rows, key=lambda r: -(r["n0"] + r["n1"] + r["n2"]))

    def marks_total(self):
        return self.acct.equity(self.marks)

