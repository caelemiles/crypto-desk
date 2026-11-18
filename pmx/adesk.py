"""Arbitrage desk: finds sets of Polymarket positions that are guaranteed to pay more than they cost.

Three kinds:
  - YES set: an event whose outcomes are mutually exclusive and cover everything (a "negRisk" event). Exactly one
    YES pays $1, so buying one YES of every outcome for less than $1 (after fees) locks in a profit.
  - NO set: same event, the other way round. N-1 of the N NOs pay $1, so buying every NO for less than N-1 wins.
  - Ladder: two markets on the same thing at different levels ("above 100k" and "above 105k"). YES on the lower
    level plus NO on the higher level pays at least $1 whatever happens, and $2 if the result lands in between:
    the worst case is the profit margin, the best case is much more.

It learns from every opportunity: how long gaps last, how much prices move between buying the first and the last
leg (leg risk), which kinds of events produce real gaps, and which thresholds and lock-up times pay best.
"""
import re
import statistics

import core as K
import gamma as G
from lab import Lab

SEED = {"min_margin": 0.01, "max_days": 7, "kinds": "all", "max_legs": 12, "min_profit": 0.25, "haircut": 1.0}
SPACE = {"min_margin": [0.003, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05], "max_days": [0.5, 1, 3, 7, 14, 30],
         "kinds": ["all", "yes_set", "no_set", "ladder", "sets"], "max_legs": [2, 4, 8, 12, 20, 40],
         "min_profit": [0.05, 0.25, 0.5, 1.0, 2.5], "haircut": [0.0, 0.5, 1.0, 1.5, 2.5]}
SIZES = [5, 10, 20, 35, 50, 75, 100, 150, 250, 400, 600, 1000, 1600, 2500]

RE_LEVEL = re.compile(r"\b(above|over|greater than|higher than|at least|more than|below|under|less than|lower than)\s*"
                      r"\$?\s*(-?[\d,]+(?:\.\d+)?)\s*([kKmMbB])?\b", re.I)
UP_WORDS = ("above", "over", "greater than", "higher than", "at least", "more than")


def _level(q):
    m = RE_LEVEL.search(q)
    if not m:
        return None
    v = float(m.group(2).replace(",", ""))
    v *= {"k": 1e3, "m": 1e6, "b": 1e9}.get((m.group(3) or "").lower(), 1)
    direction = "up" if m.group(1).lower() in UP_WORDS else "down"
    template = (q[:m.start()] + q[m.end():]).strip().lower()
    return direction, v, template


class ArbDesk:
    name = "arb"

    def __init__(self, d, journal, settings):
        d = d or {}
        self.j, self.s = journal, settings
        self.acct = K.Account(d.get("account"), settings["arb_starting_balance"])
        self.lab = Lab(d.get("lab"), "arb", SEED, SPACE, journal, settings)
        self.seen = d.get("seen", {})          # opportunity key -> {first, last, scans, best, kind, title}
        self.lifetimes = d.get("lifetimes", [])  # closed opportunities: [kind, minutes alive, scans, best margin]
        self.slips = d.get("slips", [])        # [kind, legs, planned margin, real margin] from leg-risk simulation
        self.held = d.get("held", {})          # event slug -> {"checked": ts, "markets": [...]} awaiting settlement
        self.stats = d.get("stats", {"scans": 0, "events": 0, "candidates": 0, "confirmed": 0, "by_kind": {}, "by_tag": {}})
        self.offset = d.get("offset", 0)
        self.live = []
        self.marks = {}

    def to_dict(self):
        return {"account": self.acct.to_dict(), "lab": self.lab.to_dict(), "seen": self.seen,
                "lifetimes": self.lifetimes[-3000:], "slips": self.slips[-3000:], "held": self.held,
                "stats": self.stats, "offset": self.offset}

    # ------------------------------------------------------------------ finding candidates
    def candidates(self, evs):
        t = K.now()
        out = []
        for ev in evs:
            if ev["closed"] or not ev["markets"]:
                continue
            open_m = [m for m in ev["markets"] if not m["closed"] and m["active"]]
            if any(G.resolved_yes(m) for m in ev["markets"]):
                continue
            end = max([m["end"] or 0 for m in open_m] + [ev["end"] or 0]) or None
            days = ((end - t) / 86400) if end else 999
            if days < 0.02:
                continue
            if ev["neg_risk"] and not ev["augmented"] and len(open_m) >= 2 and all(m["accepting"] for m in open_m):
                ya = [m["best_ask"] for m in open_m]
                yb = [m["best_bid"] for m in open_m]
                if all(a is not None for a in ya) and sum(ya) < 0.998:
                    out.append(self._cand(ev, "yes_set", [(m, "YES") for m in open_m], 1.0, days, sum(ya)))
                if all(b is not None for b in yb) and sum(1 - b for b in yb) < len(open_m) - 1 - 0.002:
                    out.append(self._cand(ev, "no_set", [(m, "NO") for m in open_m], len(open_m) - 1.0, days,
                                          sum(1 - b for b in yb)))
            elif not ev["neg_risk"] and len(open_m) >= 2:
                out += self._ladders(ev, open_m, days)
        return out

    def _cand(self, ev, kind, legs, payout, days, rough):
        sig = ",".join(sorted(m["id"] for m, _ in legs))
        return {"key": f"{ev['slug']}|{kind}|{sig}" if kind == "ladder" else f"{ev['slug']}|{kind}", "kind": kind,
                "event": ev["slug"], "title": ev["title"], "legs": legs, "payout": payout, "days": days,
                "rough": rough, "tags": ev["tags"][:4]}

    def _ladders(self, ev, open_m, days):
        groups = {}
        for m in open_m:
            lv = _level(m["question"])
            if lv and m["accepting"]:
                groups.setdefault((lv[0], lv[2]), []).append((lv[1], m))
        out = []
        for (direction, _), xs in groups.items():
            xs.sort(key=lambda x: x[0])
            for i in range(len(xs) - 1):
                (a, ma), (b, mb) = xs[i], xs[i + 1]
                if a == b or abs((ma["end"] or 0) - (mb["end"] or 0)) > 3600:
                    continue
                # above: YES(lower) + NO(higher);  below: YES(higher) + NO(lower)
                yes_m, no_m = (ma, mb) if direction == "up" else (mb, ma)
                if yes_m["best_ask"] is None or no_m["best_bid"] is None:
                    continue
                rough = yes_m["best_ask"] + (1 - no_m["best_bid"])
                if rough < 0.998:
                    out.append(self._cand(ev, "ladder", [(yes_m, "YES"), (no_m, "NO")], 1.0, days, rough))
        return out

    # ------------------------------------------------------------------ pricing against real books
    @staticmethod
    def _tok(m, side):
        return m["yes_token"] if side == "YES" else m["no_token"]

    def price(self, c, books, take):
        """Best size for a set on these books: shares, cost, profit, margin per set. None if not profitable."""
        best = None
        mins = max(m["min_size"] for m, _ in c["legs"])
        for q in SIZES:
            if q < mins:
                continue
            cost, worst = 0.0, []
            for m, side in c["legs"]:
                bk = books.get(self._tok(m, side))
                if not bk:
                    return best
                r = K.walk_cost(bk["asks"], q, take, m["fee_rate"], m["fee_exp"])
                if not r:
                    cost = None
                    break
                cost += r["cost"]
                worst.append(r["worst"])
            if cost is None:
                break
            profit = q * c["payout"] - cost
            margin = profit / q
            if margin <= 0:
                break
            if not best or profit > best["profit"]:
                best = {"shares": q, "cost": round(cost, 4), "profit": round(profit, 4), "margin": round(margin, 5),
                        "worst": worst}
        return best

    # ------------------------------------------------------------------ the scan
    def scan(self, evs):
        t = K.now()
        self.stats["scans"] += 1
        self.stats["events"] += len(evs)
        cands = self.candidates(evs)
        self.stats["candidates"] += len(cands)
        take = self.s["liquidity_take_pct"]
        books = G.books([self._tok(m, s) for c in cands for m, s in c["legs"]]) if cands else {}
        found, self.live = set(), []
        slip_need = self.learned_slip()
        for c in cands:
            plan = self.price(c, books, take)
            if not plan:
                continue
            found.add(c["key"])
            self.stats["confirmed"] += 1
            bk = self.stats["by_kind"].setdefault(c["kind"], 0)
            self.stats["by_kind"][c["kind"]] = bk + 1
            for tg in c["tags"]:
                self.stats["by_tag"][tg] = self.stats["by_tag"].get(tg, 0) + 1
            o = self.seen.setdefault(c["key"], {"first": t, "kind": c["kind"], "title": c["title"], "scans": 0, "best": 0,
                                                "legs": len(c["legs"])})
            o["last"], o["scans"], o["best"] = t, o["scans"] + 1, max(o["best"], plan["margin"])
            self.live.append({"kind": c["kind"], "title": c["title"], "event": c["event"], "legs": len(c["legs"]),
                              "margin": plan["margin"], "shares": plan["shares"], "profit": plan["profit"],
                              "days": round(c["days"], 2), "payout": c["payout"], "age_min": round((t - o["first"]) / 60, 1)})
            takers = [v for v in self.lab.active() if self._wants(v["params"], c, plan, slip_need)
                      and not self.lab.holds(v["id"], c["key"])]
            if not takers:
                continue
            real = self._execute_sim(c, plan, books, take)
            for v in takers:
                legs = [{"market": m["id"], "side": s, "shares": plan["shares"]} for m, s in c["legs"]]
                self.lab.vopen(v["id"], c["key"], real["cost"], legs,
                               {"kind": c["kind"], "title": c["title"], "planned": plan["margin"], "real": real["margin"]})
                if v["id"] == self.lab.champion:
                    self._champ(c, plan, real)
                self.held.setdefault(c["event"], {"checked": 0})
        # opportunities that disappeared: how long did they live?
        for k, o in list(self.seen.items()):
            if k not in found and t - o["last"] > 30:
                self.lifetimes.append([o["kind"], round((o["last"] - o["first"]) / 60, 2), o["scans"], round(o["best"], 5), o["legs"]])
                self.seen.pop(k)
        self.live.sort(key=lambda x: -x["profit"])
        self.live = self.live[:40]

    def _wants(self, P, c, plan, slip_need):
        if P["kinds"] == "sets" and c["kind"] == "ladder":
            return False
        if P["kinds"] not in ("all", "sets") and P["kinds"] != c["kind"]:
            return False
        if c["days"] > P["max_days"] or len(c["legs"]) > P["max_legs"]:
            return False
        need = P["min_margin"] + P["haircut"] * slip_need.get(c["kind"], 0.0) * max(1, len(c["legs"]) - 1)
        return plan["margin"] >= need and plan["profit"] >= P["min_profit"]

    def _execute_sim(self, c, plan, books, take):
        """Leg risk: the first leg fills on the book we saw; the others on a fresh book read a moment later,
        exactly as a real bot sending orders one after another would experience it."""
        q = plan["shares"]
        m0, s0 = c["legs"][0]
        r0 = K.walk_cost(books[self._tok(m0, s0)]["asks"], q, take, m0["fee_rate"], m0["fee_exp"])
        fresh = G.books([self._tok(m, s) for m, s in c["legs"][1:]])
        cost, missing = r0["cost"], 0
        for m, s in c["legs"][1:]:
            bk = fresh.get(self._tok(m, s)) or books.get(self._tok(m, s))
            r = K.walk_cost(bk["asks"], q, take, m["fee_rate"], m["fee_exp"]) if bk else None
            if not r:   # this leg can no longer be filled in full: buy at the worst price still shown, or 99¢
                missing += 1
                r = {"cost": q * min(0.99, (bk["asks"][-1][0] if bk and bk["asks"] else 0.99) + 0.02)}
            cost += r["cost"]
        margin = (q * c["payout"] - cost) / q
        self.slips.append([c["kind"], len(c["legs"]), plan["margin"], round(margin, 5), missing])
        return {"cost": round(cost, 4), "margin": round(margin, 5), "missing": missing,
                "status": "clean" if margin >= plan["margin"] - 1e-4 else ("slipped" if margin > 0 else "broken")}

    def learned_slip(self):
        """Average margin lost per extra leg between seeing a gap and finishing the last leg, per kind."""
        out = {}
        for kind in ("yes_set", "no_set", "ladder"):
            xs = [(p - r) / max(1, n - 1) for k, n, p, r, _ in self.slips[-400:] if k == kind]
            if len(xs) >= 5:
                out[kind] = max(0.0, statistics.mean(xs))
        return out

    def _champ(self, c, plan, real):
        a = self.acct
        if a.day_loss_hit(self.s["daily_loss_limit_pct"]) or any(p["key"] == c["key"] for p in a.positions):
            return
        eq = a.equity_cost()
        cap = min(self.s["arb_max_set_pct"] * eq, a.cash * 0.95) * self.lab.size_factor()
        if c["days"] > self.s["arb_max_days"]:
            return
        scale = min(1.0, cap / real["cost"]) if real["cost"] > 0 else 0
        shares = int(plan["shares"] * scale * 100) / 100
        if shares < max(m["min_size"] for m, _ in c["legs"]):
            return
        cost = real["cost"] * shares / plan["shares"]
        pos = {"id": a.nid("A"), "key": c["key"], "desk": "arb", "kind": c["kind"], "event": c["event"], "title": c["title"],
               "shares": shares, "cost": round(cost, 4), "payout_min": round(shares * c["payout"], 4),
               "planned_margin": plan["margin"], "real_margin": real["margin"], "fill": real["status"],
               "days": round(c["days"], 2), "strategy": self.lab.champion,
               "legs": [{"market": m["id"], "token": self._tok(m, s), "side": s, "shares": shares,
                         "label": (m["group"] or m["question"])[:60]} for m, s in c["legs"]]}
        a.open(pos)
        names = {"yes_set": "every YES", "no_set": "every NO", "ladder": "a YES/NO ladder pair"}
        self.j.activity(f"ARB {c['title'][:70]}: bought {names[c['kind']]} ({len(c['legs'])} legs) x{shares:g} for "
                        f"${cost:.2f}, guaranteed ${shares * c['payout']:.2f} "
                        f"(planned {plan['margin']*100:.2f}¢/set, after leg risk {real['margin']*100:.2f}¢: {real['status']})",
                        kind="trade")

    # ------------------------------------------------------------------ settlement
    def resolve(self):
        t = K.now()
        for slug, h in list(self.held.items()):
            if t - h.get("checked", 0) < 900:
                continue
            h["checked"] = t
            try:
                ev = G.event_by_slug(slug)
            except K.FetchError:
                continue
            if not ev:
                continue
            res = {m["id"]: G.resolved_yes(m) for m in ev["markets"]}
            # a negRisk market that closed at NO before the event ended still counts as known
            known = {k: v for k, v in res.items() if v is not None}
            self.lab.settle(known)
            for pos in list(self.acct.positions):
                if pos["event"] != slug:
                    continue
                if any(known.get(leg["market"]) is None for leg in pos["legs"]):
                    continue
                payout = sum(leg["shares"] for leg in pos["legs"] if (leg["side"] == "YES") == known[leg["market"]])
                rec = self.acct.close(pos, payout, "settled")
                self.j.trade({"desk": "arb", "id": rec["id"], "title": rec["title"][:90], "pnl": rec["pnl"],
                              "won": rec["pnl"] > 0, "evidence": "measured",
                              "body": f"{rec['kind']} with {len(rec['legs'])} legs held {((t - rec['opened']) / 86400):.1f} days. "
                                      f"Paid ${payout:.2f} on ${rec['cost']:.2f}: {K.U.money(rec['pnl'])} "
                                      f"(planned {rec['planned_margin']*100:.2f}¢/set, fill {rec['fill']})."})
            still = any(p["event"] == slug for p in self.acct.positions) or any(
                any(leg["market"] in res and known.get(leg["market"]) is None for leg in p["legs"])
                for v in self.lab.variants.values() for p in v["open"].values())
            if not still:
                self.held.pop(slug, None)

    def mark(self):
        """Sets are worth at least what they are guaranteed to pay; valued at cost until settled (conservative)."""
        self.marks = {p["id"]: min(p["cost"], p["payout_min"]) for p in self.acct.positions}

    # ------------------------------------------------------------------ learning
    def learn(self):
        lt = self.lifetimes[-1500:]
        if len(lt) >= 10:
            mins = sorted(x[1] for x in lt)
            once = sum(1 for x in lt if x[2] <= 1) / len(lt)
            self.j.discovery("How long gaps last", f"Median arbitrage gap lived {mins[len(mins)//2]:.1f} min "
                             f"({len(lt)} gaps); {once:.0%} were gone by the next scan. Kinds: "
                             + ", ".join(f"{k} {sum(1 for x in lt if x[0] == k)}" for k in ("yes_set", "no_set", "ladder")),
                             "measured", key="arb-life")
        sl = self.learned_slip()
        if sl:
            self.j.discovery("Leg risk", "Average margin lost per extra leg while buying a set: " + ", ".join(
                f"{k} {v*100:.2f}¢" for k, v in sl.items()) + ". Strategy versions add this (x their haircut) to the "
                "margin they demand.", "measured", key="arb-slip")
        broken = [x for x in self.slips[-500:] if x[3] <= 0]
        if len(self.slips) >= 20:
            self.j.discovery("Broken sets", f"{len(broken)} of the last {min(500, len(self.slips))} simulated sets "
                             f"stopped being profitable before the last leg filled.", "measured", key="arb-broken")
        tags = sorted(self.stats["by_tag"].items(), key=lambda x: -x[1])[:8]
        if tags:
            self.j.discovery("Where gaps appear", "Confirmed gaps by category: " + ", ".join(f"{k} {v}" for k, v in tags),
                             "measured", key="arb-tags")
        for kind, text, evidence in self.lab.evolve():
            self.j.experiment({"desk": "arb", "kind": kind, "body": text, "evidence": evidence})
