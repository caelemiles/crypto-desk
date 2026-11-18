"""Strategy lab: every strategy version trades every opportunity in the shadows (virtual money, the real order
book), so all versions are compared on exactly the same markets. The champion's trades go to the paper account.

- A challenger is promoted only when, over the same period, it beat the champion on profit AND return per dollar,
  with enough settled trades and a bootstrap confidence interval that keeps its return above zero.
- New challengers are born by changing one or two of the champion's rules (or from a discovery the desk made).
- Weak challengers are retired; a champion that keeps losing is put on reduced size (stand-down).
"""
import random

import core as K


def boot_lower(xs, n=1500, alpha=0.1, seed=7):
    """Lower bound of a bootstrap CI for the mean of xs."""
    if len(xs) < 5:
        return None
    rnd = random.Random(seed)
    ms = sorted(sum(rnd.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return ms[int(alpha / 2 * n)]


class Lab:
    def __init__(self, d, desk, seed_params, space, journal, settings):
        d = d or {}
        self.desk, self.space, self.j, self.settings = desk, space, journal, settings
        self.variants = d.get("variants", {})
        self.champion = d.get("champion")
        self.seq = d.get("seq", 0)
        self.rnd = random.Random(d.get("seq", 0) + 99)
        if not self.variants:
            vid = self.add(seed_params, note="starting rules", status="champion")
            self.champion = vid
        for v in self.variants.values():   # keep stored versions compatible when new rules are added
            for k, val in seed_params.items():
                v["params"].setdefault(k, val)

    def to_dict(self):
        return {"variants": self.variants, "champion": self.champion, "seq": self.seq}

    # ------------------------------------------------------------ versions
    def add(self, params, note="", parent=None, status="challenger"):
        self.seq += 1
        vid = f"{self.desk[0].upper()}{self.seq}"
        self.variants[vid] = {"id": vid, "params": dict(params), "status": status, "born": K.now(), "parent": parent,
                              "note": note, "open": {}, "settled": []}
        return vid

    def active(self):
        return [v for v in self.variants.values() if v["status"] in ("champion", "challenger")]

    def champ(self):
        return self.variants[self.champion]

    def holds(self, vid, key):
        return key in self.variants[vid]["open"]

    def vopen(self, vid, key, cost, legs, meta=None):
        self.variants[vid]["open"][key] = {"cost": round(cost, 5), "legs": legs, "meta": meta or {}, "ts": K.now()}

    def vclose(self, vid, key, proceeds, how):
        v = self.variants[vid]
        p = v["open"].pop(key, None)
        if p:
            v["settled"].append({"key": key, "cost": p["cost"], "pnl": round(proceeds - p["cost"], 5), "ts": K.now(),
                                 "how": how, "meta": p["meta"]})
            del v["settled"][:-700]

    def settle(self, results):
        """results: {market id: True/False (YES won)}. A virtual position settles once all its legs are known."""
        n = 0
        for v in self.variants.values():
            for key, p in list(v["open"].items()):
                payout, done = 0.0, True
                for leg in p["legs"]:
                    r = results.get(leg["market"])
                    if r is None:
                        done = False
                        break
                    if (leg["side"] == "YES") == r:
                        payout += leg["shares"]
                if done:
                    self.vclose(v["id"], key, payout, "settled")
                    n += 1
        return n

    # ------------------------------------------------------------ scoring
    @staticmethod
    def score(settled, since=0):
        s = [x for x in settled if x["ts"] >= since]
        cost = sum(x["cost"] for x in s)
        pnl = sum(x["pnl"] for x in s)
        rets = [x["pnl"] / x["cost"] for x in s if x["cost"] > 0]
        return {"n": len(s), "pnl": round(pnl, 2), "roi": (pnl / cost) if cost else None,
                "win_rate": (sum(1 for x in s if x["pnl"] > 0) / len(s)) if s else None,
                "roi_low": boot_lower(rets)}

    def table(self):
        rows = []
        for v in sorted(self.variants.values(), key=lambda v: (v["status"] != "champion", v["status"] == "retired", -v["born"])):
            rows.append({"id": v["id"], "status": v["status"], "params": v["params"], "note": v["note"],
                         "parent": v["parent"], "born": v["born"], "open": len(v["open"]),
                         "score": self.score(v["settled"]), "score_7d": self.score(v["settled"], K.now() - 7 * 86400)})
        return rows[:60]

    # ------------------------------------------------------------ evolution
    def mutate(self, base=None, k=None, note=None):
        base = dict(base or self.champ()["params"])
        keys = [x for x in self.space if not isinstance(self.space[x], str)]
        changed = []
        for key in self.rnd.sample(keys, k or self.rnd.choice([1, 1, 2])):
            opts = self.space[key]
            cur = base.get(key)
            choices = [o for o in opts if o != cur] or opts
            if cur in opts and len(opts) > 2:   # prefer neighbouring values: small steps
                i = opts.index(cur)
                near = [opts[j] for j in (i - 1, i + 1) if 0 <= j < len(opts)]
                choices = near if self.rnd.random() < 0.75 else choices
            base[key] = self.rnd.choice(choices)
            changed.append(f"{key} {cur} -> {base[key]}")
        if any(v["params"] == base for v in self.active()):
            return None
        vid = self.add(base, note=note or ("try " + "; ".join(changed)), parent=self.champion)
        return vid

    def evolve(self):
        """Promote, retire and breed. Returns a list of journal-ready events."""
        s, ev = self.settings, []
        need = s["min_trades_for_promotion"]
        ch = self.champ()
        best = None
        for v in self.active():
            if v["status"] != "challenger":
                continue
            mine = self.score(v["settled"])
            theirs = self.score(ch["settled"], since=v["born"])
            if mine["n"] >= need and mine["roi"] is not None and mine["roi_low"] is not None and mine["roi_low"] > 0 \
                    and mine["pnl"] > theirs["pnl"] and mine["roi"] > (theirs["roi"] or -1) + 0.02:
                if not best or mine["pnl"] - theirs["pnl"] > best[1]:
                    best = (v, mine["pnl"] - theirs["pnl"], mine, theirs)
            elif mine["n"] >= 2 * need and (mine["roi"] or 0) < min(0.0, (theirs["roi"] or 0) - 0.03):
                v["status"] = "retired"
                ev.append(("retired", f"Retired {v['id']} ({v['note']}): ROI {mine['roi']:+.1%} over {mine['n']} trades "
                                      f"vs champion {theirs['roi'] or 0:+.1%}", "measured"))
        if best:
            v, gain, mine, theirs = best
            ch["status"] = "retired"
            v["status"] = "champion"
            self.champion = v["id"]
            ev.append(("promoted", f"Promoted {v['id']} ({v['note']}) to champion: {mine['n']} trades, ROI {mine['roi']:+.1%} "
                                   f"(90% lower bound {mine['roi_low']:+.1%}) vs old champion {theirs['roi'] or 0:+.1%} "
                                   f"over the same period, {gain:+.2f} more profit", "measured"))
        live = [v for v in self.active() if v["status"] == "challenger"]
        while len(live) > s["max_challengers"]:
            worst = min(live, key=lambda v: (self.score(v["settled"])["roi"] or 0) if v["settled"] else 1)
            worst["status"] = "retired"
            live.remove(worst)
        while len(live) < s["max_challengers"]:
            vid = self.mutate()
            if not vid:
                if self.rnd.random() < 0.5:
                    break
                continue
            live.append(self.variants[vid])
            ev.append(("born", f"New challenger {vid}: {self.variants[vid]['note']}", "hypothesis"))
        # forget the details of long-retired versions so memory stays small
        for v in self.variants.values():
            if v["status"] == "retired":
                v["open"] = {}
                v["settled"] = v["settled"][-50:]
        old = sorted((v for v in self.variants.values() if v["status"] == "retired"), key=lambda v: v["born"])
        for v in old[:-80]:
            self.variants.pop(v["id"], None)
        return ev

    def size_factor(self):
        """Champion stand-down: half size after a losing fortnight, quarter size if that is significant."""
        sc = self.score(self.champ()["settled"], K.now() - 14 * 86400)
        if sc["n"] >= 20 and (sc["roi"] or 0) < 0:
            return 0.25 if sc["roi_low"] is not None and sc["roi"] < -0.05 else 0.5
        return 1.0
