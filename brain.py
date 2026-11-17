#!/usr/bin/env python3
"""Crypto Desk trader brain (v8) - learns day trading the way a person does.

A human learning to day trade keeps a small PLAYBOOK of setups (pullbacks, breakouts, opening-range
breaks, VWAP reclaims, stop-hunt reversals, momentum bursts). They practise each one on past charts,
only take it with real (here: paper) money once it has worked on charts they have not studied, start
small, size up when it keeps working, drop it when it stops working, and keep refining the rules.

This module holds:
  * the setups and their rule "genes" (thresholds a person would tweak: volume needed, stop width,
    target, holding time, trade only with the trend / with BTC / in certain sessions)
  * a fast trade simulator that uses the same exits and costs as the live engine
  * `evolve`: the practice session. For every setup and side it tries many rule versions on the older
    part of the data (study period), keeps the best and refines them, then checks the finalists on the
    most recent week it never studied (exam). Only versions that pass the exam go into the playbook.
Shorts are tested on mirrored prices (1/price), so every setup works both ways with the same code.
"""
import hashlib, json, math, random, time

BAR = 900000
H = 3600000
DAY = 24 * H
WARM = 100
SIDES = ["long", "short"]

COMMON = {"stop": (0.8, 3.0, 0.1), "rr": (1.0, 4.0, 0.25), "hold": (4, 48, 4), "be": (0, 1, 1),
          "trend": (0, 1, 1), "btc": (0, 1, 1), "sess": ["all", "asia", "eu", "us", "eu+us"]}
SPEC = {
    "pullback": {"rsiMax": (30, 65, 1), "touch": (1, 6, 1)},
    "breakout": {"look": (16, 96, 8), "vol": (1.0, 3.0, 0.1)},
    "orb": {"vol": (1.0, 3.0, 0.1), "maxH": (1, 8, 1)},
    "vwap": {"vol": (1.0, 2.5, 0.1), "below": (2, 12, 1)},
    "sweep": {"look": (16, 96, 8), "rsiMax": (20, 55, 1)},
    "momentum": {"bars": (2, 16, 1), "move": (1.0, 4.0, 0.25), "vol": (1.0, 2.5, 0.1)},
}
LABEL = {"pullback": ("Trend pullback", "Rally fade in a downtrend"), "breakout": ("Breakout", "Breakdown"),
         "orb": ("Opening-range breakout", "Opening-range breakdown"), "vwap": ("VWAP reclaim", "VWAP loss"),
         "sweep": ("Stop-hunt reversal at the lows", "Stop-hunt reversal at the highs"),
         "momentum": ("Momentum burst", "Downside momentum burst")}
SESS_TXT = {"all": "", "asia": " during Asia hours (08-16 SGT)", "eu": " during Europe hours (16-24 SGT)",
            "us": " during US hours (00-08 SGT)", "eu+us": " during Europe and US hours (16-08 SGT)"}

# exam: what a rule version must show on the unseen week before it is traded
GATE = {"trainN": 30, "trainR": 0.05, "valN": 8, "valR": 0.08, "valPF": 1.2, "tstat": 2.0}


def label(setup, side):
    return LABEL[setup][0 if side == "long" else 1]


# ---------------------------------------------------------------- indicators
def ema(x, n):
    k = 2 / (n + 1); e = x[0]; out = []
    for v in x:
        e = v * k + e * (1 - k); out.append(e)
    return out


def inverse(bars):
    """Mirror prices so a short looks like a long: 1/price, high and low swap."""
    return [[b[0], 1 / b[1], 1 / b[3], 1 / b[2], 1 / b[4], b[5]] for b in bars if b[1] > 0 and b[3] > 0]


def sgt_hour(t):
    return ((t // H) + 8) % 24


class Series:
    def __init__(self, bars, btc_up=None):
        self.t = [b[0] for b in bars]; self.o = [b[1] for b in bars]; self.h = [b[2] for b in bars]
        self.l = [b[3] for b in bars]; self.c = [b[4] for b in bars]; self.v = [b[5] for b in bars]
        n = len(bars); c, h, l = self.c, self.h, self.l
        self.n = n
        a = [0.0] * n; tr0 = []
        for i in range(1, n):
            tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
            if i <= 14:
                tr0.append(tr); a[i] = sum(tr0) / len(tr0)
            else:
                a[i] = (a[i - 1] * 13 + tr) / 14
        a[0] = a[1] if n > 1 else 0.0
        self.atr = a
        self.e20 = ema(c, 20); self.eT = ema(c, 80)
        self.up = [i >= 8 and c[i] > self.eT[i] and self.eT[i] > self.eT[i - 8] for i in range(n)]
        r = [50.0] * n; ag = al = 0.0
        for i in range(1, n):
            d = c[i] - c[i - 1]; g, lo = max(d, 0), max(-d, 0)
            if i <= 14:
                ag += g / 14; al += lo / 14
            else:
                ag = (ag * 13 + g) / 14; al = (al * 13 + lo) / 14
            if i >= 14: r[i] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
        self.rsi = r
        vr = [1.0] * n; s = 0.0
        for i in range(n):
            if i >= 20:
                vr[i] = self.v[i] / (s / 20) if s > 0 else 1.0
                s -= self.v[i - 20]
            s += self.v[i]
        self.vr = vr
        vw = [0.0] * n; orh = [None] * n; di = [0] * n
        day = None; pv = vv = 0.0; k = 0; hi = lo_ = None
        for i in range(n):
            d = self.t[i] // DAY
            if d != day:
                day, pv, vv, k, hi, lo_ = d, 0.0, 0.0, 0, None, None
            tp = (h[i] + l[i] + c[i]) / 3
            pv += tp * self.v[i]; vv += self.v[i]
            vw[i] = pv / vv if vv > 0 else c[i]
            if k < 4:
                hi = h[i] if hi is None else max(hi, h[i])
            di[i] = k
            orh[i] = hi if k >= 4 else None
            k += 1
        self.vw, self.orh, self.di = vw, orh, di
        self.hr = [sgt_hour(t) for t in self.t]
        self.btc = btc_up if btc_up is not None else [True] * n
        self._c = {}

    def cache(self, key, fn):
        if key not in self._c: self._c[key] = fn()
        return self._c[key]

    def hh(self, k):
        return self.cache(("hh", k), lambda: [max(self.h[i - k:i]) if i >= k else float("inf") for i in range(self.n)])

    def ll(self, k):
        return self.cache(("ll", k), lambda: [min(self.l[i - k:i]) if i >= k else 0.0 for i in range(self.n)])

    def minlow(self, k):
        return self.cache(("ml", k), lambda: [min(self.l[max(0, i - k + 1):i + 1]) for i in range(self.n)])

    def below(self, k):
        def f():
            b = [1 if self.c[i] < self.vw[i] else 0 for i in range(self.n)]
            return [i >= k and sum(b[i - k:i]) > 0 for i in range(self.n)]
        return self.cache(("bw", k), f)

    def mom(self, k):
        return self.cache(("mo", k), lambda: [(self.c[i] - self.c[i - k]) / self.atr[i] if i >= k and self.atr[i] > 0 else 0.0
                                              for i in range(self.n)])

    def allowed(self, trend, btc, sess):
        def f():
            out = []
            for i in range(WARM, self.n):
                if self.atr[i] <= 0: continue
                if trend and not self.up[i]: continue
                if btc and not self.btc[i]: continue
                hr = self.hr[i]
                if sess == "asia" and not (8 <= hr < 16): continue
                if sess == "eu" and not (hr >= 16): continue
                if sess == "us" and not (hr < 8): continue
                if sess == "eu+us" and not (hr >= 16 or hr < 8): continue
                out.append(i)
            return out
        return self.cache(("al", trend, btc, sess), f)


def btc_up_map(btc_series):
    return {t: u for t, u in zip(btc_series.t, btc_series.up)}


def make_series(bars, btc_bars, side):
    """Series for one coin seen from one side (short = mirrored prices), with the BTC filter aligned."""
    if side == "short":
        bars = inverse(bars); btc_bars = inverse(btc_bars) if btc_bars else None
    if btc_bars:
        m = btc_up_map(Series(btc_bars))
        return Series(bars, [m.get(t, False) for t in (b[0] for b in bars)])
    return Series(bars)


# ---------------------------------------------------------------- rules
def cond(S, setup, g):
    c, o = S.c, S.o
    vr, rsi = S.vr, S.rsi
    if setup == "pullback":
        e, ml, rm = S.e20, S.minlow(int(g["touch"])), g["rsiMax"]
        return lambda i: c[i] > e[i] and c[i] > o[i] and rsi[i] <= rm and ml[i] <= e[i]
    if setup == "breakout":
        hh, vol = S.hh(int(g["look"])), g["vol"]
        return lambda i: c[i] > hh[i] and vr[i] >= vol
    if setup == "orb":
        orh, di, vol, mx = S.orh, S.di, g["vol"], g["maxH"] * 4
        return lambda i: orh[i] is not None and di[i] <= mx and c[i] > orh[i] and c[i - 1] <= orh[i] and vr[i] >= vol
    if setup == "vwap":
        vw, bw, vol = S.vw, S.below(int(g["below"])), g["vol"]
        return lambda i: c[i] > vw[i] and c[i] > o[i] and vr[i] >= vol and bw[i]
    if setup == "sweep":
        ll, l, rm = S.ll(int(g["look"])), S.l, g["rsiMax"]
        return lambda i: l[i] < ll[i] and c[i] > ll[i] and rsi[i] <= rm
    if setup == "momentum":
        mo, mv, vol = S.mom(int(g["bars"])), g["move"], g["vol"]
        return lambda i: mo[i] >= mv and vr[i] >= vol and c[i] > o[i]
    raise ValueError(setup)


def base(S, setup, g):
    return S.allowed(1 if (g["trend"] or setup == "pullback") else 0, int(g["btc"]), g["sess"])


def signal_at(S, i, setup, g):
    """Does the rule fire on bar i (a completed bar)? Same logic as the practice simulator."""
    if i < WARM or i >= S.n or S.atr[i] <= 0: return False
    if i not in set(base(S, setup, g)): return False
    return cond(S, setup, g)(i)


def simulate(S, i, g, fee, slip):
    """One trade from the close of bar i, exits exactly like the live engine. Returns (net R, exit bar) or None."""
    c, h, l, o = S.c, S.h, S.l, S.o
    entry = c[i] * (1 + slip)
    risk = g["stop"] * S.atr[i]
    if risk <= 0: return None
    stop = entry - risk; tgt = entry + g["rr"] * risk; be_lvl = entry + risk; moved = False
    last = i + int(g["hold"])
    if last >= S.n: return None
    ex = None; j = i
    for j in range(i + 1, last + 1):
        if l[j] <= stop:
            ex = min(o[j], stop); break
        if h[j] >= tgt:
            ex = tgt; break
        if g["be"] and not moved and h[j] >= be_lvl:
            stop = max(stop, entry * (1 + 2 * fee)); moved = True
    if ex is None: ex = c[last]; j = last
    ex *= (1 - slip)
    return ((ex - entry) - fee * (entry + ex)) / risk, j


def stats(rs):
    n = len(rs)
    if not n: return {"n": 0, "avgR": 0.0, "pf": 0.0, "win": 0.0, "sumR": 0.0}
    gw = sum(r for r in rs if r > 0); gl = -sum(r for r in rs if r <= 0)
    return {"n": n, "avgR": round(sum(rs) / n, 3), "pf": round(gw / gl, 2) if gl > 0 else 9.99,
            "win": round(sum(1 for r in rs if r > 0) / n, 3), "sumR": round(sum(rs), 2)}


def evaluate(series, setup, g, split, fee, slip, frm=None):
    tr, va = [], []
    for S in series:
        f = cond(S, setup, g); busy = -1
        for i in base(S, setup, g):
            if i <= busy: continue
            if frm is not None and S.t[i] < frm: continue
            if not f(i): continue
            r = simulate(S, i, g, fee, slip)
            if r is None: continue
            (tr if S.t[i] < split else va).append((S.t[i], r[0])); busy = r[1]
    tr.sort(); mid = len(tr) // 2
    a = stats([r for _, r in tr]); a["h1"] = stats([r for _, r in tr[:mid]])["avgR"]; a["h2"] = stats([r for _, r in tr[mid:]])["avgR"]
    allr = [r for _, r in tr] + [r for _, r in va]
    if len(allr) > 2:
        m = sum(allr) / len(allr); sd = math.sqrt(sum((x - m) ** 2 for x in allr) / (len(allr) - 1)) or 1e-9
        a["t"] = round(m / sd * math.sqrt(len(allr)), 2)
    else:
        a["t"] = 0.0
    return a, stats([r for _, r in va])


# ---------------------------------------------------------------- genes
def grid(lo, hi, st):
    n = int(round((hi - lo) / st))
    return [round(lo + k * st, 4) for k in range(n + 1)]


def genes_spec(setup):
    sp = dict(COMMON); sp.update(SPEC[setup]); return sp


def rand_genes(setup, rnd):
    g = {}
    for k, s in genes_spec(setup).items():
        g[k] = rnd.choice(s) if isinstance(s, list) else rnd.choice(grid(*s))
    if setup == "pullback": g["trend"] = 1
    return g


def mutate(g0, setup, rnd):
    g = dict(g0); sp = genes_spec(setup)
    for k in rnd.sample(list(sp), rnd.randint(1, 3)):
        s = sp[k]
        if isinstance(s, list):
            g[k] = rnd.choice(s)
        else:
            vals = grid(*s); cur = min(range(len(vals)), key=lambda x: abs(vals[x] - g.get(k, vals[0])))
            g[k] = vals[max(0, min(len(vals) - 1, cur + rnd.choice([-3, -2, -1, 1, 2, 3])))]
    if setup == "pullback": g["trend"] = 1
    return g


def ver(g):
    return hashlib.sha1(json.dumps(g, sort_keys=True).encode()).hexdigest()[:8]


def fit(st):
    if st["n"] < GATE["trainN"]: return -9 + st["n"] / 100
    return st["avgR"] * min(1.0, st["n"] / 40) - 0.25 / math.sqrt(st["n"])


def passes(tr, va):
    """The exam: enough practice trades, profitable in BOTH halves of the study period, profitable on the
    unseen weeks, and a t-stat of 2+ over all of it (unlikely to be luck)."""
    return (tr["n"] >= GATE["trainN"] and tr["avgR"] >= GATE["trainR"] and tr.get("h1", 0) > 0 and tr.get("h2", 0) > 0
            and va["n"] >= GATE["valN"] and va["avgR"] >= GATE["valR"] and va["pf"] >= GATE["valPF"]
            and tr.get("t", 0) >= GATE["tstat"])


def describe(setup, side, g):
    L = side == "long"
    buy = "Buy" if L else "Sell short"
    up, dn = ("up", "down") if L else ("down", "up")
    hi, lo = ("high", "low") if L else ("low", "high")
    above = "above" if L else "below"
    if setup == "pullback":
        core = (f"{buy} when the 1h trend is {up}, price dips back to the 20-bar average within the last {int(g['touch'])} bars, "
                f"RSI is {'at most ' + str(int(g['rsiMax'])) if L else 'at least ' + str(100 - int(g['rsiMax']))} and the candle closes back {above} the average")
    elif setup == "breakout":
        core = f"{buy} when price closes {above} the {int(g['look'])}-bar ({int(g['look']) / 4:g}h) {hi} on {g['vol']:g}x normal volume"
    elif setup == "orb":
        core = (f"{buy} when price breaks {above} the first hour's {hi} (opening range, 08:00 SGT) within {int(g['maxH'])}h of the open "
                f"on {g['vol']:g}x normal volume")
    elif setup == "vwap":
        core = (f"{buy} when price was {'below' if L else 'above'} VWAP in the last {int(g['below'])} bars and closes back {above} it "
                f"on {g['vol']:g}x normal volume")
    elif setup == "sweep":
        core = (f"{buy} when price pokes {'under' if L else 'over'} the {int(g['look'])}-bar {lo} (stop hunt) and closes back inside, "
                f"RSI {'at most ' + str(int(g['rsiMax'])) if L else 'at least ' + str(100 - int(g['rsiMax']))}")
    else:
        core = f"{buy} when price has moved {g['move']:g} ATR {up} in {int(g['bars'])} bars on {g['vol']:g}x volume"
    filt = []
    if g["trend"] and setup != "pullback": filt.append(f"only with the 1h trend {up}")
    if g["btc"]: filt.append(f"only when BTC is trending {up}")
    s = core + (", " + ", ".join(filt) if filt else "") + SESS_TXT.get(g["sess"], "") + "."
    s += (f" Stop {g['stop']:g} ATR away, target {g['rr']:g}R"
          + (", stop to breakeven at +1R" if g["be"] else "") + f", exit after {int(g['hold']) / 4:g}h if neither is hit.")
    return s


# ---------------------------------------------------------------- practice session
def evolve(data, prev=None, fee=0.001, slip=0.001, val_days=14, seed=None, budget_s=1500, pop=60, gens=4, kids=6, log=print):
    """data: {coin: 15m bars}. prev: the previous playbook (dict) or None. Returns the new playbook."""
    t_start = time.time()
    rnd = random.Random(seed if seed is not None else int(t_start))
    btc = data.get("BTC")
    end = max(rows[-1][0] for rows in data.values()) + BAR
    split = end - val_days * DAY
    series = {side: [make_series(rows, btc, side) for c, rows in sorted(data.items()) if len(rows) > WARM + 50] for side in SIDES}
    prev_by = {s["id"]: s for s in (prev or {}).get("setups", [])}
    out, notes, tested = [], [], 0
    now = int(time.time() * 1000)
    order = [(s, d) for s in SPEC for d in SIDES]
    per = budget_s / len(order)
    for setup, side in order:
        sid = f"{setup}:{side}"; lab = label(setup, side); t0 = time.time()
        old = prev_by.get(sid)
        cands = [old["genes"]] if old and old.get("genes") else []
        cands += [rand_genes(setup, rnd) for _ in range(pop)]
        seen, scored = set(), []
        def run(gs):
            nonlocal tested
            for g in gs:
                v = ver(g)
                if v in seen: continue
                seen.add(v)
                tr, va = evaluate(series[side], setup, g, split, fee, slip)
                scored.append((fit(tr), g, tr, va)); tested += 1
        run(cands)
        for _ in range(gens):
            if time.time() - t0 > per: break
            top = sorted(scored, key=lambda x: -x[0])[:8]
            run([mutate(g, setup, rnd) for _, g, _, _ in top for _ in range(kids)])
        ranked = sorted(scored, key=lambda x: -x[0])
        finalists = ranked[:3]  # judged on the study period only; the exam can only veto
        ok = [x for x in finalists if passes(x[2], x[3])]
        best = ok[0] if ok else None
        old_eval = None
        if old and old.get("genes"):
            old_eval = next(((x[2], x[3]) for x in scored if ver(x[1]) == ver(old["genes"])), None)
        entry = {"id": sid, "setup": setup, "side": side, "label": lab, "tested": len(seen)}
        if best:
            keep_old = (old_eval and old.get("enabled") and passes(*old_eval)
                        and min(old_eval[0]["avgR"], old_eval[1]["avgR"]) >= min(best[2]["avgR"], best[3]["avgR"]) - 0.02)
            if keep_old:
                g, tr, va = old["genes"], old_eval[0], old_eval[1]
                note = (f"{lab} ({side}): still works. Re-checked the current rules: unseen week {va['n']} trades, "
                        f"{va['avgR']:+.2f}R each. Keeping them.")
            else:
                g, tr, va = best[1], best[2], best[3]
                note = (f"{lab} ({side}): {'found better rules' if old and old.get('enabled') else 'passed the exam'}. Practised {len(seen)} versions; "
                        f"the best made {tr['avgR']:+.2f}R per trade over {tr['n']} practice trades and {va['avgR']:+.2f}R over "
                        f"{va['n']} trades in the unseen week (win rate {va['win']:.0%}). "
                        + ("Updated the playbook." if old and old.get("enabled") else "Added to the playbook: it starts live at half size."))
            entry.update({"enabled": True, "genes": g, "ver": ver(g), "train": tr, "val": va, "plain": describe(setup, side, g)})
        elif old_eval and old.get("enabled") and old_eval[1]["n"] >= 4 and old_eval[1]["avgR"] > 0:
            g, tr, va = old["genes"], old_eval[0], old_eval[1]
            note = (f"{lab} ({side}): no new version passed, but the current rules are still positive on the latest week "
                    f"({va['n']} trades, {va['avgR']:+.2f}R each), so they stay.")
            entry.update({"enabled": True, "genes": g, "ver": ver(g), "train": tr, "val": va, "plain": describe(setup, side, g)})
        else:
            top = ranked[0]
            why = (f"best practice version made {top[2]['avgR']:+.2f}R over {top[2]['n']} trades"
                   + (f" but {top[3]['avgR']:+.2f}R in the unseen week ({top[3]['n']} trades)" if top[2]['n'] else ""))
            note = (f"{lab} ({side}): {'dropped from the playbook: it stopped working' if old and old.get('enabled') else 'not ready'}. "
                    f"Practised {len(seen)} versions; {why}. Keeps practising.")
            entry.update({"enabled": False, "genes": top[1], "ver": ver(top[1]), "train": top[2], "val": top[3],
                          "plain": describe(setup, side, top[1])})
        entry["since"] = old["since"] if old and old.get("ver") == entry["ver"] and old.get("since") else now
        out.append(entry); notes.append({"t": now, "text": note})
        log(note)
    live = [s for s in out if s["enabled"]]
    summary = (f"Practice session: {tested} rule versions tested on {len(series['long'])} coins, "
               f"{len(live)} of {len(out)} setups in the playbook"
               + (f" ({', '.join(s['label'] + ' ' + s['side'] for s in live)})." if live else " - none passed the exam this time, so it sits out and keeps practising."))
    notes.insert(0, {"t": now, "text": summary})
    hist = (prev or {}).get("notes", [])
    return {"generated": now, "from": min(rows[0][0] for rows in data.values()), "to": end, "split": split,
            "coins": sorted(data), "gate": GATE, "fee": fee, "slip": slip, "tested": tested,
            "seconds": round(time.time() - t_start, 1), "summary": summary, "setups": out,
            "notes": (notes + hist)[:120]}
