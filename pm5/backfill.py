"""Backfill: gives the AI a history to learn from on day one, fills gaps after restarts, and keeps extending
the history further back a day at a time (up to MAX_DAYS).

Outcomes come from Polymarket's official resolutions (Gamma, 50 markets per request). Prices come from
Binance 1-second candles: the daily archive files for complete days, the API for today. Features are built
with the same code as live (no order-book data, which cannot be recovered), so these rows train
probability models but are never used to judge trading strategies.
"""
import csv
import io
import json
import time
import urllib.request
import zipfile

import config as C
import features as F
import market as MK
from util import UA, FetchError, http_json

BF_T = list(range(15, 300, 15))
MAX_DAYS = 14


def _api_klines(start, end):
    rows, t = {}, start
    while t < end:
        data = None
        for attempt in range(3):
            try:
                data = http_json(f"{C.BINANCE[0]}/klines?symbol={C.A['bn']}&interval=1s&startTime={t * 1000}&limit=1000", timeout=10)
                break
            except FetchError:
                time.sleep(1 + attempt)
        if not data:
            break
        for k in data:
            rows[int(k[0]) // 1000] = (float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), float(k[9]))
        nxt = int(data[-1][0]) // 1000 + 1
        if nxt <= t:
            break
        t = nxt
    return rows


def _day_archive(day):
    """Binance public archive of 1-second candles for one complete UTC day (YYYY-MM-DD)."""
    url = f"https://data.binance.vision/data/spot/daily/klines/{C.A['bn']}/1s/{C.A['bn']}-1s-{day}.zip"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        blob = r.read()
    rows = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open(z.namelist()[0]) as f:
            for k in csv.reader(io.TextIOWrapper(f)):
                if not k or not k[0].isdigit():
                    continue
                ts = int(k[0])
                ts = ts // 1_000_000 if ts > 10 ** 14 else ts // 1000   # archives switched to microseconds in 2025
                rows[ts] = (float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), float(k[9]))
    return rows


def _klines(start, end):
    """1-second candles for [start, end): archives for complete past days, API for the rest."""
    out = {}
    day0 = start // 86400 * 86400
    today = int(time.time()) // 86400 * 86400
    d = day0
    while d < end:
        lo, hi = max(start, d), min(end, d + 86400)
        got = False
        if d < today:
            try:
                rows = _day_archive(time.strftime("%Y-%m-%d", time.gmtime(d)))
                out.update({ts: v for ts, v in rows.items() if lo <= ts < hi})
                got = True
            except Exception:  # noqa: BLE001 - archive not published yet: fall back to the API
                got = False
        if not got:
            out.update(_api_klines(lo, hi))
        d += 86400
    return out


def _outcomes(starts):
    res = {}
    for i in range(0, len(starts), 50):
        chunk = starts[i:i + 50]
        q = "&".join(f"slug={MK.slug_for(s)}" for s in chunk)
        for attempt in range(3):
            try:
                rows = http_json(f"{C.GAMMA}/markets?closed=true&limit=100&{q}", timeout=15)
                break
            except FetchError:
                rows = []
                time.sleep(1 + attempt)
        for m in rows:
            try:
                s = int(m["slug"].rsplit("-", 1)[1])
                p = json.loads(m["outcomePrices"]) if isinstance(m.get("outcomePrices"), str) else m.get("outcomePrices")
                up, dn = float(p[0]), float(p[1])
            except (KeyError, ValueError, TypeError, IndexError):
                continue
            if m.get("closed") and up >= 0.99 and dn <= 0.01:
                res[s] = "UP"
            elif m.get("closed") and dn >= 0.99 and up <= 0.01:
                res[s] = "DOWN"
    return res


def run(db, journal, start_ts, end_ts, budget_s=420, label="Backfilling"):
    first = MK.window_start(start_ts)
    last = MK.window_start(end_ts) - C.WINDOW
    starts = list(range(first, last + 1, C.WINDOW))
    have = {r[0] for r in db.q("SELECT DISTINCT m.start FROM markets m JOIN checkpoints c ON c.slug=m.slug "
                               "WHERE m.start>=? AND m.start<=? AND c.schema=?", (first, last, F.SCHEMA))}
    todo = [s for s in starts if s not in have]
    if not todo:
        return 0
    t0 = time.time()
    journal.activity(f"{label} {len(todo)} past markets so the models have history to learn from", kind="research")
    outs = _outcomes(todo)
    todo = [s for s in todo if s in outs]
    if not todo:
        return 0
    btc = _klines(todo[0] - 960, todo[-1] + C.WINDOW + 1)
    db.many("INSERT OR IGNORE INTO btc1s VALUES(?,?,?,?,?,?,?)", [(ts,) + v for ts, v in btc.items()])
    done = 0
    for s in todo:
        if time.time() - t0 > budget_s:
            journal.activity(f"Backfill paused after {done} markets (time budget); it continues next run", kind="research")
            break
        pre = [btc[x][3] for x in range(s - 60, s) if x in btc]
        if len(pre) < 40:
            continue
        open_proxy = sum(pre) / len(pre)   # Binance TWAP before the open: same-currency proxy for the price to beat
        slug = MK.slug_for(s)
        prev = outs.get(s - C.WINDOW)
        po = 0 if prev is None else (1 if prev == "UP" else -1)
        db.upsert_market({"slug": slug, "start": s, "end": s + C.WINDOW, "title": f"(history) {slug}", "fee_rate": 0.07,
                          "fee_exp": 1.0, "twap_secs": C.TWAP_SECONDS, "backfill": 1})
        db.set_result(slug, None, None, outs[s], "official (backfill)")
        for t in BF_T:
            ts = s + t
            f = F.build(ts, s, s + C.WINDOW, C.TWAP_SECONDS, open_proxy, None, btc, None, None, None, po, None,
                        backfill=True)
            db.put_checkpoint(slug, t, ts, F.SCHEMA, f, None, F.get(f, "p_fair") is not None)
        done += 1
    db.commit()
    journal.activity(f"Backfill: added {done} markets with official outcomes in {time.time() - t0:.0f}s", kind="research")
    return done


def oldest_history(db):
    r = db.one("SELECT MIN(m.start) FROM markets m JOIN checkpoints c ON c.slug=m.slug "
               "WHERE m.outcome IS NOT NULL AND c.schema=?", (F.SCHEMA,))
    return r[0] if r and r[0] else None
