"""Backfill: gives the AI a history to learn from on day one, and fills gaps after restarts.

For past windows it fetches the official open/close reference (Polymarket's Chainlink prices) and Binance
1-second candles, then builds the same features as live (without order-book data, which cannot be recovered).
These rows train probability models; they are never used to judge trading strategies (no real books).
"""
import time
from concurrent.futures import ThreadPoolExecutor

import config as C
import features as F
import market as MK
from util import FetchError, http_json

BF_T = list(range(15, 300, 15))


def _klines(sym, start, end):
    rows, t = {}, start
    while t < end:
        data = None
        for attempt in range(3):
            try:
                data = http_json(f"{C.BINANCE[0]}/klines?symbol={sym}&interval=1s&startTime={t * 1000}&limit=1000", timeout=10)
                break
            except FetchError:
                time.sleep(1 + attempt)
        if not data:
            break
        for k in data:
            ts = int(k[0]) // 1000
            rows[ts] = (float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), float(k[9]))
        nxt = int(data[-1][0]) // 1000 + 1
        if nxt <= t:
            break
        t = nxt
    return rows


def run(db, journal, start_ts, end_ts, max_windows=1400):
    first = MK.window_start(start_ts)
    last = MK.window_start(end_ts) - C.WINDOW  # only finished windows
    starts = list(range(first, last + 1, C.WINDOW))[-max_windows:]
    have = {r[0] for r in db.q("SELECT DISTINCT m.start FROM markets m JOIN checkpoints c ON c.slug=m.slug "
                               "WHERE m.start>=? AND m.start<=?", (first, last))}
    todo = [s for s in starts if s not in have]
    if not todo:
        return 0
    journal.activity(f"Backfilling {len(todo)} past markets so the models have history to learn from", kind="research")
    t0 = time.time()

    def ref(s):
        try:
            return s, MK.crypto_price(s)
        except FetchError:
            return s, None

    with ThreadPoolExecutor(max_workers=6) as ex:
        refs = dict(ex.map(ref, todo))
    btc = _klines("BTCUSDT", todo[0] - 960, todo[-1] + C.WINDOW + 1)
    eth = _klines("ETHUSDT", todo[0] - 960, todo[-1] + C.WINDOW + 1)
    db.many("INSERT OR IGNORE INTO btc1s VALUES(?,?,?,?,?,?,?)", [(ts,) + v for ts, v in btc.items()])
    db.many("INSERT OR IGNORE INTO eth1s VALUES(?,?)", [(ts, v[3]) for ts, v in eth.items()])
    eth_c = {ts: v[3] for ts, v in eth.items()}
    done = 0
    prev = None
    for s in todo:
        r = refs.get(s)
        outcome = None
        if r and r["completed"] and r["open"] and r["close"]:
            outcome = "UP" if r["close"] >= r["open"] else "DOWN"
        if outcome is None:
            prev = None
            continue
        # Binance TWAP over the 60s before the open, as the same-currency proxy for the price to beat
        pre = [btc[x][3] for x in range(s - 60, s) if x in btc]
        if len(pre) < 40:
            prev = outcome
            continue
        open_proxy = sum(pre) / len(pre)
        slug = MK.slug_for(s)
        db.upsert_market({"slug": slug, "start": s, "end": s + C.WINDOW, "title": f"(backfill) {slug}", "fee_rate": 0.07,
                          "fee_exp": 1.0, "twap_secs": C.TWAP_SECONDS, "backfill": 1})
        db.set_result(slug, r["open"], r["close"], outcome, "backfill")
        po = 0 if prev is None else (1 if prev == "UP" else -1)
        for t in BF_T:
            ts = s + t
            f = F.build(ts, s, s + C.WINDOW, C.TWAP_SECONDS, open_proxy, None, btc, eth_c, None, None, po, None,
                        backfill=True)
            valid = F.get(f, "p_fair") is not None
            db.put_checkpoint(slug, t, ts, F.SCHEMA, f, None, valid)
        prev = outcome
        done += 1
    db.commit()
    journal.activity(f"Backfill complete: {done} markets with outcomes in {time.time() - t0:.0f}s", kind="research")
    return done
