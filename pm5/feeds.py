"""External Data Intelligence: Chainlink reference stream (Polymarket RTDS), Binance spot, Coinbase, OKX derivatives.

Every feed records its own health so the bot can refuse to act on stale or missing data.
"""
import collections
import json
import threading
import time

import config as C
from util import FetchError, http_json


class Health:
    def __init__(self):
        self.last_ok = {}
        self.errors = collections.Counter()
        self.last_err = {}

    def ok(self, name):
        self.last_ok[name] = time.time()

    def bad(self, name, err):
        self.errors[name] += 1
        self.last_err[name] = (time.time(), str(err)[:160])

    def age(self, name):
        t = self.last_ok.get(name)
        return 1e9 if t is None else time.time() - t

    def summary(self):
        now = time.time()
        return {k: {"age_s": round(now - v, 1), "errors": self.errors.get(k, 0),
                    "last_error": self.last_err.get(k, (0, ""))[1]} for k, v in self.last_ok.items()} | \
               {k: {"age_s": None, "errors": n, "last_error": self.last_err.get(k, (0, ""))[1]}
                for k, n in self.errors.items() if k not in self.last_ok}


class ChainlinkStream(threading.Thread):
    """Chainlink BTC/USD (the settlement reference family) via Polymarket's real-time data socket.

    Values arrive about once a second. A snapshot of the previous ~2 minutes comes with every (re)subscribe,
    so short disconnects leave no gap.
    """

    def __init__(self, health):
        super().__init__(daemon=True)
        self.health = health
        self.lock = threading.Lock()
        self.prices = collections.OrderedDict()  # ts (s) -> price
        self.new_rows = []
        self.stop = False
        self.reconnects = 0

    def _add(self, ts_ms, value):
        ts = int(ts_ms) // 1000
        v = float(value)
        if not (1000 < v < 10_000_000):
            return
        with self.lock:
            if ts not in self.prices:
                self.prices[ts] = v
                self.new_rows.append((ts, v))
            while len(self.prices) > 4 * 3600:
                self.prices.popitem(last=False)
        self.health.ok("chainlink")

    def run(self):
        from websockets.sync.client import connect
        sub = {"action": "subscribe", "subscriptions": [
            {"topic": "crypto_prices_chainlink", "type": "*", "filters": json.dumps({"symbol": "btc/usd"})}]}
        backoff = 1
        while not self.stop:
            try:
                with connect(C.RTDS_WS, open_timeout=10, close_timeout=2, max_size=2 ** 22) as ws:
                    ws.send(json.dumps(sub))
                    last_ping = time.time()
                    backoff = 1
                    while not self.stop:
                        if time.time() - last_ping > 5:
                            ws.send("PING")
                            last_ping = time.time()
                        try:
                            msg = ws.recv(timeout=3)
                        except TimeoutError:
                            if self.health.age("chainlink") > 20:
                                raise ConnectionError("chainlink stream silent for 20s")
                            continue
                        if not msg or not isinstance(msg, str) or msg[0] != "{":
                            continue
                        d = json.loads(msg)
                        p = d.get("payload") or {}
                        if isinstance(p.get("data"), list):  # snapshot
                            for pt in p["data"]:
                                self._add(pt["timestamp"], pt["value"])
                        elif "value" in p and "timestamp" in p:
                            self._add(p["timestamp"], p["value"])
            except Exception as e:  # noqa: BLE001 - reconnect on anything
                self.health.bad("chainlink", e)
                self.reconnects += 1
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)

    def take_rows(self):
        with self.lock:
            rows, self.new_rows = self.new_rows, []
        return rows

    def series(self, since_ts):
        with self.lock:
            return [(t, p) for t, p in self.prices.items() if t >= since_ts]

    def last(self):
        with self.lock:
            if not self.prices:
                return None, None
            t = next(reversed(self.prices))
            return t, self.prices[t]


class Spot:
    """Binance 1-second candles (BTC, ETH), Binance order book, Coinbase ticker, OKX derivatives."""

    def __init__(self, health):
        self.h = health
        self.btc = collections.OrderedDict()   # ts -> (o,h,l,c,v,taker_buy_vol)
        self.eth = collections.OrderedDict()   # ts -> close
        self.book = None                       # (ts, bid_qty_10bp, ask_qty_10bp, mid, spread)
        self.coinbase = None                   # (ts, price)
        self.okx = {}                          # kind -> (ts, value)
        self.new_btc, self.new_eth, self.new_ext = [], [], []
        self.t_btc = self.t_eth = self.t_book = self.t_cb = self.t_okx = 0

    def _klines(self, sym, start_ms):
        for base in C.BINANCE:
            try:
                return http_json(f"{base}/klines?symbol={sym}&interval=1s&startTime={start_ms}&limit=1000", timeout=6)
            except FetchError as e:
                err = e
        raise err

    def poll(self, now, backlog_s=900):
        if now - self.t_btc >= 2:
            self.t_btc = now
            try:
                last = next(reversed(self.btc)) if self.btc else int(now) - backlog_s
                rows = self._klines("BTCUSDT", (last + 1) * 1000)
                for k in rows:
                    ts = int(k[0]) // 1000
                    if ts >= int(now):  # still-open second: skip, it is incomplete
                        continue
                    v = (float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), float(k[9]))
                    if ts not in self.btc:
                        self.new_btc.append((ts,) + v)
                    self.btc[ts] = v
                while len(self.btc) > 4 * 3600:
                    self.btc.popitem(last=False)
                self.h.ok("binance_btc")
            except FetchError as e:
                self.h.bad("binance_btc", e)
        if now - self.t_eth >= 5:
            self.t_eth = now
            try:
                last = next(reversed(self.eth)) if self.eth else int(now) - backlog_s
                for k in self._klines("ETHUSDT", (last + 1) * 1000):
                    ts = int(k[0]) // 1000
                    if ts >= int(now):
                        continue
                    if ts not in self.eth:
                        self.new_eth.append((ts, float(k[4])))
                    self.eth[ts] = float(k[4])
                while len(self.eth) > 4 * 3600:
                    self.eth.popitem(last=False)
                self.h.ok("binance_eth")
            except FetchError as e:
                self.h.bad("binance_eth", e)
        if now - self.t_book >= 2:
            self.t_book = now
            try:
                d = http_json(f"{C.BINANCE[0]}/depth?symbol=BTCUSDT&limit=100", timeout=5)
                bids = [(float(p), float(q)) for p, q in d["bids"]]
                asks = [(float(p), float(q)) for p, q in d["asks"]]
                if bids and asks and bids[0][0] < asks[0][0]:
                    mid = (bids[0][0] + asks[0][0]) / 2
                    bq = sum(q for p, q in bids if p >= mid * 0.999)
                    aq = sum(q for p, q in asks if p <= mid * 1.001)
                    self.book = (now, bq, aq, mid, asks[0][0] - bids[0][0])
                    self.h.ok("binance_book")
            except (FetchError, KeyError, IndexError) as e:
                self.h.bad("binance_book", e)
        if now - self.t_cb >= 5:
            self.t_cb = now
            try:
                d = http_json(f"{C.COINBASE}/products/BTC-USD/ticker", timeout=5)
                self.coinbase = (now, float(d["price"]))
                self.h.ok("coinbase")
            except (FetchError, KeyError, ValueError) as e:
                self.h.bad("coinbase", e)
        if now - self.t_okx >= 60:
            self.t_okx = now
            self._okx(now)

    def _okx(self, now):
        ts = int(now)
        try:
            d = http_json(f"{C.OKX}/public/funding-rate?instId=BTC-USDT-SWAP", timeout=6)["data"][0]
            self._ext(ts, "funding", float(d["fundingRate"]))
            self._ext(ts, "premium", float(d.get("premium") or 0))
            d = http_json(f"{C.OKX}/public/open-interest?instType=SWAP&instId=BTC-USDT-SWAP", timeout=6)["data"][0]
            self._ext(ts, "oi_usd", float(d["oiUsd"]))
            d = http_json(f"{C.OKX}/rubik/stat/taker-volume?ccy=BTC&instType=CONTRACTS&period=5m", timeout=6)["data"]
            if d:
                self._ext(ts, "fut_taker_buy", float(d[0][1]))
                self._ext(ts, "fut_taker_sell", float(d[0][2]))
            d = http_json(f"{C.OKX}/public/liquidation-orders?instType=SWAP&uly=BTC-USDT&state=filled&limit=100", timeout=6)["data"]
            long_liq = short_liq = 0.0
            for grp in d:
                for x in grp.get("details", []):
                    if int(x["ts"]) / 1000 >= now - 300:
                        usd = float(x["sz"]) * 0.01 * float(x["bkPx"])  # BTC-USDT-SWAP contract = 0.01 BTC
                        if x.get("posSide") == "long":
                            long_liq += usd
                        else:
                            short_liq += usd
            self._ext(ts, "liq_long_5m", long_liq)
            self._ext(ts, "liq_short_5m", short_liq)
            self.h.ok("okx")
        except (FetchError, KeyError, IndexError, ValueError, TypeError) as e:
            self.h.bad("okx", e)

    def _ext(self, ts, kind, value):
        self.okx[kind] = (ts, value)
        self.new_ext.append((ts, kind, value))

    def take_rows(self):
        a, b, c = self.new_btc, self.new_eth, self.new_ext
        self.new_btc, self.new_eth, self.new_ext = [], [], []
        return a, b, c

    def btc_last(self):
        if not self.btc:
            return None, None
        t = next(reversed(self.btc))
        return t, self.btc[t][3]
