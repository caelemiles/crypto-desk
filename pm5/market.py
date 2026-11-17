"""Market Intelligence: finds the live BTC 5-minute market, reads its order book, and tracks its lifecycle.

Safety rules enforced here:
- a market is only usable if its slug, start and end all match the current 5-minute window exactly;
- never usable once closed, not accepting orders, or within LAST_ENTRY_GUARD seconds of the end;
- order books are sanity-checked (sorted, inside 0..1, fresh) before anyone may trade on them.
"""
import json
import time

import config as C
from util import FetchError, http_json, iso_utc


def slug_for(start):
    return f"{C.ASSET}-updown-5m-{int(start)}"


def window_start(ts):
    return int(ts) // C.WINDOW * C.WINDOW


def parse_market(m, start):
    """Validate a Gamma market record for the window starting at `start`. Returns a clean dict or raises."""
    slug = slug_for(start)
    if m.get("slug") != slug:
        raise ValueError(f"slug mismatch {m.get('slug')} != {slug}")
    outs = json.loads(m["outcomes"]) if isinstance(m.get("outcomes"), str) else m.get("outcomes")
    toks = json.loads(m["clobTokenIds"]) if isinstance(m.get("clobTokenIds"), str) else m.get("clobTokenIds")
    if [o.lower() for o in outs] != ["up", "down"] or len(toks) != 2:
        raise ValueError(f"unexpected outcomes {outs}")
    est = m.get("eventStartTime")
    if est and est.replace(".000", "") != iso_utc(start):
        raise ValueError(f"eventStartTime {est} does not match window {iso_utc(start)}")
    if m.get("endDate") and m["endDate"].replace(".000", "") != iso_utc(start + C.WINDOW):
        raise ValueError(f"endDate {m.get('endDate')} does not match window end")
    fs = m.get("feeSchedule") or {}
    cfg = m.get("cryptoMarketConfig") or {}
    return {
        "slug": slug, "start": int(start), "end": int(start) + C.WINDOW, "market_id": str(m.get("id")),
        "condition_id": m.get("conditionId"), "up_token": toks[0], "down_token": toks[1],
        "title": m.get("question"), "description": (m.get("description") or "")[:1200],
        "fee_rate": float(fs.get("rate", 0.07)) if m.get("feesEnabled", True) else 0.0,
        "fee_exp": float(fs.get("exponent", 1)),
        "twap_secs": int(cfg.get("twapLookbackSeconds") or (C.TWAP_SECONDS if cfg.get("twapEnabled", True) else 0)),
        "min_size": float(m.get("orderMinSize") or 5), "tick": float(m.get("orderPriceMinTickSize") or 0.01),
        "accepting": bool(m.get("acceptingOrders", True)), "closed": bool(m.get("closed", False)),
        "liquidity": float(m.get("liquidityClob") or m.get("liquidity") or 0),
        "volume": float(m.get("volumeNum") or m.get("volume") or 0),
        "resolution_source": m.get("resolutionSource"), "backfill": 0,
    }


def fetch_market(start, closed=None):
    url = f"{C.GAMMA}/markets?slug={slug_for(start)}"
    if closed is not None:
        url += f"&closed={'true' if closed else 'false'}"
    rows = http_json(url, timeout=6)
    if not rows:
        return None
    return rows[0]


def parse_book(raw):
    bids = sorted(((float(x["price"]), float(x["size"])) for x in raw.get("bids", [])), reverse=True)
    asks = sorted((float(x["price"]), float(x["size"])) for x in raw.get("asks", []))
    for p, s in bids + asks:
        if not (0 < p < 1) or s < 0:
            raise ValueError("price/size out of range")
    if bids and asks and bids[0][0] >= asks[0][0]:
        raise ValueError("crossed book")
    return {"bids": bids, "asks": asks, "ts": int(raw.get("timestamp") or 0) / 1000.0}


def fetch_books(mkt):
    """Both outcome books in one request. Returns {'up': book, 'down': book, 'fetched': ts}."""
    raw = http_json(f"{C.CLOB}/books", data=[{"token_id": mkt["up_token"]}, {"token_id": mkt["down_token"]}], timeout=5)
    out = {"fetched": time.time()}
    for b in raw:
        if b.get("asset_id") == mkt["up_token"]:
            out["up"] = parse_book(b)
        elif b.get("asset_id") == mkt["down_token"]:
            out["down"] = parse_book(b)
    if "up" not in out or "down" not in out:
        raise ValueError("book missing an outcome")
    for side in ("up", "down"):
        if out[side]["ts"] and out["fetched"] - out[side]["ts"] > 30:
            raise ValueError(f"{side} book is stale ({out['fetched'] - out[side]['ts']:.0f}s)")
    return out


def top(book, side):
    """(best_bid, bid_size, best_ask, ask_size, mid) for 'up' or 'down'."""
    b = book[side]
    bb, bs = b["bids"][0] if b["bids"] else (0.0, 0.0)
    ba, az = b["asks"][0] if b["asks"] else (1.0, 0.0)
    mid = (bb + ba) / 2 if b["bids"] and b["asks"] else None
    return bb, bs, ba, az, mid


def compact_book(book, levels=6):
    """Small copy of the book stored with each checkpoint so strategies can be replayed later."""
    return {s: {"b": [[p, round(q, 2)] for p, q in book[s]["bids"][:levels]],
                "a": [[p, round(q, 2)] for p, q in book[s]["asks"][:levels]]} for s in ("up", "down")}


def expand_book(cb):
    return {s: {"bids": [tuple(x) for x in cb[s]["b"]], "asks": [tuple(x) for x in cb[s]["a"]], "ts": 0} for s in ("up", "down")}


def crypto_price(start):
    """Polymarket's own open/close reference for a window (the Chainlink 'price to beat' and final price)."""
    d = http_json(f"{C.CRYPTO_PRICE}?symbol={C.A['cp']}&eventStartTime={iso_utc(start)}&variant=fiveminute"
                  f"&endDate={iso_utc(start + C.WINDOW)}", timeout=6)
    return {"open": d.get("openPrice"), "close": d.get("closePrice"), "completed": bool(d.get("completed"))}


def official_outcome(start):
    """The on-chain resolution once Polymarket marks the market closed (can lag several minutes)."""
    m = fetch_market(start, closed=True) or fetch_market(start)
    if not m or not m.get("closed"):
        return None
    prices = json.loads(m["outcomePrices"]) if isinstance(m.get("outcomePrices"), str) else m.get("outcomePrices")
    try:
        up, dn = float(prices[0]), float(prices[1])
    except (TypeError, ValueError, IndexError):
        return None
    if up >= 0.99 and dn <= 0.01:
        return "UP"
    if dn >= 0.99 and up <= 0.01:
        return "DOWN"
    return None


class Lifecycle:
    """Keeps the current and next market loaded and tells the controller what changed."""

    def __init__(self, health, db, journal):
        self.h, self.db, self.j = health, db, journal
        self.markets = {}       # start -> parsed market
        self.failed = {}        # start -> last failure time
        self.open_price = {}    # start -> Chainlink open (price to beat)
        self.open_src = {}
        self.t_meta = 0

    def ensure(self, start, now):
        if start in self.markets:
            return self.markets[start]
        if now - self.failed.get(start, 0) < 3:
            return None
        try:
            raw = fetch_market(start)
            if raw is None:
                raise ValueError("not listed yet")
            m = parse_market(raw, start)
            self.markets[start] = m
            self.db.upsert_market(m)
            self.h.ok("gamma")
            self.j.activity(f"New {C.A['name']} five-minute market detected: {m['title']}", kind="market")
            return m
        except (FetchError, ValueError, KeyError, TypeError) as e:
            self.failed[start] = now
            self.h.bad("gamma", e)
            return None

    def refresh_meta(self, start, now):
        """Re-read accepting/closed flags every 20s for the live market."""
        if now - self.t_meta < 20 or start not in self.markets:
            return
        self.t_meta = now
        try:
            raw = fetch_market(start)
            if raw:
                fresh = parse_market(raw, start)
                self.markets[start].update({k: fresh[k] for k in ("accepting", "closed", "liquidity", "volume")})
        except (FetchError, ValueError, KeyError, TypeError) as e:
            self.h.bad("gamma", e)

    def get_open(self, start, now, cl=None):
        """Price to beat = Chainlink 60-second TWAP at the open.

        Measured on 200 resolved markets: comparing 60s TWAPs at the close and the open reproduced Polymarket's
        official result 99% of the time, while the open/close shown by Polymarket's crypto-price endpoint matched
        only ~88%. So the AI computes the TWAP itself from the Chainlink stream and uses the endpoint only as a
        fallback.
        """
        if start in self.open_price:
            return self.open_price[start]
        if cl is not None and now >= start + 1:
            pts = [p for ts, p in cl.series(start - 59) if ts <= start]
            if len(pts) >= 45:
                self.open_price[start] = sum(pts) / len(pts)
                self.open_src[start] = "own Chainlink TWAP"
                self.db.set_open(slug_for(start), self.open_price[start])
                return self.open_price[start]
            if now < start + 15:
                return None  # give the stream a moment before falling back
        try:
            cp = crypto_price(start)
            if cp["open"]:
                self.open_price[start] = float(cp["open"])
                self.open_src[start] = "Polymarket crypto-price endpoint (fallback)"
                self.db.set_open(slug_for(start), float(cp["open"]))
                self.h.ok("reference")
                return self.open_price[start]
        except FetchError as e:
            self.h.bad("reference", e)
        return None

    def tradable(self, m, now):
        """True only for the exact current window, still open, accepting orders, and not in the final seconds."""
        if not m:
            return False, "no market"
        if m["start"] != window_start(now):
            return False, "not the current window"
        if now >= m["end"] - C.LAST_ENTRY_GUARD:
            return False, "too close to the close"
        if m.get("closed") or not m.get("accepting", True):
            return False, "market not accepting orders"
        return True, ""

    def forget_old(self, now):
        for s in [s for s in self.markets if s < now - 3 * 3600]:
            self.markets.pop(s, None)
            self.open_price.pop(s, None)
