"""Polymarket data (read-only): events and markets from Gamma, order books from the CLOB.

Nothing here can place an order: the only CLOB call is POST /books, which returns order books.
"""
import json
import time
import urllib.parse

import core as K


def _j(v, default=None):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return default
    return v if v is not None else default


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def parse_time(s):
    if not s:
        return None
    s = str(s).replace("Z", "+00:00")
    try:
        from datetime import datetime
        return datetime.fromisoformat(s.replace(".000", "")).timestamp()
    except ValueError:
        return None


def parse_market(m):
    """A clean Gamma market record (binary YES/NO)."""
    outs = _j(m.get("outcomes"), []) or []
    toks = _j(m.get("clobTokenIds"), []) or []
    prices = [_f(x) for x in (_j(m.get("outcomePrices"), []) or [])]
    if len(outs) != 2 or len(toks) != 2:
        return None
    fs = m.get("feeSchedule") or {}
    if m.get("feesEnabled") is False:
        rate, exp = 0.0, 1.0
    elif fs.get("rate") is not None:
        rate, exp = _f(fs.get("rate"), 0.0), _f(fs.get("exponent"), 1.0)
    elif m.get("feesEnabled"):
        rate, exp = 0.07, 1.0      # fees on but no schedule shown: assume the crypto rate (conservative)
    else:
        rate, exp = 0.0, 1.0
    return {
        "id": str(m.get("id")), "slug": m.get("slug"), "question": m.get("question") or "",
        "group": m.get("groupItemTitle") or "", "threshold": _f(m.get("groupItemThreshold")),
        "condition_id": m.get("conditionId"), "outcomes": outs, "yes_token": toks[0], "no_token": toks[1],
        "prices": prices, "best_bid": _f(m.get("bestBid")), "best_ask": _f(m.get("bestAsk")),
        "closed": bool(m.get("closed")), "active": bool(m.get("active", True)),
        "accepting": bool(m.get("acceptingOrders", True)), "neg_risk": bool(m.get("negRisk")),
        "fee_rate": rate, "fee_exp": exp, "min_size": _f(m.get("orderMinSize"), 5.0) or 5.0,
        "tick": _f(m.get("orderPriceMinTickSize"), 0.01) or 0.01, "end": parse_time(m.get("endDate")),
        "liquidity": _f(m.get("liquidityClob") or m.get("liquidity"), 0.0) or 0.0,
        "volume": _f(m.get("volumeNum") or m.get("volume"), 0.0) or 0.0,
        "description": (m.get("description") or "")[:1500], "uma": m.get("umaResolutionStatus"),
    }


def resolved_yes(mk):
    """True / False once a market has settled at $1/$0, else None."""
    if not mk.get("closed") or len(mk.get("prices") or []) != 2:
        return None
    y, n = mk["prices"]
    if y is None or n is None:
        return None
    if y >= 0.99 and n <= 0.01:
        return True
    if n >= 0.99 and y <= 0.01:
        return False
    return None


def parse_event(e):
    mks = [x for x in (parse_market(m) for m in (e.get("markets") or [])) if x]
    return {
        "id": str(e.get("id")), "slug": e.get("slug") or "", "title": e.get("title") or "",
        "description": (e.get("description") or "")[:2000], "neg_risk": bool(e.get("negRisk") or e.get("enableNegRisk")),
        "augmented": bool(e.get("negRiskAugmented")), "end": parse_time(e.get("endDate")),
        "closed": bool(e.get("closed")), "active": bool(e.get("active", True)),
        "tags": [t.get("slug") for t in (e.get("tags") or []) if isinstance(t, dict)],
        "markets": mks, "resolution_source": e.get("resolutionSource") or "",
    }


_PAGE = {"limit": 20, "order": True}   # Gamma answers 406 when a page is too big; learned at runtime


def events(params, pages=1, limit=20, cursor=None):
    """Events via Gamma's keyset endpoint (offset paging is rejected). Returns (events, next cursor).
    Event records are large (20 events can be 400 KB), so pages are small; on HTTP 406 the page shrinks."""
    out = []
    for _ in range(pages):
        while True:
            q = {k: v for k, v in params.items() if k != "active"}
            if not _PAGE["order"]:
                q.pop("order", None)
                q.pop("ascending", None)
            q["limit"] = min(int(limit), _PAGE["limit"])
            if cursor:
                q["after_cursor"] = cursor
            try:
                d = K.http(f"{K.GAMMA}/events/keyset?{urllib.parse.urlencode(q)}", timeout=20) or {}
                break
            except K.FetchError as e:
                if "406" not in str(e):
                    raise
                if _PAGE["limit"] > 5:
                    _PAGE["limit"] = max(5, _PAGE["limit"] // 2)
                elif _PAGE["order"]:
                    _PAGE["order"] = False
                else:
                    raise
        rows = d.get("events") if isinstance(d, dict) else d
        out.extend(parse_event(e) for e in rows or [])
        cursor = d.get("next_cursor") if isinstance(d, dict) else None
        if not cursor:
            break
    return out, cursor


def event_by_slug(slug):
    rows = K.http(f"{K.GAMMA}/events?slug={urllib.parse.quote(slug)}", timeout=10)
    return parse_event(rows[0]) if rows else None


def parse_book(raw):
    bids = sorted(((float(x["price"]), float(x["size"])) for x in raw.get("bids", [])), reverse=True)
    asks = sorted((float(x["price"]), float(x["size"])) for x in raw.get("asks", []))
    for p, s in bids + asks:
        if not (0 < p < 1) or s < 0:
            raise ValueError("price/size out of range")
    if bids and asks and bids[0][0] >= asks[0][0]:
        raise ValueError("crossed book")
    return {"bids": bids, "asks": asks, "ts": int(raw.get("timestamp") or 0) / 1000.0}


def books(tokens, chunk=40):
    """Order books for many tokens. Returns {token: book}; tokens whose book is bad or missing are left out."""
    out, toks = {}, [t for t in dict.fromkeys(tokens) if t]
    for i in range(0, len(toks), chunk):
        part = toks[i:i + chunk]
        try:
            raw = K.http(f"{K.CLOB}/books", data=[{"token_id": t} for t in part], timeout=12)
        except K.FetchError:
            time.sleep(0.5)
            continue
        fetched = K.now()
        if i + chunk < len(toks):
            time.sleep(0.15)   # stay well inside the CLOB's rate limit
        for b in raw or []:
            try:
                bk = parse_book(b)
            except (ValueError, KeyError, TypeError):
                continue
            if bk["ts"] and fetched - bk["ts"] > 120:
                continue  # stale
            bk["fetched"] = fetched
            out[b.get("asset_id")] = bk
    return out


def best(book):
    """(best bid, best ask) of a book, with 0 / 1 when a side is empty."""
    if not book:
        return 0.0, 1.0
    return (book["bids"][0][0] if book["bids"] else 0.0), (book["asks"][0][0] if book["asks"] else 1.0)
