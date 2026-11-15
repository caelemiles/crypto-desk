#!/usr/bin/env python3
"""Crypto Desk - standalone paper-trading runner (no AI usage).

Runs the self-learning engine (engine.py) on fresh 15-minute Binance candles, then saves the state
and a small dashboard data file. GitHub Actions calls this every ~10 minutes.
PAPER MONEY ONLY: no keys, no wallet, no real orders.
"""
import json, os, time, urllib.request, urllib.error
import engine as E

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(ROOT, "state", "state.json")
TRADES = os.path.join(ROOT, "state", "trades.json")
RUNS = os.path.join(ROOT, "state", "runs.json")
DASH = os.path.join(ROOT, "docs", "data.json")
SOURCES = ["https://data-api.binance.vision/api/v3/klines", "https://api.binance.com/api/v3/klines"]
FIRST_BARS = 960  # 10 days of history on the first run: the engine learns from all of it straight away


def load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save(path, data, compact=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, allow_nan=False, separators=(",", ":") if compact else None, indent=None if compact else 1)
    os.replace(tmp, path)


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "crypto-desk-paper-bot"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def fetch(coin, start, now):
    """All 15m candles from start up to and including the open one."""
    rows, t = [], start
    while t <= now:
        data, err = None, None
        for base in SOURCES:
            try:
                data = get_json(f"{base}?symbol={coin}USDT&interval=15m&startTime={t}&limit=1000")
                break
            except (urllib.error.URLError, TimeoutError, ValueError) as e:
                err = e
        if data is None:
            raise RuntimeError(f"{coin}: {err}")
        if not data:
            break
        rows += [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), round(float(k[5]))] for k in data]
        if len(data) < 1000:
            break
        t = int(data[-1][0]) + E.BAR
    return rows


def main():
    now = int(time.time() * 1000)
    state = load(STATE, {})
    hist = state.get("history") or {}
    bar = now // E.BAR * E.BAR
    new, skipped = {}, []
    for c in E.COINS:
        h = hist.get(c)
        start = h[-1][0] + E.BAR if h else bar - (FIRST_BARS - 1) * E.BAR
        try:
            rows = fetch(c, start, now)
            if rows:
                new[c] = rows
        except Exception as e:  # noqa: BLE001 - a coin with bad data is skipped, never guessed
            skipped.append(f"{c}: {e}")
    if not new:
        print(json.dumps({"ok": False, "error": "no data fetched", "skipped": skipped}))
        raise SystemExit(1)

    rep = E.run(state, new, now)
    bot, model = rep["bot"], rep["model"]
    save(STATE, {"bot": bot, "model": model, "history": rep["history"]})

    trades = load(TRADES, {})
    for tr in rep["opened"] + rep["closed"] + bot["positions"]:
        trades[tr["id"]] = tr
    save(TRADES, trades)

    runs = load(RUNS, [])
    runs.append({"t": now, "equity": bot["equity"], "opened": [t["id"] for t in rep["opened"]],
                 "closed": [[t["id"], t["coin"], t["side"], t["pnl"]] for t in rep["closed"]],
                 "open": len(bot["positions"]), "resolved": rep["learning"]["resolved"],
                 "warnings": rep["warnings"] + skipped})
    runs = runs[-500:]
    save(RUNS, runs)

    closed = sorted([t for t in trades.values() if t.get("status") == "closed"], key=lambda t: -t.get("exitTime", 0))
    dash = {"updated": now, "bot": {k: bot.get(k) for k in (
                "startBalance", "equity", "cash", "realized", "fees", "closedN", "wins", "losses", "grossWin",
                "grossLoss", "maxDD", "peak", "daily", "today", "dailyTarget", "curve", "brake", "riskNote",
                "readiness", "startedAt", "runs", "orders", "bySide", "byCoin")},
            "positions": bot["positions"], "closed": closed[:150],
            "scan": [{k: it.get(k) for k in ("coin", "price", "chg24", "decision", "why", "best")} for it in rep["scan"]],
            "learning": {"samples": model.get("samples"), "minEvR": model["params"].get("minEvR"),
                         "calib": model.get("calib2"), "patterns": model.get("patterns"),
                         "settings": model.get("settings"), "lessons": model.get("lessons", [])[-15:]},
            "runs": runs[-40:], "params": {k: model["params"].get(k) for k in (
                "fixedStake", "atrStop", "rr", "timeStopH", "hardStopH", "maxOpen", "partialAtR", "runnerR")}}
    save(DASH, dash)
    print(json.dumps({"ok": True, "equity": bot["equity"], "opened": len(rep["opened"]), "closed": len(rep["closed"]),
                      "open": len(bot["positions"]), "resolved": rep["learning"]["resolved"], "skipped": skipped}))


if __name__ == "__main__":
    main()
