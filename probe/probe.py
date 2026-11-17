import json, time, urllib.request, urllib.error, asyncio
out = {}
def get(url, data=None):
    t=time.time()
    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode() if data else None, headers={"User-Agent": "pm5-paper-bot/1.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode()), round(time.time()-t,2)
    except urllib.error.HTTPError as e:
        return e.code, (e.read() or b'')[:200].decode('utf8','ignore'), round(time.time()-t,2)
    except Exception as e:
        return 0, repr(e)[:200], round(time.time()-t,2)
now = int(time.time()); cur = now//300*300
# b) gamma multi-slug
slugs = [f"btc-updown-5m-{cur-300*k}" for k in range(2, 52)]
st, body, dt = get("https://gamma-api.polymarket.com/markets?closed=true&limit=100&" + "&".join("slug="+s for s in slugs))
out["gamma_multi"] = {"status": st, "n": len(body) if isinstance(body, list) else body, "dt": dt,
    "sample": [(m["slug"], m.get("outcomePrices"), m.get("closed"), m.get("umaResolutionStatus")) for m in body[:5]] if isinstance(body, list) else None}
# older one: 4 days ago
old = cur - 4*86400
st, body, dt = get(f"https://gamma-api.polymarket.com/markets?closed=true&slug=btc-updown-5m-{old}")
out["gamma_old"] = {"status": st, "body": str(body)[:300]}
# c) binance 1s klines 4 days ago
st, body, dt = get(f"https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1s&startTime={(old)*1000}&limit=3")
out["bn_1s_old"] = {"status": st, "body": str(body)[:300], "dt": dt}
# d) crypto-price rate test
from datetime import datetime, timezone
iso = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
codes = []
t0=time.time()
for k in range(2, 42):
    s = cur - 300*k
    st, body, dt = get(f"https://polymarket.com/api/crypto/crypto-price?symbol=BTC&eventStartTime={iso(s)}&variant=fiveminute&endDate={iso(s+300)}")
    codes.append((st, dt))
out["cp_rate"] = {"codes": codes, "total_s": round(time.time()-t0,1)}

async def ws(name, mode, secs=100):
    import websockets
    sub = {"action": "subscribe", "subscriptions": [{"topic": "crypto_prices_chainlink", "type": "*", "filters": json.dumps({"symbol": "btc/usd"})}]}
    times, other = [], []
    kw = {"ping_interval": None} if mode == "noping_lib" else {}
    try:
        async with websockets.connect("wss://ws-live-data.polymarket.com", open_timeout=10, **kw) as w:
            await w.send(json.dumps(sub))
            start = time.time(); last_ping = start
            while time.time() - start < secs:
                if mode in ("PING", "ping") and time.time() - last_ping > 5:
                    await w.send(mode); last_ping = time.time()
                try:
                    m = await asyncio.wait_for(w.recv(), timeout=1)
                    if isinstance(m, str) and m.startswith("{") and '"value"' in m:
                        times.append(round(time.time()-start,1))
                    else:
                        other.append((round(time.time()-start,1), str(m)[:40]))
                except asyncio.TimeoutError:
                    pass
        closed = None
    except Exception as e:
        closed = repr(e)[:200]
    gaps = [round(b-a,1) for a,b in zip(times, times[1:])]
    out[name] = {"n": len(times), "first": times[:3], "last": times[-3:], "max_gap": max(gaps) if gaps else None,
                 "big_gaps": [(times[i], g) for i,g in enumerate(gaps) if g > 3][:10], "other": other[:10], "closed": closed}
async def main():
    await asyncio.gather(ws("ws_PING", "PING"), ws("ws_ping", "ping"), ws("ws_none", "none"), ws("ws_noping_lib", "noping_lib"))
asyncio.run(main())
json.dump(out, open("probe_out.json","w"), indent=1)
