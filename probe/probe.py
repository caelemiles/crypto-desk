import json, time, asyncio
out = {}
async def ws(name, sub, secs=45, ping="PING"):
    import websockets
    times, other, snaps = [], [], []
    try:
        async with websockets.connect("wss://ws-live-data.polymarket.com", open_timeout=10) as w:
            await w.send(json.dumps(sub))
            start = time.time(); last_ping = start
            while time.time() - start < secs:
                if ping and time.time() - last_ping > 5:
                    await w.send(ping); last_ping = time.time()
                try:
                    m = await asyncio.wait_for(w.recv(), timeout=1)
                except asyncio.TimeoutError:
                    continue
                if isinstance(m, str) and m.startswith("{"):
                    d = json.loads(m); p = d.get("payload") or {}
                    if isinstance(p.get("data"), list):
                        ts = [x["timestamp"] for x in p["data"]]
                        snaps.append({"at": round(time.time()-start,1), "n": len(ts), "newest_age_s": round(time.time()-max(ts)/1000,1) if ts else None, "sym": p.get("symbol")})
                    elif "value" in p:
                        times.append((round(time.time()-start,1), p.get("symbol"), round(time.time()-p["timestamp"]/1000,1)))
                    else:
                        other.append(str(m)[:150])
                else:
                    other.append(str(m)[:40])
        closed = None
    except Exception as e:
        closed = repr(e)[:200]
    out[name] = {"updates": len(times), "first": times[:4], "last": times[-3:], "snaps": snaps[:4], "other": other[:6], "closed": closed}
async def main():
    await ws("A_cl_json", {"action": "subscribe", "subscriptions": [{"topic": "crypto_prices_chainlink", "type": "*", "filters": json.dumps({"symbol": "btc/usd"})}]})
    await ws("B_cl_nofilter", {"action": "subscribe", "subscriptions": [{"topic": "crypto_prices_chainlink", "type": "*", "filters": ""}]})
    await ws("C_cl_update", {"action": "subscribe", "subscriptions": [{"topic": "crypto_prices_chainlink", "type": "update", "filters": json.dumps({"symbol": "btc/usd"})}]})
    await ws("D_binance", {"action": "subscribe", "subscriptions": [{"topic": "crypto_prices", "type": "update", "filters": "btcusdt"}]})
    await ws("E_cl_noping", {"action": "subscribe", "subscriptions": [{"topic": "crypto_prices_chainlink", "type": "*", "filters": json.dumps({"symbol": "btc/usd"})}]}, ping=None)
asyncio.run(main())
json.dump(out, open("probe_out.json","w"), indent=1)
