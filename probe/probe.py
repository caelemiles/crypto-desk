import json, time, urllib.request, asyncio
def get(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pm5-paper-bot/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r: return json.loads(r.read().decode())
    except Exception as e: return repr(e)[:200]
now=int(time.time()); cur=now//300*300
out={}
for a in ("eth","sol","xrp"):
    m=get(f"https://gamma-api.polymarket.com/markets?slug={a}-updown-5m-{cur}")
    out[a]= m[0] | {} if isinstance(m,list) and m else m
    if isinstance(out[a],dict): out[a]={k:out[a].get(k) for k in ("slug","question","description","resolutionSource","clobTokenIds","feeSchedule","cryptoMarketConfig","liquidityClob","volumeNum","eventStartTime","endDate")}
slugs="&".join(f"slug=eth-updown-5m-{cur-300*k}" for k in range(2,52))
r=get(f"https://gamma-api.polymarket.com/markets?closed=true&limit=100&{slugs}")
out["eth_closed_n"]=len(r) if isinstance(r,list) else r
day=time.strftime("%Y-%m-%d", time.gmtime(now-2*86400))
try:
    req=urllib.request.Request(f"https://data.binance.vision/data/spot/daily/klines/ETHUSDT/1s/ETHUSDT-1s-{day}.zip", method="HEAD", headers={"User-Agent":"x"})
    out["eth_archive"]=urllib.request.urlopen(req, timeout=15).status
except Exception as e: out["eth_archive"]=repr(e)[:100]
async def ws():
    import websockets
    syms={}
    async with websockets.connect("wss://ws-live-data.polymarket.com") as w:
        await w.send(json.dumps({"action":"subscribe","subscriptions":[{"topic":"crypto_prices_chainlink","type":"*","filters":""}]}))
        t=time.time()
        while time.time()-t<25:
            try: m=await asyncio.wait_for(w.recv(),2)
            except asyncio.TimeoutError: continue
            if isinstance(m,str) and m.startswith("{"):
                p=json.loads(m).get("payload") or {}
                s=p.get("symbol"); syms[s]=syms.get(s,0)+1
    out["rtds_symbols"]=syms
asyncio.run(ws())
json.dump(out, open("probe_out.json","w"), indent=1, default=str)
