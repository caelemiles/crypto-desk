import json, time, urllib.request, asyncio, datetime as dt
out = {}
def get(name, url, data=None, n=3000):
    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode() if data else None,
              headers={"User-Agent": "Mozilla/5.0 pm5-probe", "Content-Type": "application/json", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read().decode()
            out[name] = {"status": r.status, "body": body[:n]}
            return json.loads(body)
    except Exception as e:
        out[name] = {"error": repr(e)[:300]}
now = int(time.time()); cur = now // 300 * 300
out["now"] = now; out["cur"] = cur
iso = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
ev = get("gamma_event_cur", f"https://gamma-api.polymarket.com/events?slug=btc-updown-5m-{cur}", n=8000)
mk = get("gamma_market_cur", f"https://gamma-api.polymarket.com/markets?slug=btc-updown-5m-{cur}", n=8000)
get("gamma_market_next", f"https://gamma-api.polymarket.com/markets?slug=btc-updown-5m-{cur+300}", n=1500)
for k in (1, 2, 4, 12):
    get(f"gamma_market_prev{k}", f"https://gamma-api.polymarket.com/markets?slug=btc-updown-5m-{cur-300*k}", n=6000)
    get(f"cryptoprice_prev{k}", f"https://polymarket.com/api/crypto/crypto-price?symbol=BTC&eventStartTime={iso(cur-300*k)}&variant=fiveminute&endDate={iso(cur-300*k+300)}")
get("cryptoprice_cur", f"https://polymarket.com/api/crypto/crypto-price?symbol=BTC&eventStartTime={iso(cur)}&variant=fiveminute&endDate={iso(cur+300)}")
try:
    m = mk[0]; toks = json.loads(m["clobTokenIds"]); cid = m["conditionId"]
    get("clob_book_up", f"https://clob.polymarket.com/book?token_id={toks[0]}", n=5000)
    get("clob_books", "https://clob.polymarket.com/books", data=[{"token_id": toks[0]}, {"token_id": toks[1]}], n=2000)
    get("clob_fee", f"https://clob.polymarket.com/fee-rate?token_id={toks[0]}")
    get("clob_market", f"https://clob.polymarket.com/markets/{cid}", n=3000)
    get("clob_tick", f"https://clob.polymarket.com/tick-size?token_id={toks[0]}")
    get("clob_hist", f"https://clob.polymarket.com/prices-history?market={toks[0]}&interval=1h&fidelity=1", n=1500)
    get("data_trades", f"https://data-api.polymarket.com/trades?market={cid}&limit=5", n=2500)
    get("clob_lasttrade", f"https://clob.polymarket.com/last-trade-price?token_id={toks[0]}")
except Exception as e:
    out["tok_err"] = repr(e)
get("series", "https://gamma-api.polymarket.com/series?slug=btc-up-or-down-5m", n=1500)
get("bn_vision_1m", "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=2")
get("bn_vision_depth", "https://data-api.binance.vision/api/v3/depth?symbol=BTCUSDT&limit=5", n=600)
get("bn_vision_trades", "https://data-api.binance.vision/api/v3/aggTrades?symbol=BTCUSDT&limit=2")
get("bn_fapi", "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT")
get("okx_funding", "https://www.okx.com/api/v5/public/funding-rate?instId=BTC-USDT-SWAP")
get("okx_oi", "https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId=BTC-USDT-SWAP")
get("okx_liq", "https://www.okx.com/api/v5/public/liquidation-orders?instType=SWAP&uly=BTC-USDT&state=filled&limit=2", n=1500)
get("okx_taker", "https://www.okx.com/api/v5/rubik/stat/taker-volume?ccy=BTC&instType=CONTRACTS&period=5m", n=600)
get("bybit", "https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT", n=1500)
get("hyperliquid", "https://api.hyperliquid.xyz/info", data={"type": "metaAndAssetCtxs"}, n=1500)
get("coinbase", "https://api.exchange.coinbase.com/products/BTC-USD/ticker")
get("coinbase_candles", "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=60", n=400)
get("kraken", "https://api.kraken.com/0/public/Ticker?pair=XBTUSD", n=600)
get("deribit", "https://www.deribit.com/api/v2/public/get_index_price?index_name=btc_usd")

async def ws(name, url, subs, secs=8, ping=None):
    import websockets
    msgs = []
    try:
        async with websockets.connect(url, open_timeout=10, max_size=2**22) as w:
            for s in subs: await w.send(json.dumps(s) if not isinstance(s, str) else s)
            end = time.time() + secs
            while time.time() < end:
                try:
                    m = await asyncio.wait_for(w.recv(), timeout=2)
                    msgs.append(m[:700] if isinstance(m, str) else repr(m[:200]))
                except asyncio.TimeoutError:
                    if ping: await w.send(ping)
        out[name] = {"n": len(msgs), "msgs": msgs[:8]}
    except Exception as e:
        out[name] = {"error": repr(e)[:300], "msgs": msgs[:4]}

async def allws():
    await ws("rtds_chainlink", "wss://ws-live-data.polymarket.com", [{"action": "subscribe", "subscriptions": [{"topic": "crypto_prices_chainlink", "type": "*", "filters": "{\"symbol\":\"btc/usd\"}"}]}], ping="PING")
    await ws("rtds_binance", "wss://ws-live-data.polymarket.com", [{"action": "subscribe", "subscriptions": [{"topic": "crypto_prices", "type": "update", "filters": "btcusdt"}]}], ping="PING")
    await ws("v2_noauth", "wss://ws-live-v2.polymarket.com/ws", [{"op": "subscribe", "rid": "s1", "subscriptions": [{"channel": "price.crypto", "filter": {"symbol": "btcusd"}}]}])
    try:
        toks = json.loads(mk[0]["clobTokenIds"])
        await ws("clob_ws", "wss://ws-subscriptions-clob.polymarket.com/ws/market", [{"assets_ids": toks, "type": "market"}], ping="PING")
    except Exception as e:
        out["clob_ws"] = repr(e)
asyncio.run(allws())
json.dump(out, open("probe_out.json", "w"), indent=1)
