import json, time, urllib.request
from datetime import datetime, timezone
iso = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "pm5-paper-bot/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r: return json.loads(r.read().decode())
now = int(time.time()); cur = now//300*300
starts = [cur - 300*k for k in range(3, 203)]
off = {}
for i in range(0, len(starts), 50):
    q = "&".join(f"slug=btc-updown-5m-{s}" for s in starts[i:i+50])
    for m in get(f"https://gamma-api.polymarket.com/markets?closed=true&limit=100&{q}"):
        p = json.loads(m["outcomePrices"]); s = int(m["slug"].rsplit("-",1)[1])
        off[s] = {"up": p[0], "closed": m.get("closed"), "uma": m.get("umaResolutionStatus"), "updated": m.get("updatedAt"), "closedTime": m.get("closedTime")}
rows = []
for s in starts:
    try:
        cp = get(f"https://polymarket.com/api/crypto/crypto-price?symbol=BTC&eventStartTime={iso(s)}&variant=fiveminute&endDate={iso(s+300)}")
    except Exception as e:
        cp = {"err": repr(e)}
    o = off.get(s, {})
    ref = None if not cp.get("closePrice") else ("UP" if cp["closePrice"] >= cp["openPrice"] else "DOWN")
    offi = None if o.get("up") not in ("1","0") else ("UP" if o["up"] == "1" else "DOWN")
    rows.append({"s": s, "open": cp.get("openPrice"), "close": cp.get("closePrice"), "ref": ref, "official": offi, "diff_bp": None if not cp.get("closePrice") else round((cp["closePrice"]-cp["openPrice"])/cp["openPrice"]*1e4, 3), "o": o})
agree = sum(1 for r in rows if r["ref"] and r["official"] and r["ref"] == r["official"])
dis = [r for r in rows if r["ref"] and r["official"] and r["ref"] != r["official"]]
json.dump({"n": len(rows), "agree": agree, "disagree": len(dis), "dis": dis[:30], "sample": rows[:5], "missing_official": sum(1 for r in rows if not r["official"])}, open("probe_out.json","w"), indent=1)
