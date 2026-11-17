import json, time, urllib.request
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "pm5-paper-bot/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r: return json.loads(r.read().decode())
prev = json.load(open("probe_out.json"))
now = int(time.time()); cur = now//300*300
starts = [cur - 300*k for k in range(3, 203)]
off = {}
for i in range(0, len(starts), 50):
    q = "&".join(f"slug=btc-updown-5m-{s}" for s in starts[i:i+50])
    for m in get(f"https://gamma-api.polymarket.com/markets?closed=true&limit=100&{q}"):
        p = json.loads(m["outcomePrices"]); s = int(m["slug"].rsplit("-",1)[1])
        if p[0] in ("1","0"): off[s] = 1 if p[0] == "1" else 0
bn = {}
t = min(starts) - 120
while t < max(starts) + 400:
    d = get(f"https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1s&startTime={t*1000}&limit=1000")
    if not d: break
    for k in d: bn[int(k[0])//1000] = float(k[4])
    t = int(d[-1][0])//1000 + 1
def px(s):
    for k in range(5):
        if s-k in bn: return bn[s-k]
def avg(a, b):  # mean over seconds (a, b]
    v = [bn[x] for x in range(a+1, b+1) if x in bn]; return sum(v)/len(v) if v else None
H = {
 "H1 twap60(E) >= twap60(S)": lambda S,E: avg(E-60,E) >= avg(S-60,S),
 "H2 mean[S,E] >= spot(S)": lambda S,E: avg(S,E) >= px(S),
 "H3 mean[S,E] >= twap60(S)": lambda S,E: avg(S,E) >= avg(S-60,S),
 "H4 spot(E) >= spot(S)": lambda S,E: px(E) >= px(S),
 "H5 twap60(E) >= spot(S)": lambda S,E: avg(E-60,E) >= px(S),
 "H6 twap60(E) >= twap60(S) shifted -30s": lambda S,E: avg(E-90,E-30) >= avg(S-90,S-30),
}
res = {}
for name, f in H.items():
    ok = n = 0
    for s in starts:
        if s not in off: continue
        try:
            v = f(s, s+300)
        except TypeError:
            continue
        n += 1; ok += (1 if v else 0) == off[s]
    res[name] = [ok, n, round(ok/max(n,1),3)]
json.dump({"hyp": res, "n_official": len(off), "n_bn": len(bn)}, open("probe_out2.json","w"), indent=1)
