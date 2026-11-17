#!/usr/bin/env python3
"""Practice session: download recent candles, let the brain practise every setup, write playbook.json.

  python evolve.py                 # 60 days of real Binance candles (46 to study, the last 14 as the exam)
  python evolve.py --synthetic     # offline test on made-up prices (nothing should pass the exam)
"""
import argparse, json, os, time
import brain, engine as E

ROOT = os.path.dirname(os.path.abspath(__file__))
PLAYBOOK = os.path.join(ROOT, "playbook.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--val-days", type=int, default=14)
    ap.add_argument("--budget", type=int, default=1500, help="seconds")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default=PLAYBOOK)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    global SEED; SEED = a.seed
    n = a.days * 96
    if a.synthetic:
        t0 = (int(time.time() * 1000) // E.BAR - n - 2) * E.BAR
        spec = {"BTC": (70000, .004), "ETH": (3000, .005), "SOL": (140, .006), "DOGE": (.15, .007), "PEPE": (.00001, .009), "WIF": (1.5, .01)}
        data = {c: [[b[0], b[1], b[2], b[3], b[4], b[5] * 5e7 / s[0]] for b in E.synth(n, i + SEED, drift=0, vol=s[1], start=s[0], t0=t0)]
                for i, (c, s) in enumerate(spec.items())}
    else:
        import bot as B
        end = int(time.time() * 1000) // E.BAR * E.BAR
        data = {}
        for c in E.COINS:
            try:
                rows = [r for r in B.fetch(c, end - n * E.BAR, end - 1) if r[0] < end]
            except Exception as e:  # noqa: BLE001
                print(f"skip {c}: {e}"); continue
            if len(rows) > n // 2: data[c] = rows
    prev = None
    if os.path.exists(a.out):
        with open(a.out) as f: prev = json.load(f)
    pb = brain.evolve(data, prev, val_days=a.val_days, budget_s=a.budget)
    if a.synthetic: pb["synthetic"] = True
    with open(a.out, "w") as f:
        json.dump(pb, f, indent=1)
    print(pb["summary"], f"({pb['seconds']}s)")


if __name__ == "__main__":
    main()
