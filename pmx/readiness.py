"""The "target reached" gate: the paper results a desk must show before it counts as proven.

Reaching the target does NOT switch anything to real money. Polymarket does not accept new orders from
Singapore or from the United States (GitHub's servers), so this project has no live-trading mode at all.
"""
import core as K

REAL_TRADING = {
    "available": False,
    "reason": "Polymarket lists Singapore and the United States as close-only (no new positions, website and API), "
              "and Singapore's regulator treats Polymarket as unlicensed gambling. GitHub runs this bot in the US. "
              "So there is no real-money mode; the bots keep paper trading and learning.",
}


def _check(label, ok, value):
    return {"label": label, "ok": bool(ok), "value": value}


def weather(desk, s):
    a = desk.acct
    st = a.stats()
    wk = a.stats(K.now() - 7 * 86400)
    days = (K.now() - a.created) / 86400
    checks = [
        _check(f"Running at least {s['target_days']:.0f} days", days >= s["target_days"], f"{days:.1f} days"),
        _check(f"At least {s['target_weather_trades']} settled trades", st["n"] >= s["target_weather_trades"], st["n"]),
        _check(f"Return on money staked at least {s['target_weather_roi']:.0%}", (st["roi"] or -1) >= s["target_weather_roi"],
               f"{(st['roi'] or 0):+.1%}"),
        _check("Profit unlikely to be luck (t-stat 2+)", (st["t_stat"] or 0) >= 2, f"{st['t_stat']:.2f}" if st["t_stat"] else "-"),
        _check("Profitable over the last 7 days", wk["pnl"] > 0, K.U.money(wk["pnl"])),
        _check(f"Worst drop below {s['target_max_drawdown']:.0%}", a.max_dd <= s["target_max_drawdown"], f"{a.max_dd:.1%}"),
    ]
    return {"reached": all(c["ok"] for c in checks), "checks": checks}


def arb(desk, s):
    a = desk.acct
    st = a.stats()
    days = (K.now() - a.created) / 86400
    sl = desk.slips[-300:]
    broken = (sum(1 for x in sl if x[3] <= 0) / len(sl)) if sl else None
    checks = [
        _check(f"Running at least {s['target_days']:.0f} days", days >= s["target_days"], f"{days:.1f} days"),
        _check(f"At least {s['target_arb_sets']} settled sets", st["n"] >= s["target_arb_sets"], st["n"]),
        _check(f"Profit at least ${s['target_arb_profit']:.0f}", st["pnl"] >= s["target_arb_profit"], K.U.money(st["pnl"])),
        _check("No losing sets", st["n"] > 0 and st["wins"] == st["n"], f"{st['n'] - st['wins']} losing"),
        _check("Under 10% of sets broken by leg risk", broken is not None and broken < 0.10,
               f"{broken:.0%}" if broken is not None else "-"),
        _check(f"Worst drop below {s['target_max_drawdown']:.0%}", a.max_dd <= s["target_max_drawdown"], f"{a.max_dd:.1%}"),
    ]
    return {"reached": all(c["ok"] for c in checks), "checks": checks}
