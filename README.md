# Crypto Desk – paper trader

A self-learning memecoin day-trading bot that trades **paper money only** (no wallet, no keys, no real orders).

- Starts with 100 USDT of paper money. Since v6, each trade is **sized by risk**: about 2% of equity at risk, scaled by confidence and the recent record (1%-3%), up to 1x equity per trade with 2x paper leverage.
- Keeps a **learning journal** (every closed trade, every tuning change, coin+side pairs benched for losing, and a daily report card) shown on the dashboard.
- Quick trades on 15-minute candles from Binance: BTC, ETH, SOL, DOGE, PEPE, SHIB, WIF, BONK, FLOKI, PENGU, TRUMP, long and short.
- Learns from every candle on every coin (shadow long and short ideas), measures which setups actually make money after fees, and only trades when its measured edge is positive.
- **Protections (v7):** pauses new entries for 2h after 3 losses in a row, stops opening trades for the day after a 6% daily loss, waits 1h before re-trading a coin it just lost on, and skips coins with under 1M USDT traded in 24h.
- **Backtester:** replays this exact engine on past Binance candles (fees and slippage included, no peeking at future candles) and shows the result on the dashboard.
- Runs about every 10 minutes on GitHub Actions. No AI usage.

## Where things are

| Path | What |
|---|---|
| `engine.py` | the trading and learning engine (v7) |
| `backtest.py` | replays the engine on past candles, writes `docs/backtest.json` |
| `bot.py` | fetches candles, runs the engine, writes the dashboard data |
| `.github/workflows/trade.yml` | the schedule |
| `.github/workflows/backtest.yml` | the backtest (manual, plus every Sunday) |
| `docs/index.html` | the dashboard (published with GitHub Pages) |
| `backup/` | a daily backup of the bot's memory and trades |

The bot's working memory lives in the GitHub Actions cache so the repository does not grow every run; a backup is committed once a day.

## Controls

- **Pause:** Actions tab → "Crypto Desk paper trader" → `…` → Disable workflow.
- **Run now:** Actions tab → "Crypto Desk paper trader" → Run workflow.
- **Backtest:** Actions tab → "Crypto Desk backtest" → Run workflow. Pick the number of days, and optionally try other settings, e.g. `atrStop=2 rr=2` or `riskPct=0.01`. The result appears on the dashboard within about 10 minutes.
- **Start over:** delete the `backup/` folder and clear the caches (Actions → Caches).
