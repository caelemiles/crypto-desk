# Crypto Desk – paper trader

A self-learning memecoin day-trading bot that trades **paper money only** (no wallet, no keys, no real orders).

- Starts with 100 USDT of paper money and puts a **fixed 10 USDT into every trade**.
- Quick trades on 15-minute candles from Binance: BTC, ETH, SOL, DOGE, PEPE, SHIB, WIF, BONK, FLOKI, PENGU, TRUMP, long and short.
- Learns from every candle on every coin (shadow long and short ideas), measures which setups actually make money after fees, and only trades when its measured edge is positive.
- Runs about every 10 minutes on GitHub Actions. No AI usage.

## Where things are

| Path | What |
|---|---|
| `engine.py` | the trading and learning engine (v5.1) |
| `bot.py` | fetches candles, runs the engine, writes the dashboard data |
| `.github/workflows/trade.yml` | the schedule |
| `docs/index.html` | the dashboard (published with GitHub Pages) |
| `backup/` | a daily backup of the bot's memory and trades |

The bot's working memory lives in the GitHub Actions cache so the repository does not grow every run; a backup is committed once a day.

## Controls

- **Pause:** Actions tab → "Crypto Desk paper trader" → `…` → Disable workflow.
- **Run now:** Actions tab → "Crypto Desk paper trader" → Run workflow.
- **Start over:** delete the `backup/` folder and clear the caches (Actions → Caches).
