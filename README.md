# Crypto Desk – paper trader

A day-trading bot that learns the way a person does, trading **paper money only** (no wallet, no keys, no real orders).

- **Playbook of human setups:** trend pullbacks, breakouts, opening-range breaks, VWAP reclaims, stop-hunt reversals and momentum bursts, each long and short.
- **Practice every 6 hours** (`evolve.py`): on the last 60 days of 15-minute Binance candles it tries thousands of rule versions (volume needed, stop width, target, holding time, trend/BTC/session filters), studies the first 46 days and then sits an **exam on the last 14 days** it never studied. A setup only goes into the playbook if it made money in both halves of practice, in the exam, and the result is statistically unlikely to be luck (t-stat 2+). Tested: on random prices nothing passes; on prices with a real pattern the right setups pass.
- **Learns live like a trader:** new setups trade at half size (probation), get promoted to full size after 8 profitable live trades, get cut back or sent back to practice when they stop working. Everything is written to the learning journal in plain English.
- **Risk:** about 2% of equity at risk per trade; pauses after 3 losses in a row, stops for the day after a 6% loss, cools off 1h on a coin after a loss, skips thin coins, circuit breaker at 15% drawdown.
- **Backtester** (`backtest.py`) replays the engine on past candles with fees and slippage.
- Coins: BTC, ETH, SOL, DOGE, PEPE, SHIB, WIF, BONK, FLOKI, PENGU, TRUMP. Runs about every 10 minutes on GitHub Actions. No AI usage.

## Where things are

| Path | What |
|---|---|
| `engine.py` | the live trading engine (v8: playbook trader) |
| `brain.py` | the setups, the trade simulator and the practice/exam logic |
| `evolve.py` | a practice session; writes `playbook.json` |
| `playbook.json` | the current playbook (rules, practice and exam results) |
| `backtest.py` | replays the engine on past candles, writes `docs/backtest.json` |
| `bot.py` | fetches candles, runs the engine, writes the dashboard data |
| `.github/workflows/trade.yml` | the schedule |
| `.github/workflows/evolve.yml` | practice session every 6 hours |
| `.github/workflows/backtest.yml` | the backtest (manual, plus every Sunday) |
| `docs/index.html` | the dashboard (published with GitHub Pages) |
| `backup/` | a daily backup of the bot's memory and trades |

The bot's working memory lives in the GitHub Actions cache so the repository does not grow every run; a backup is committed once a day.

## Controls

- **Pause:** Actions tab → "Crypto Desk paper trader" → `…` → Disable workflow.
- **Run now:** Actions tab → "Crypto Desk paper trader" → Run workflow.
- **Practise now:** Actions tab → "Crypto Desk practice session" → Run workflow.
- **Backtest:** Actions tab → "Crypto Desk backtest" → Run workflow. Pick the number of days, and optionally try other settings, e.g. `atrStop=2 rr=2` or `riskPct=0.01`. The result appears on the dashboard within about 10 minutes.
- **Start over:** delete the `backup/` folder and clear the caches (Actions → Caches).
