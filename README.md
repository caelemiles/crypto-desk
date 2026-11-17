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

---

# PM5 – Polymarket BTC 5-minute AI (paper trading)

A second, separate bot that watches every Polymarket "Bitcoin Up or Down" 5-minute market around the clock,
forecasts each one, paper-trades it against the real order book, and keeps improving itself under strict testing.
**Paper money only:** no wallet, no keys, no code that places orders.

**Dashboard:** `https://caelemiles.github.io/crypto-desk/pm5/`

## How it works

| Component | File | What it does |
|---|---|---|
| Central controller | `pm5/run.py` | one loop, once a second; coordinates everything below; saves and hands over every ~5h40m |
| Market Intelligence | `pm5/market.py` | finds the live market by its exact window, reads both order books, tracks open → close → resolution, refuses wrong or expired windows |
| Data Intelligence | `pm5/feeds.py` | Chainlink BTC/USD (the settlement reference) via Polymarket's live socket, Binance 1-second candles + order book, ETH, Coinbase, OKX funding / open interest / liquidations / taker flow |
| Prediction Engine | `pm5/features.py`, `pm5/models.py` | TWAP-aware fair value (these markets settle on Chainlink's **60-second TWAP**), logistic models, blends, LightGBM; every model is a frozen version |
| Strategy Development | `pm5/strategy.py` | rules for when, which side, how much (fractional Kelly), and early exits |
| Trading Engine | `pm5/paper.py` | walks the real book level by level, partial fills, 5-share minimum, Polymarket's crypto taker fee (0.07 × p × (1−p) per share), settlement at $1/$0, early exits at the bid |
| Learning Engine | `pm5/learning.py` | evaluates every market, analyses every trade, proposes experiments, replay-tests them, shadow-tests them live, promotes only significant winners, rolls back on deterioration |
| Learning Journal | `pm5/journal.py`, `pm5/report.py` | live activity feed, trade analyses, daily reports, strategy experiments, discoveries (measured vs hypothesis), model history |

History (Chainlink, Binance, order-book snapshots, every 5-second checkpoint, results) is stored in SQLite (Actions cache).
The AI's memory (models, strategies, paper account, journal) is on the `pm5-data` branch, updated every few minutes.

## Controls

- **Settings:** edit `pm5/settings.json` (starting balance, risk limits, test thresholds). Re-read every ~10 minutes.
  Change `reset_token` to restart the paper account; learning is kept.
- **Pause / resume:** Actions → "PM5 BTC 5-min paper AI" → Disable / Enable workflow.
- **Offline test:** `python pm5/sim_test.py 6` runs the real controller for 6 simulated hours on a synthetic market.
