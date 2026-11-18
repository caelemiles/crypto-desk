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

**Dashboards:** BTC `https://caelemiles.github.io/crypto-desk/pm5/` · ETH `https://caelemiles.github.io/crypto-desk/pm5/?asset=eth`

The same code runs twice: `pm5.yml` (BTC, memory on `pm5-data`) and `pm5-eth.yml` (ETH, `PM5_ASSET=eth`, memory on `pm5-eth-data`, settings in `pm5/settings-eth.json`).

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
| Pattern search | `pm5/patterns.py` | every 5 min: a measured account of why each market moved; every hour: tests which signals beat the market price, turns real patterns into strategies |
| Learning Journal | `pm5/journal.py`, `pm5/report.py` | live activity feed, trade analyses, daily reports, strategy experiments, discoveries (measured vs hypothesis), model history |

History (Chainlink, Binance, order-book snapshots, every 5-second checkpoint, results) is stored in SQLite (Actions cache).
The AI's memory (models, strategies, paper account, journal) is on the `pm5-data` branch, updated every few minutes.

## Controls

- **Settings:** edit `pm5/settings.json` (starting balance, risk limits, test thresholds). Re-read every ~10 minutes.
  Change `reset_token` to restart the paper account; learning is kept.
- **Pause / resume:** Actions → "PM5 BTC 5-min paper AI" → Disable / Enable workflow.
- **Offline test:** `python pm5/sim_test.py 6` runs the real controller for 6 simulated hours on a synthetic market.

---

# PMX – Polymarket Weather + Arbitrage AI (paper trading)

Two more paper-trading desks in one bot (`pmx/`), run by `.github/workflows/pmx.yml` around the clock.
**Paper money only.** Polymarket lists Singapore and the United States (where GitHub runs the bots) as close-only:
no new positions on the website or the API. So no bot in this repository has a real-money mode.

**Dashboard:** `https://caelemiles.github.io/crypto-desk/pmx/` (memory and data on the `pmx-data` branch)

| Desk | What it does | How it learns |
|---|---|---|
| Weather (`pmx/wdesk.py`, `pmx/weather.py`) | Daily "highest temperature in <city>" markets. Reads the exact airport station each market settles on (Weather Underground / METAR), what it has measured so far today, and GFS / ECMWF / ICON ensembles from Open-Meteo; prices every bucket and buys YES or NO when the model beats the order book. | After every settled day: each model's error per station, the station's bias, extra uncertainty, how Polymarket rounds, and whether its probabilities beat the market's (Brier score). Stations where the market is better get blocked. |
| Arbitrage (`pmx/adesk.py`) | Scans the active events every minute for sets that are guaranteed to pay more than they cost after fees: every YES of an exclusive event under $1, every NO under N-1, and "ladders" (YES on a lower level + NO on a higher level: worst case the margin, best case double). | Every gap: how long it lasts, how much the price moves between the first and last leg (leg risk, simulated with a second book read), which categories produce gaps. The margin it demands includes the leg risk it has measured. |

Both desks run a **strategy lab** (`pmx/lab.py`): up to 12 rule versions trade every opportunity in the shadows on the
real order book; a challenger replaces the champion only after beating it on profit and return over the same period
with 40+ settled trades and a confidence interval above zero. Only the champion's trades go to the paper account.

Each desk has a **target** (`pmx/readiness.py`, thresholds in `pmx/settings.json`): days running, settled trades,
return, t-stat, recent profit, drawdown. The BTC and ETH 5-minute dashboards show the same kind of checklist.
Reaching it means "proven on paper"; it does not switch anything to real money.

- **Settings:** `pmx/settings.json` (re-read every ~10 minutes). Change `reset_token` to restart both paper accounts.
- **Pause / resume:** Actions → "PMX weather + arbitrage paper AI" → Disable / Enable workflow.
- **Offline test:** `python pmx/sim_test.py 6` runs the real controller for 6 simulated days on a fake market.
