"""Settings. PAPER TRADING ONLY: nothing in this project can place a real order.

User-editable values live in pm5/settings.json (re-read every ~10 minutes from GitHub).
"""
import json
import os

from util import http_json

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("PM5_WORK", os.path.join(ROOT, "work"))   # runtime folder (not in git)
DB_PATH = os.path.join(WORK, "history.sqlite")
STATE_PATH = os.path.join(WORK, "state.json.gz")
PUBLISH_DIR = os.environ.get("PM5_PUBLISH", os.path.join(WORK, "publish"))  # pushed to the pm5-data branch

PAPER_ONLY = True  # hard-wired. There is deliberately no live execution code anywhere.
WINDOW = 300       # seconds in one market
TWAP_SECONDS = 60  # these markets settle on Chainlink's 60-second TWAP (read from the market at runtime too)
CHECK_EVERY = 5    # decision checkpoints every 5 seconds inside a window
LAST_ENTRY_GUARD = 6  # never enter in the last 6 seconds (orders could land after the close)

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
CRYPTO_PRICE = "https://polymarket.com/api/crypto/crypto-price"
RTDS_WS = "wss://ws-live-data.polymarket.com"
BINANCE = ["https://data-api.binance.vision/api/v3"]
COINBASE = "https://api.exchange.coinbase.com"
OKX = "https://www.okx.com/api/v5"

DEFAULTS = {
    "starting_balance": 1000.0,
    "reset_token": "1",
    "max_stake_pct": 0.03,
    "daily_loss_limit_pct": 0.08,
    "max_losses_in_a_row": 6,
    "kelly_fraction": 0.25,
    "liquidity_take_pct": 0.6,
    "min_shadow_markets_for_promotion": 150,
    "min_shadow_trades_for_promotion": 25,
    "max_challengers": 10,
}

LIMITS = {  # guard rails so a typo in settings.json cannot do anything silly
    "starting_balance": (10, 1_000_000),
    "max_stake_pct": (0.001, 0.10),
    "daily_loss_limit_pct": (0.01, 0.5),
    "max_losses_in_a_row": (2, 50),
    "kelly_fraction": (0.05, 1.0),
    "liquidity_take_pct": (0.1, 1.0),
    "min_shadow_markets_for_promotion": (50, 5000),
    "min_shadow_trades_for_promotion": (10, 1000),
    "max_challengers": (2, 30),
}


def _sanitize(raw):
    s = dict(DEFAULTS)
    for k, v in (raw or {}).items():
        if k not in DEFAULTS:
            continue
        if k == "reset_token":
            s[k] = str(v)
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        lo, hi = LIMITS[k]
        s[k] = min(max(v, lo), hi)
    for k in ("max_losses_in_a_row", "min_shadow_markets_for_promotion", "min_shadow_trades_for_promotion", "max_challengers"):
        s[k] = int(s[k])
    return s


def load_settings():
    """Prefer the live copy on GitHub (so edits apply without a restart), else the local file."""
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        try:
            return _sanitize(http_json(f"https://raw.githubusercontent.com/{repo}/main/pm5/settings.json", retries=1)), "github"
        except Exception:  # noqa: BLE001
            pass
    try:
        with open(os.path.join(ROOT, "settings.json")) as f:
            return _sanitize(json.load(f)), "file"
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULTS), "defaults"
