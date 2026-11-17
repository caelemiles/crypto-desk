"""Small shared helpers: HTTP with retries, time helpers, safe JSON files."""
import gzip
import json
import math
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

UA = {"User-Agent": "pm5-paper-bot/1.0 (paper trading research; no orders)", "Accept": "application/json"}
SGT = timezone(timedelta(hours=8))


class FetchError(Exception):
    pass


def http_json(url, data=None, timeout=8, retries=2, backoff=0.6):
    """GET (or POST when data is given) a JSON endpoint, retrying transient failures."""
    body = json.dumps(data).encode() if data is not None else None
    headers = dict(UA)
    if body is not None:
        headers["Content-Type"] = "application/json"
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (400, 401, 403, 404, 410, 451):
                break  # not transient
        except (urllib.error.URLError, TimeoutError, ValueError, ConnectionError, OSError) as e:
            last = e
        if attempt < retries:
            time.sleep(backoff * (attempt + 1))
    raise FetchError(f"{url.split('?')[0]}: {last!r}"[:240])


def now_s():
    return time.time()


def iso_utc(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sgt_day(ts):
    return datetime.fromtimestamp(ts, SGT).strftime("%Y-%m-%d")


def sgt_hm(ts):
    return datetime.fromtimestamp(ts, SGT).strftime("%H:%M:%S")


def load_json(path, default):
    try:
        if path.endswith(".gz"):
            with gzip.open(path, "rt") as f:
                return json.load(f)
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError, EOFError):
        return default


def clean(o):
    """Make an object JSON-safe: no NaN/inf, round floats."""
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
        return round(o, 6)
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, "item"):  # numpy scalar
        return clean(o.item())
    return o


def save_json(path, data, gz=False):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    payload = json.dumps(clean(data), separators=(",", ":"), allow_nan=False)
    if gz:
        with gzip.open(tmp, "wt") as f:
            f.write(payload)
    else:
        with open(tmp, "w") as f:
            f.write(payload)
    os.replace(tmp, path)


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(p):
    p = clamp(p, 1e-4, 1 - 1e-4)
    return math.log(p / (1 - p))


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def money(v):
    """Signed dollars: +$1.23 / -$4.56."""
    v = float(v or 0)
    return ("+" if v >= 0 else "-") + f"${abs(v):,.2f}"
