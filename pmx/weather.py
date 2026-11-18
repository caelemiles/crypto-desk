"""Weather intelligence: reads Polymarket's highest-temperature markets, gets forecasts for the exact airport
station each market settles on, turns them into a probability for every temperature bucket, and learns from
every settled day how wrong each forecast model was for each station.

Sources (all free, no key): Open-Meteo ensembles (GFS 31 members, ECMWF 51, ICON 40) and single-run models,
aviationweather.gov METAR observations (what the airport actually measured so far today).
"""
import math
import re
from datetime import datetime, timedelta, timezone

import core as K

MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                           "september", "october", "november", "december"])}
ENS_MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless"]
DET_MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless"]
LEADS = ("0", "1", "2")   # same day, next day, two or more days ahead

# ------------------------------------------------------------------ parsing Polymarket weather events
RE_TITLE = re.compile(r"highest temperature in (.+?) on ", re.I)
RE_SLUGDATE = re.compile(r"-on-([a-z]+)-(\d{1,2})-(\d{4})")
RE_ICAO = [re.compile(r"aviationweather\.gov/api/data/metar\?ids=([A-Z0-9]{4})"),
           re.compile(r"\b(?:Airport|Station|Aerodrome|Airfield)[^()\n]{0,40}\(([A-Z]{4})\)"),
           re.compile(r"wunderground\.com/history/daily/(?:[A-Za-z0-9\-]+/)*?([A-Z]{4})(?=[/?\s\"'.,)]|$)")]
RE_BETWEEN = re.compile(r"between\s+(-?\d+)\s*(?:-|to|–)\s*(-?\d+)\s*°?\s*([FC])", re.I)
RE_BELOW = re.compile(r"(-?\d+)\s*°?\s*([FC])\s+or\s+(?:below|lower|less)", re.I)
RE_ABOVE = re.compile(r"(-?\d+)\s*°?\s*([FC])\s+or\s+(?:higher|above|more)", re.I)
RE_EXACT = re.compile(r"\bbe\s+(-?\d+)\s*°\s*([FC])\b", re.I)


def parse_bucket(text):
    """'between 84-85°F' -> (84, 85, 'F'); '77°F or below' -> (None, 77, 'F'); '19°C' -> (19, 19, 'C')."""
    m = RE_BETWEEN.search(text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return min(a, b), max(a, b), m.group(3).upper()
    m = RE_BELOW.search(text)
    if m:
        return None, int(m.group(1)), m.group(2).upper()
    m = RE_ABOVE.search(text)
    if m:
        return int(m.group(1)), None, m.group(2).upper()
    m = RE_EXACT.search(text)
    if m:
        v = int(m.group(1))
        return v, v, m.group(2).upper()
    return None


def is_weather(ev):
    return bool(RE_TITLE.search(ev["title"])) and "lowest" not in ev["title"].lower()


def parse_weather_event(ev):
    """A tradeable description of a highest-temperature event, or (None, reason)."""
    m = RE_TITLE.search(ev["title"])
    if not m:
        return None, "not a highest-temperature event"
    city = m.group(1).strip()
    d = RE_SLUGDATE.search(ev["slug"])
    if not d or d.group(1) not in MONTHS:
        return None, "date not found in slug"
    date = f"{int(d.group(3)):04d}-{MONTHS[d.group(1)]:02d}-{int(d.group(2)):02d}"
    text = ev["description"] + " " + ev.get("resolution_source", "") + " " + " ".join(x["description"] for x in ev["markets"][:2])
    st = None
    for rx in RE_ICAO:
        st = rx.search(text)
        if st:
            break
    if not st:
        return None, "could not find the airport station the market settles on"
    buckets, units = [], set()
    for mk in ev["markets"]:
        b = parse_bucket(mk["question"]) or parse_bucket(mk["group"])
        if not b:
            return None, f"could not read bucket '{mk['question'][:60]}'"
        lo, hi, unit = b
        units.add(unit)
        buckets.append({"market": mk["id"], "lo": lo, "hi": hi, "label": _label(lo, hi, unit), "mk": mk})
    if len(units) != 1:
        return None, "mixed units"
    buckets.sort(key=lambda b: (b["lo"] if b["lo"] is not None else -1e9))
    return {"slug": ev["slug"], "event_id": ev["id"], "title": ev["title"], "city": city, "date": date,
            "station": st.group(1), "unit": units.pop(), "buckets": buckets, "neg_risk": ev["neg_risk"],
            "end": ev["end"]}, None


def _label(lo, hi, u):
    if lo is None:
        return f"{hi}°{u} or below"
    if hi is None:
        return f"{lo}°{u} or higher"
    return f"{lo}°{u}" if lo == hi else f"{lo}-{hi}°{u}"


# ------------------------------------------------------------------ data feeds
def metar(station, hours=36):
    """Routine observations: [(epoch, temp °C)]. Polymarket settles on the highest hourly temperature remark
    (T-group, tenths of °C) of routine METARs on the local date; SPECI reports do not count. Without a T-group
    the whole-degree body temperature is used, exactly as the market rules say."""
    rows = K.http(f"{K.AWC}/metar?ids={station}&format=json&hours={hours}", timeout=12) or []
    out, lat, lon = [], None, None
    for r in rows:
        t = r.get("obsTime")
        temp = r.get("temp")
        raw = r.get("rawOb") or ""
        if str(r.get("metarType") or "").upper() == "SPECI" or raw.startswith("SPECI"):
            continue
        mt = re.search(r"\bT([01])(\d{3})[01]\d{3}\b", raw)
        if mt:
            temp = (-1 if mt.group(1) == "1" else 1) * int(mt.group(2)) / 10.0
        if t is None or temp is None:
            continue
        lat, lon = r.get("lat", lat), r.get("lon", lon)
        out.append((float(t), float(temp)))
    out.sort()
    return out, lat, lon


def _hourly(url):
    return K.http(url, timeout=20)


def forecasts(lat, lon, unit):
    """{'tz_offset': s, 'det': {model: {iso_hour: temp}}, 'ens': {model: [ {iso_hour: temp}, ... members ]}}"""
    tu = "&temperature_unit=fahrenheit" if unit == "F" else ""
    base = f"latitude={lat:.4f}&longitude={lon:.4f}&hourly=temperature_2m&timezone=auto&forecast_days=4&past_days=1{tu}"
    out = {"tz_offset": None, "det": {}, "ens": {}}
    try:
        d = _hourly(f"{K.OPEN_METEO}?{base}&models={','.join(DET_MODELS)}")
        out["tz_offset"] = d.get("utc_offset_seconds")
        h = d.get("hourly") or {}
        times = h.get("time") or []
        for mdl in DET_MODELS:
            vals = h.get(f"temperature_2m_{mdl}") or (h.get("temperature_2m") if len(DET_MODELS) == 1 else None)
            if vals:
                out["det"][mdl] = {t: v for t, v in zip(times, vals) if v is not None}
    except K.FetchError:
        pass
    for mdl in ENS_MODELS:
        try:
            d = _hourly(f"{K.ENSEMBLE}?{base}&models={mdl}")
        except K.FetchError:
            continue
        out["tz_offset"] = out["tz_offset"] if out["tz_offset"] is not None else d.get("utc_offset_seconds")
        h = d.get("hourly") or {}
        times = h.get("time") or []
        mem = []
        for k, vals in h.items():
            if k == "temperature_2m" or k.startswith("temperature_2m_member"):
                mem.append({t: v for t, v in zip(times, vals) if v is not None})
        if mem:
            out["ens"][mdl] = mem
    return out


def local_day(ts, tz_offset):
    return datetime.fromtimestamp(ts + (tz_offset or 0), timezone.utc).strftime("%Y-%m-%d")


def local_hour_iso(ts, tz_offset):
    return datetime.fromtimestamp(ts + (tz_offset or 0), timezone.utc).strftime("%Y-%m-%dT%H:00")


def lead_of(date, ts, tz_offset):
    today = datetime.strptime(local_day(ts, tz_offset), "%Y-%m-%d")
    return (datetime.strptime(date, "%Y-%m-%d") - today).days


def day_max(series, date, from_iso=None):
    vals = [v for t, v in series.items() if t.startswith(date) and (from_iso is None or t >= from_iso)]
    return max(vals) if vals else None


# ------------------------------------------------------------------ calibration memory (learned per station)
DEFAULT_SIGMA_C = {"0": 0.7, "1": 1.1, "2": 1.6}   # extra uncertainty on top of the ensemble spread, °C


def to_c(x, unit):
    return x / 1.8 if unit == "F" else x


class Calibration:
    """What the AI has learned about each station: bias per lead, error of each model, extra uncertainty,
    and whether Polymarket's resolution rounds or truncates."""

    def __init__(self, d=None):
        d = d or {}
        self.st = d.get("stations", {})      # station -> {"unit", "samples": {lead: [..]}, "rounding": {...}}
        self.brier = d.get("brier", [])      # [ts, station, ours, market, outcome] per bucket at decision time

    def to_dict(self):
        return {"stations": self.st, "brier": self.brier[-6000:]}

    def station(self, s, unit="F"):
        return self.st.setdefault(s, {"unit": unit, "samples": {k: [] for k in LEADS}, "round": 0, "floor": 0})

    def _samples(self, s, lead):
        return self.station(s)["samples"].get(lead, [])

    def bias(self, s, lead):
        xs = [x["r"] for x in self._samples(s, lead)]
        return sum(xs) / (len(xs) + 4.0) if xs else 0.0   # shrunk towards zero until there is evidence

    def model_weights(self, s, lead):
        errs = {}
        for x in self._samples(s, lead):
            for m, e in (x.get("m") or {}).items():
                errs.setdefault(m, []).append(e)
        w = {}
        for m in DET_MODELS:
            es = errs.get(m, [])
            mse = (sum(e * e for e in es) + 4 * 2.0 ** 2) / (len(es) + 4)   # prior: 2 degree RMS error
            w[m] = 1.0 / max(mse, 0.05)
        tot = sum(w.values())
        return {m: v / tot for m, v in w.items()}

    def sigma(self, s, lead, unit):
        """Extra spread (in the station's unit) so that past forecast errors would have been well calibrated."""
        prior = DEFAULT_SIGMA_C[lead] * (1.8 if unit == "F" else 1.0)
        xs = self._samples(s, lead)
        if not xs:
            return prior
        b = self.bias(s, lead)
        resid = sum((x["r"] - b) ** 2 for x in xs) / len(xs)
        ens = sum(x.get("ev", 0.0) for x in xs) / len(xs)
        est = math.sqrt(max(resid - ens, (0.35 * (1.8 if unit == "F" else 1.0)) ** 2))
        k = 6.0
        return math.sqrt((len(xs) * est ** 2 + k * prior ** 2) / (len(xs) + k))

    def rounding(self, s):
        st = self.station(s)
        return "floor" if st["floor"] > st["round"] + 2 else "round"

    def learn(self, s, unit, lead, actual, snap, rounding_obs=None):
        st = self.station(s, unit)
        st["unit"] = unit
        r = {"ts": K.now(), "r": round(actual - snap["center"], 3), "ev": round(snap.get("ens_var", 0.0), 3),
             "m": {m: round(actual - v, 3) for m, v in (snap.get("models") or {}).items()}}
        lst = st["samples"].setdefault(lead, [])
        lst.append(r)
        del lst[:-60]
        if rounding_obs in ("round", "floor"):
            st[rounding_obs] += 1
        return r


# ------------------------------------------------------------------ probability model
def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bucket_interval(b, mode):
    lo, hi = b["lo"], b["hi"]
    if mode == "floor":
        a = -math.inf if lo is None else lo
        z = math.inf if hi is None else hi + 1
    else:
        a = -math.inf if lo is None else lo - 0.5
        z = math.inf if hi is None else hi + 0.5
    return a, z


def forecast_state(fc, date, now_ts, obs_max, cal, station, unit):
    """Turn raw forecasts into a distribution of today's/target day's maximum. Returns None if not enough data."""
    tz = fc.get("tz_offset") or 0
    lead_n = lead_of(date, now_ts, tz)
    if lead_n < 0:
        return None
    lead = str(min(lead_n, 2))
    from_iso = local_hour_iso(now_ts, tz) if lead_n == 0 else None
    weights = cal.model_weights(station, lead)
    members, models_full = [], {}
    for mdl, mem in fc.get("ens", {}).items():
        vals = [day_max(s, date, from_iso) for s in mem]
        vals = [v for v in vals if v is not None]
        full = [day_max(s, date) for s in mem]
        full = [v for v in full if v is not None]
        if vals:
            w = weights.get(mdl, 1 / 3) / len(vals)
            members += [(v, w) for v in vals]
        if full:
            models_full[mdl + "_ens"] = sum(full) / len(full)
    for mdl, s in fc.get("det", {}).items():
        v = day_max(s, date, from_iso)
        f = day_max(s, date)
        if f is not None:
            models_full[mdl] = f
        if v is not None:
            members.append((v, weights.get(mdl, 1 / 3) * 0.5))   # one run counts half as much as a whole ensemble
    if not members and obs_max is None:
        return None
    tot = sum(w for _, w in members) or 1.0
    members = [(v, w / tot) for v, w in members]
    det_full = {m: v for m, v in models_full.items() if not m.endswith("_ens")}
    ens_full = [v for m, v in models_full.items() if m.endswith("_ens")]
    if det_full:
        center = sum(weights.get(m, 1 / 3) * v for m, v in det_full.items()) / sum(weights.get(m, 1 / 3) for m in det_full)
    elif ens_full:
        center = sum(ens_full) / len(ens_full)
    else:
        center = obs_max
    mean = sum(v * w for v, w in members) if members else obs_max
    var = sum(w * (v - mean) ** 2 for v, w in members) if members else 0.0
    bias = cal.bias(station, lead)
    sig = cal.sigma(station, lead, unit)
    if lead_n == 0 and obs_max is not None:
        # later in the day less can still change; shrink the extra uncertainty with the hours left
        hours_left = max(1, 24 - int(datetime.fromtimestamp(now_ts + tz, timezone.utc).hour))
        sig *= max(0.35, min(1.0, hours_left / 14.0))
    return {"lead": lead, "lead_n": lead_n, "members": members, "bias": bias, "sigma": sig, "obs_max": obs_max,
            "center": center, "models": det_full, "ens_mean": mean, "ens_var": var,
            "spread_c": to_c(math.sqrt(var + sig * sig), unit), "rounding": cal.rounding(station)}


def bucket_probs(state, buckets):
    """P(reported maximum falls in each bucket). The final maximum can never be below what was already observed."""
    mode = state["rounding"]
    obs = state["obs_max"]
    e = max(state["sigma"], 0.05)
    out = []
    for b in buckets:
        a, z = bucket_interval(b, mode)
        p = 0.0
        if state["members"]:
            for v, w in state["members"]:
                y = v + state["bias"]
                hi = 1.0 if z == math.inf else _phi((z - y) / e)
                lo = 0.0 if a == -math.inf else _phi((a - y) / e)
                if obs is not None:
                    if z <= obs:
                        pi = 0.0
                    elif a <= obs:
                        pi = hi          # everything that would have ended below obs ends at obs instead
                    else:
                        pi = hi - lo
                else:
                    pi = hi - lo
                p += w * max(0.0, pi)
        else:   # no forecast left (end of day): the observed maximum decides
            p = 1.0 if (obs is not None and a <= obs < z) else 0.0
        out.append(p)
    s = sum(out)
    return [min(0.995, max(0.001, x / s)) for x in out] if s > 0 else [1.0 / len(out)] * len(out)


def actual_from(buckets, winner_idx, metar_max, cal, station):
    """Best estimate of the true maximum for learning, and which rounding rule the result is consistent with."""
    b = buckets[winner_idx]
    rounding = None
    if metar_max is not None:
        r_ok = bucket_interval(b, "round")
        f_ok = bucket_interval(b, "floor")
        in_r = r_ok[0] <= metar_max < r_ok[1]
        in_f = f_ok[0] <= metar_max < f_ok[1]
        if in_r != in_f:
            rounding = "round" if in_r else "floor"
        if in_r or in_f:
            return metar_max, rounding
    if b["lo"] is None or b["hi"] is None:
        return None, None
    return (b["lo"] + b["hi"]) / 2.0, None


def next_dates(ts, days=3):
    d = datetime.fromtimestamp(ts, timezone.utc)
    return [(d + timedelta(days=i)) for i in range(-1, days + 1)]
