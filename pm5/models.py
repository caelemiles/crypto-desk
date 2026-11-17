"""Forecasting models. Every model is a frozen, versioned object: once created its parameters never change.

Kinds
- market : the Polymarket mid price itself (the benchmark every model must beat)
- fair   : TWAP-aware random-walk probability from the Chainlink distance to the price to beat, with a
           temperature (confidence scaling) parameter
- blend  : logistic blend of the fair probability and the market price (fitted weights)
- logit  : regularised logistic regression on a chosen feature set (fitted)
- gbm    : LightGBM gradient-boosted trees (fitted, optional if lightgbm is installed)
"""
import base64
import math
import warnings
import pickle
import time

import numpy as np

import features as F
from util import clamp, logit, sigmoid

P_MIN, P_MAX = 0.01, 0.99

FEATURE_SETS = {
    "core": ["d_open", "tau", "vol_bn_300", "ret_bn_10", "ret_bn_30", "ret_bn_60", "lead_5", "flow_30"],
    "flow": ["ret_bn_10", "ret_bn_30", "ret_bn_60", "ret_bn_180", "lead_5", "flow_30", "flow_120", "volu_60",
             "book_imb", "fut_flow", "liq_imb"],
    "regime": ["vol_ratio", "vol_bn_60", "vol_bn_900", "volu_60", "hour_sin", "hour_cos", "eth_ret_60",
               "eth_ret_300", "funding", "premium", "prev_outcome"],
    "all": [n for n in F.NAMES if n not in ("t", "p_fair", "z_fair", "pm_mid_up", "cl_age")],
}


def _lf(f, name):
    v = F.get(f, name)
    return None if v is None else logit(clamp(v, 0.003, 0.997))


class Model:
    def __init__(self, mid, kind, params, parent=None, note="", trained_until=None, created=None):
        self.id, self.kind, self.params = mid, kind, params
        self.parent, self.note, self.trained_until = parent, note, trained_until
        self.created = created or time.time()
        self._gbm = None

    # ---------------------------------------------------------------- prediction
    def predict(self, f):
        k = self.kind
        if k == "market":
            m = F.get(f, "pm_mid_up")
            return None if m is None else clamp(m, P_MIN, P_MAX)
        lp = _lf(f, "p_fair")
        if lp is None:
            return None
        if k == "fair":
            return clamp(sigmoid(lp / self.params.get("temp", 1.0)), P_MIN, P_MAX)
        if k == "blend":
            lm = _lf(f, "pm_mid_up")
            if lm is None:
                if self.params.get("needs_market", True):
                    return None
                lm = lp
            w = self.params
            return clamp(sigmoid(w["wf"] * lp + w["wm"] * lm + w["b"]), P_MIN, P_MAX)
        if k == "logit":
            x = self._vec(f, lp)
            if x is None:
                return None
            z = float(np.dot(self.params["w"], x) + self.params["b"])
            return clamp(sigmoid(z), P_MIN, P_MAX)
        if k == "gbm":
            x = self._vec(f, lp, standardize=False)
            if x is None:
                return None
            booster = self._booster()
            if booster is None:
                return None
            raw = float(booster.predict(np.array([x]), raw_score=True)[0])
            return clamp(sigmoid(raw + self.params.get("base_w", 1.0) * lp), P_MIN, P_MAX)
        return None

    def _vec(self, f, lp, standardize=True):
        names = self.params["features"]
        use_mkt = self.params.get("use_market", False)
        vals = [lp]
        if use_mkt:
            lm = _lf(f, "pm_mid_up")
            if lm is None:
                return None
            vals.append(lm)
        for n in names:
            v = F.get(f, n)
            vals.append(np.nan if v is None else float(v))
        x = np.array(vals, dtype=float)
        if standardize:
            mu, sd = np.array(self.params["mu"]), np.array(self.params["sd"])
            x = (x - mu) / sd
            x = np.where(np.isnan(x), 0.0, x)
            x = np.clip(x, -6, 6)
        return x

    def _booster(self):
        if self._gbm is None and self.params.get("model_b64"):
            try:
                self._gbm = pickle.loads(base64.b64decode(self.params["model_b64"]))
            except Exception:  # noqa: BLE001 - lightgbm missing or bad blob
                self._gbm = False
        return self._gbm or None

    # ---------------------------------------------------------------- persistence
    def to_dict(self):
        return {"id": self.id, "kind": self.kind, "params": self.params, "parent": self.parent, "note": self.note,
                "trained_until": self.trained_until, "created": self.created}

    @staticmethod
    def from_dict(d):
        return Model(d["id"], d["kind"], d["params"], d.get("parent"), d.get("note", ""), d.get("trained_until"),
                     d.get("created"))

    def describe(self):
        k = self.kind
        if k == "market":
            return "Market price (benchmark)"
        if k == "fair":
            return f"TWAP fair value, temperature {self.params.get('temp', 1):.2f}"
        if k == "blend":
            return f"Fair value + market blend (w_fair {self.params['wf']:.2f}, w_market {self.params['wm']:.2f})"
        if k == "logit":
            fs = self.params.get("fset", "?")
            return f"Logistic regression [{fs}{' + market' if self.params.get('use_market') else ''}], L2 {self.params.get('l2')}"
        if k == "gbm":
            return f"LightGBM trees [{self.params.get('fset')}], {self.params.get('rounds')} rounds"
        return k


# ==================================================================== fitting
def _matrix(rows, names, use_market):
    X, y, keep = [], [], []
    for i, (f, label) in enumerate(rows):
        lp = _lf(f, "p_fair")
        if lp is None:
            continue
        vals = [lp]
        if use_market:
            lm = _lf(f, "pm_mid_up")
            if lm is None:
                continue
            vals.append(lm)
        for n in names:
            v = F.get(f, n)
            vals.append(np.nan if v is None else float(v))
        X.append(vals)
        y.append(label)
        keep.append(i)
    return np.array(X, dtype=float), np.array(y, dtype=float), keep


def fit_logit(rows, fset, l2=1.0, use_market=False, iters=400):
    names = FEATURE_SETS[fset]
    X, y, _ = _matrix(rows, names, use_market)
    if len(y) < 200:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    sd = np.where((sd < 1e-12) | np.isnan(sd), 1.0, sd)
    Z = np.clip(np.where(np.isnan((X - mu) / sd), 0.0, (X - mu) / sd), -6, 6)
    n, d = Z.shape
    w, b = np.zeros(d), 0.0
    # Newton / IRLS with L2: converges in a few steps and is deterministic
    for _ in range(25):
        z = Z @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g_w = Z.T @ (p - y) / n + l2 * w / n
        g_b = float(np.mean(p - y))
        s = p * (1 - p)
        H = (Z.T * s) @ Z / n + l2 * np.eye(d) / n
        Hb = float(np.mean(s)) + 1e-9
        try:
            step = np.linalg.solve(H + 1e-8 * np.eye(d), g_w)
        except np.linalg.LinAlgError:
            break
        w -= step
        b -= g_b / Hb
        if np.max(np.abs(step)) < 1e-6:
            break
    return {"features": names, "fset": fset, "l2": l2, "use_market": use_market, "w": w.tolist(), "b": b,
            "mu": mu.tolist(), "sd": sd.tolist()}


def fit_blend(rows):
    X, y, _ = _matrix(rows, [], True)
    if len(y) < 300:
        return None
    Z = X[:, :2]
    w, b = np.array([1.0, 0.0]), 0.0
    n = len(y)
    for _ in range(25):
        p = 1 / (1 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        s = p * (1 - p)
        A = np.c_[Z, np.ones(n)]
        H = (A.T * s) @ A / n + 1e-3 * np.eye(3)
        g = A.T @ (p - y) / n
        step = np.linalg.solve(H, g)
        w -= step[:2]
        b -= step[2]
        if np.max(np.abs(step)) < 1e-7:
            break
    return {"wf": float(w[0]), "wm": float(w[1]), "b": float(b), "needs_market": True}


def fit_gbm(rows, fset, rounds=150, leaves=15):
    try:
        import lightgbm as lgb
    except ImportError:
        return None
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
    names = FEATURE_SETS[fset]
    X, y, _ = _matrix(rows, names, False)
    if len(y) < 3000:
        return None
    base = X[:, 0]
    ds = lgb.Dataset(np.ascontiguousarray(X), label=y, init_score=base, free_raw_data=True)
    booster = lgb.train({"objective": "binary", "num_leaves": leaves, "learning_rate": 0.03, "min_data_in_leaf": 200,
                         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
                         "verbose": -1, "seed": 7}, ds, num_boost_round=rounds)
    blob = base64.b64encode(pickle.dumps(booster)).decode()
    return {"features": names, "fset": fset, "rounds": rounds, "leaves": leaves, "base_w": 1.0, "model_b64": blob}


def fit_temperature(rows):
    """Best single confidence scaling for the fair model (1-D search on log loss)."""
    lps = [(_lf(f, "p_fair"), y) for f, y in rows if F.get(f, "p_fair") is not None]
    if len(lps) < 200:
        return None
    best, best_ll = 1.0, 1e9
    for temp in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.4, 3.0]:
        ll = 0.0
        for lp, y in lps:
            p = clamp(sigmoid(lp / temp), 1e-4, 1 - 1e-4)
            ll -= y * math.log(p) + (1 - y) * math.log(1 - p)
        if ll < best_ll:
            best, best_ll = temp, ll
    return best
