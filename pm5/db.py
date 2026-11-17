"""Historical data store (SQLite). Nothing is overwritten: new rows are appended, old raw ticks pruned slowly."""
import json
import os
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS markets(
  slug TEXT PRIMARY KEY, start INTEGER, end INTEGER, market_id TEXT, condition_id TEXT,
  up_token TEXT, down_token TEXT, title TEXT, description TEXT, fee_rate REAL, fee_exp REAL,
  twap_secs INTEGER, min_size REAL, tick REAL, open_price REAL, close_price REAL, outcome TEXT,
  outcome_src TEXT, official TEXT, backfill INTEGER DEFAULT 0, resolved_at REAL);
CREATE TABLE IF NOT EXISTS chainlink(ts INTEGER PRIMARY KEY, price REAL);
CREATE TABLE IF NOT EXISTS btc1s(ts INTEGER PRIMARY KEY, o REAL, h REAL, l REAL, c REAL, v REAL, tbv REAL);
CREATE TABLE IF NOT EXISTS eth1s(ts INTEGER PRIMARY KEY, c REAL);
CREATE TABLE IF NOT EXISTS pm_snap(ts REAL, slug TEXT, up_bid REAL, up_ask REAL, dn_bid REAL, dn_ask REAL,
  up_bid_sz REAL, up_ask_sz REAL, dn_bid_sz REAL, dn_ask_sz REAL);
CREATE INDEX IF NOT EXISTS pm_snap_slug ON pm_snap(slug);
CREATE TABLE IF NOT EXISTS ext(ts INTEGER, kind TEXT, value REAL, PRIMARY KEY(ts, kind));
CREATE TABLE IF NOT EXISTS checkpoints(slug TEXT, t INTEGER, ts REAL, schema INTEGER, feats TEXT, book TEXT,
  valid INTEGER, PRIMARY KEY(slug, t));
CREATE TABLE IF NOT EXISTS predictions(slug TEXT PRIMARY KEY, data TEXT);
CREATE TABLE IF NOT EXISTS trades(id TEXT PRIMARY KEY, slug TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS integrity(ts REAL, kind TEXT, detail TEXT);
"""


class DB:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        try:
            self.c = sqlite3.connect(path, timeout=30)
            self.c.execute("PRAGMA integrity_check").fetchone()
        except sqlite3.DatabaseError:
            # corrupted file: keep it aside for inspection and start a clean store
            os.replace(path, path + f".corrupt-{int(time.time())}")
            self.c = sqlite3.connect(path, timeout=30)
        self.c.execute("PRAGMA journal_mode=WAL")
        self.c.execute("PRAGMA synchronous=NORMAL")
        self.c.executescript(SCHEMA)
        self.c.commit()
        self._pending = 0

    def x(self, sql, args=()):
        cur = self.c.execute(sql, args)
        self._pending += 1
        return cur

    def many(self, sql, rows):
        if rows:
            self.c.executemany(sql, rows)
            self._pending += 1

    def q(self, sql, args=()):
        return self.c.execute(sql, args).fetchall()

    def one(self, sql, args=()):
        return self.c.execute(sql, args).fetchone()

    def commit(self):
        if self._pending:
            self.c.commit()
            self._pending = 0

    # ---- markets
    def upsert_market(self, m):
        cols = ["slug", "start", "end", "market_id", "condition_id", "up_token", "down_token", "title",
                "description", "fee_rate", "fee_exp", "twap_secs", "min_size", "tick", "backfill"]
        self.x(f"INSERT INTO markets({','.join(cols)}) VALUES({','.join('?' * len(cols))}) "
               f"ON CONFLICT(slug) DO UPDATE SET " + ",".join(f"{c}=COALESCE(excluded.{c},{c})" for c in cols[1:]),
               [m.get(c) for c in cols])

    def set_open(self, slug, price):
        self.x("UPDATE markets SET open_price=? WHERE slug=? AND open_price IS NULL", (price, slug))

    def set_result(self, slug, open_p, close_p, outcome, src):
        self.x("UPDATE markets SET open_price=COALESCE(?,open_price), close_price=?, outcome=?, outcome_src=?, resolved_at=? "
               "WHERE slug=?", (open_p, close_p, outcome, src, time.time(), slug))

    def set_official(self, slug, official):
        self.x("UPDATE markets SET official=? WHERE slug=?", (official, slug))

    def market(self, slug):
        r = self.c.execute("SELECT * FROM markets WHERE slug=?", (slug,))
        row = r.fetchone()
        if not row:
            return None
        return dict(zip([d[0] for d in r.description], row))

    # ---- checkpoints
    def put_checkpoint(self, slug, t, ts, schema, feats, book, valid):
        self.x("INSERT OR REPLACE INTO checkpoints VALUES(?,?,?,?,?,?,?)",
               (slug, t, ts, schema, json.dumps(feats, separators=(",", ":")),
                json.dumps(book, separators=(",", ":")) if book else None, 1 if valid else 0))

    def labelled_checkpoints(self, schema, since=0, until=None, live_only=False):
        """Resolved checkpoints: (slug, start, t, feats, book, outcome_up)."""
        sql = ("SELECT c.slug, m.start, c.t, c.feats, c.book, m.outcome FROM checkpoints c JOIN markets m ON m.slug=c.slug "
               "WHERE c.schema=? AND c.valid=1 AND m.outcome IS NOT NULL AND m.start>=?")
        args = [schema, since]
        if until is not None:
            sql += " AND m.start<?"
            args.append(until)
        if live_only:
            sql += " AND m.backfill=0"
        sql += " ORDER BY m.start, c.t"
        out = []
        for slug, start, t, f, b, o in self.q(sql, args):
            out.append((slug, start, t, json.loads(f), json.loads(b) if b else None, 1 if o == "UP" else 0))
        return out

    def log_integrity(self, kind, detail):
        self.x("INSERT INTO integrity VALUES(?,?,?)", (time.time(), kind, str(detail)[:400]))

    def prune(self, now):
        self.x("DELETE FROM pm_snap WHERE ts<?", (now - 14 * 86400,))
        self.x("DELETE FROM btc1s WHERE ts<?", (now - 30 * 86400,))
        self.x("DELETE FROM eth1s WHERE ts<?", (now - 30 * 86400,))
        self.x("DELETE FROM chainlink WHERE ts<?", (now - 30 * 86400,))
        self.x("DELETE FROM checkpoints WHERE ts<?", (now - 75 * 86400,))
        self.x("DELETE FROM integrity WHERE ts<?", (now - 30 * 86400,))
        self.commit()

    def close(self):
        self.commit()
        try:
            self.c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.DatabaseError:
            pass
        self.c.close()
