"""Publishes the dashboard data and the bot's memory to the `pm5-data` branch of the repository.

- live.json  : small, every ~90 seconds (live AI feed, current market, open trades)
- data.json  : everything the dashboard shows, every ~10 minutes
- state.json : the bot's full memory (models, strategies, account, journal), every ~10 minutes and at shutdown

History on that branch is squashed once a day so the repository does not grow.
"""
import json
import os
import shutil
import subprocess

from util import clean


class Publisher:
    def __init__(self, folder, repo=None, token=None, branch="pm5-data"):
        self.dir, self.repo, self.token, self.branch = folder, repo, token, branch
        self.enabled = bool(repo and token)
        self.last_squash_day = None
        self.failures = 0

    def _git(self, *args, check=True, timeout=120):
        return subprocess.run(["git", "-C", self.dir, *args], check=check, capture_output=True, text=True, timeout=timeout)

    def init(self):
        """Clone the data branch (restoring the bot's memory) or start an empty one."""
        if os.path.isdir(self.dir):
            shutil.rmtree(self.dir)
        os.makedirs(self.dir, exist_ok=True)
        if not self.enabled:
            return False
        url = f"https://x-access-token:{self.token}@github.com/{self.repo}.git"
        r = subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", self.branch, url, self.dir],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            shutil.rmtree(self.dir, ignore_errors=True)
            os.makedirs(self.dir, exist_ok=True)
            self._git("init", "-q")
            self._git("checkout", "-q", "-b", self.branch)
            self._git("remote", "add", "origin", url)
        self._git("config", "user.name", "pm5-bot")
        self._git("config", "user.email", "pm5-bot@users.noreply.github.com")
        return True

    def read(self, name, default=None):
        try:
            with open(os.path.join(self.dir, name)) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return default

    def write(self, name, obj, indent=None):
        path = os.path.join(self.dir, name)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(clean(obj), f, separators=(",", ":") if indent is None else None, indent=indent, allow_nan=False)
        os.replace(tmp, path)

    def push(self, message, squash_day=None):
        if not self.enabled:
            return False
        try:
            self._git("add", "-A")
            if squash_day and squash_day != self.last_squash_day:
                # start a fresh single-commit history once a day
                self._git("checkout", "-q", "--orphan", "squash-tmp")
                self._git("add", "-A")
                self._git("commit", "-q", "-m", f"pm5 data (history squashed {squash_day})")
                self._git("branch", "-D", self.branch, check=False)
                self._git("branch", "-m", self.branch)
                self._git("push", "-q", "-f", "origin", f"HEAD:refs/heads/{self.branch}", timeout=180)
                self.last_squash_day = squash_day
                self.failures = 0
                return True
            if self._git("diff", "--cached", "--quiet", check=False).returncode == 0:
                return True
            self._git("commit", "-q", "-m", message)
            r = self._git("push", "-q", "origin", f"HEAD:refs/heads/{self.branch}", check=False, timeout=180)
            if r.returncode != 0:
                # someone else (or a previous run) moved the branch: we are the only writer, so force it
                self._git("push", "-q", "-f", "origin", f"HEAD:refs/heads/{self.branch}", timeout=180)
            self.failures = 0
            return True
        except (subprocess.SubprocessError, OSError) as e:
            self.failures += 1
            print(f"publish failed ({self.failures}): {e}", flush=True)
            return False
