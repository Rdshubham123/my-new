"""Global identity manager: links closed tracklets to persistent identities.

Links only when the best candidate is good AND clearly better than the runner-up;
otherwise it creates a new identity and records the ambiguous candidates.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Tracklet:
    tid: int
    cls: str
    start_pts: float          # seconds
    end_pts: float
    exit_xy: tuple            # normalised (0..1) position at last observation
    entry_xy: tuple           # normalised position at first observation
    emb: np.ndarray           # L2-normalised mean embedding


@dataclass
class Identity:
    gid: int
    cls: str
    last_pts: float
    last_xy: tuple
    emb: np.ndarray
    n_tracklets: int = 1
    status: str = "confirmed"
    candidates: list = field(default_factory=list)


class IdentityManager:
    def __init__(self, t_max=300.0, max_speed=0.15, w_app=0.6, w_mot=0.4,
                 tau_link=0.65, tau_margin=0.10, db_path=":memory:"):
        # max_speed: normalised frame-widths per second; t_max: longest gap considered
        self.t_max, self.max_speed = t_max, max_speed
        self.w_app, self.w_mot = w_app, w_mot
        self.tau_link, self.tau_margin = tau_link, tau_margin
        self.ids: dict[int, Identity] = {}
        self._next = 1
        self.db = sqlite3.connect(db_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS links(tid INT, gid INT, score REAL, margin REAL, decision TEXT)")

    def _score(self, ident: Identity, t: Tracklet):
        gap = t.start_pts - ident.last_pts
        if ident.cls != t.cls or gap < 0 or gap > self.t_max:
            return None
        app = float(np.dot(ident.emb, t.emb))               # cosine in [-1, 1]
        app = (app + 1) / 2
        dist = float(np.hypot(ident.last_xy[0] - t.entry_xy[0], ident.last_xy[1] - t.entry_xy[1]))
        reach = max(self.max_speed * max(gap, 1e-3), 0.05)
        mot = max(0.0, 1.0 - dist / reach) if dist <= 3 * reach else 0.0
        return self.w_app * app + self.w_mot * mot

    def assign(self, tracklets: list[Tracklet]) -> dict[int, int]:
        """Process tracklets that started in the same window; returns tid -> gid."""
        out: dict[int, int] = {}
        gids = list(self.ids)
        S = np.full((len(tracklets), len(gids)), -1.0)
        for i, t in enumerate(tracklets):
            for j, g in enumerate(gids):
                s = self._score(self.ids[g], t)
                if s is not None:
                    S[i, j] = s
        taken = set()
        if len(gids) and len(tracklets):
            r, c = linear_sum_assignment(-S)
            match = {i: j for i, j in zip(r, c) if S[i, j] >= 0}
        else:
            match = {}
        for i, t in enumerate(tracklets):
            row = np.sort(S[i][S[i] >= 0])[::-1] if len(gids) else np.array([])
            best = row[0] if len(row) else None
            margin = (row[0] - row[1]) if len(row) > 1 else (row[0] if len(row) else None)
            j = match.get(i)
            if j is not None and best is not None and S[i, j] >= self.tau_link \
                    and margin >= self.tau_margin and gids[j] not in taken:
                g = gids[j]
                taken.add(g)
                ident = self.ids[g]
                ident.emb = _unit(ident.emb * ident.n_tracklets + t.emb)
                ident.n_tracklets += 1
                ident.last_pts, ident.last_xy = t.end_pts, t.exit_xy
                out[t.tid] = g
                self.db.execute("INSERT INTO links VALUES(?,?,?,?,?)", (t.tid, g, float(S[i, j]), float(margin), "link"))
            else:
                g = self._next; self._next += 1
                ident = Identity(g, t.cls, t.end_pts, t.exit_xy, t.emb)
                if best is not None and best >= self.tau_link:      # plausible but ambiguous
                    ident.status = "uncertain"
                    self.db.execute("INSERT INTO links VALUES(?,?,?,?,?)",
                                    (t.tid, gids[j] if j is not None else -1, float(best), float(margin), "abstain"))
                self.ids[g] = ident
                out[t.tid] = g
        return out


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v
