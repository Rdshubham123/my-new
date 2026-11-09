"""Synthetic tests: distinctive fish re-link after a gap; look-alikes trigger abstention."""
import numpy as np
from identity import IdentityManager, Tracklet, _unit

rng = np.random.default_rng(0)
D = 32


def emb(base, noise=0.05):
    return _unit(base + noise * rng.standard_normal(D))


def tl(tid, base, t0, t1, entry, exit_):
    return Tracklet(tid, "fish", t0, t1, exit_, entry, emb(base))


def test_relink_distinct():
    a, b = _unit(rng.standard_normal(D)), _unit(rng.standard_normal(D))
    m = IdentityManager()
    r1 = m.assign([tl(1, a, 0, 10, (0.1, 0.5), (0.2, 0.5)), tl(2, b, 0, 10, (0.8, 0.5), (0.7, 0.5))])
    r2 = m.assign([tl(3, a, 60, 70, (0.25, 0.5), (0.3, 0.5)), tl(4, b, 60, 70, (0.65, 0.5), (0.6, 0.5))])
    assert r2[3] == r1[1] and r2[4] == r1[2], (r1, r2)


def test_abstain_lookalikes():
    base = _unit(rng.standard_normal(D))
    m = IdentityManager()
    r1 = m.assign([tl(1, base, 0, 10, (0.5, 0.5), (0.5, 0.5)), tl(2, base, 0, 10, (0.52, 0.5), (0.52, 0.5))])
    r2 = m.assign([tl(3, base, 60, 70, (0.5, 0.5), (0.5, 0.5))])
    assert r2[3] not in r1.values(), "should not link when two candidates are indistinguishable"
    assert m.ids[r2[3]].status == "uncertain"


if __name__ == "__main__":
    test_relink_distinct(); test_abstain_lookalikes(); print("ok")
