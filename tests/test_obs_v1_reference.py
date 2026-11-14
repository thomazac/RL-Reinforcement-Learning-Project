"""The observation format and teammate profile are frozen: they must reproduce the recorded outputs.

Tolerance 1e-6 rather than exact, so a vectorized rewrite may sum in a different order. If this fails,
don't regenerate the reference; find what changed. Policies trained on v1 observations depend on it.
"""

import lzma
import pickle
from pathlib import Path

import numpy as np
import pytest

from playstyle import profile_v1
from playstyle.obs import EntityObs
from playstyle.versions import OBS_VERSIONS

REF = Path(__file__).parent / "fixtures" / "obs_v1_reference.pkl.xz"
TOL = 1e-6


@pytest.fixture(scope="module")
def ref():
    with lzma.open(REF, "rb") as f:
        return pickle.load(f)


def test_reference_is_current_version(ref):
    assert ref["versions"] == OBS_VERSIONS


def test_reference_covers_demos_goals_and_touches(ref):
    rows = ref["rows"]
    assert any(c.is_demoed for r in rows for c in r["state"].cars.values())
    assert sum(bool(r["info"]["match"]["goal"]) for r in rows) >= 1
    assert any(c.ball_touches > 0 for r in rows for c in r["state"].cars.values())


def test_observations_match(ref):
    builder = EntityObs()
    for i, r in enumerate(ref["rows"]):
        out = builder.build_obs(sorted(r["state"].cars), r["state"], r["info"])
        for agent, expected in r["obs"].items():
            np.testing.assert_allclose(out[agent], expected, atol=TOL, rtol=0, err_msg=f"step {i}, {agent}")


def test_profile_matches(ref):
    tracker, last_tick = profile_v1.ProfileTracker(), None
    for i, r in enumerate(ref["rows"]):
        s, info = r["state"], r["info"]
        if i == 0:
            values = tracker.resume(s, info["profile"])
        elif r["start"]:
            values = tracker.reset(s)
        else:
            values = tracker.update(s, info["obs_ttb"], (s.tick_count - last_tick) / 120)
        last_tick = s.tick_count
        for agent, expected in info["profile"].items():
            np.testing.assert_allclose(values[agent], expected, atol=TOL, rtol=0, err_msg=f"step {i}, {agent}")
