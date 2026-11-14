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


def test_hand_built_block_covers_what_play_does_not(ref):
    rows = ref["hand_rows"]
    assert 50 <= len(rows) <= 100
    cars = [c for r in rows for c in r["state"].cars.values()]
    balls = [r["state"].ball for r in rows]
    match = [r["info"]["match"] for r in rows]
    speed = lambda v: float(np.linalg.norm(v))
    assert any(c.physics.position[2] > 1900 for c in cars)  # near the ceiling
    assert any(c.physics.up[2] < -0.9 and not c.on_ground for c in cars)  # upside down in the air
    assert any(c.has_flipped and not c.has_flip and c.has_jumped for c in cars)  # mid-flip, nothing left
    assert any(abs(c.physics.position[1]) > 5120 for c in cars) and any(abs(b.position[1]) > 5120 for b in balls)
    assert any(speed(b.linear_velocity) > 5999 for b in balls) and any(speed(b.angular_velocity) > 5.99 for b in balls)
    far = max(speed(a.physics.position - b.physics.position) for r in rows
              for a in r["state"].cars.values() for b in r["state"].cars.values())
    assert far > 11500
    assert any(m["overtime"] for m in match) and any(m["time_remaining"] == 0 and not m["overtime"] for m in match)
    assert {m["score_diff"] for m in match} >= {3, -3} and min(m["score_diff"] for m in match) < -3
    demoed = [{team: sum(c.is_demoed for c in r["state"].cars.values() if c.team_num == team) for team in (0, 1)}
              for r in rows]
    assert any(d[1] == 2 and d[0] == 0 for d in demoed) and any(d[0] == 2 and d[1] == 0 for d in demoed)
    assert any(d[0] == 1 for d in demoed) and any(d[1] == 1 for d in demoed)  # a demoed teammate, either side
    assert any(not r["info"]["prev_action"] for r in rows)  # no previous action yet


def test_observations_match(ref):
    builder = EntityObs()
    for i, r in enumerate(ref["rows"] + ref["hand_rows"]):
        out = builder.build_obs(sorted(r["state"].cars), r["state"], r["info"])
        for agent, expected in r["obs"].items():
            np.testing.assert_allclose(out[agent], expected, atol=TOL, rtol=0, err_msg=f"step {i}, {agent}")


def test_profile_matches(ref):
    tracker, last_tick = profile_v1.ProfileTracker(), None
    for i, r in enumerate(ref["rows"]):
        s, info = r["state"], r["info"]
        if i == 0:
            values = tracker.resume(r["profile_snapshot"])
        elif r["start"]:
            values = tracker.reset(s)
        else:
            values = tracker.update(s, info["obs_ttb"], (s.tick_count - last_tick) / 120)
        last_tick = s.tick_count
        for agent, expected in info["profile"].items():
            np.testing.assert_allclose(values[agent], expected, atol=TOL, rtol=0, err_msg=f"step {i}, {agent}")


def test_profile_matches_on_hand_built_states(ref):
    """Each hand-built state is one update from its own recorded tracker state."""
    tracker = profile_v1.ProfileTracker()
    for i, r in enumerate(ref["hand_rows"]):
        tracker.resume(r["profile_before"])
        values = tracker.update(r["state"], r["info"]["obs_ttb"], r["dt"])
        for agent, expected in r["info"]["profile"].items():
            np.testing.assert_allclose(values[agent], expected, atol=TOL, rtol=0, err_msg=f"{r['name']} {i}, {agent}")
