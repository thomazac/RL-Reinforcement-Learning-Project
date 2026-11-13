"""TTB v1 is frozen: it must reproduce its saved outputs exactly, forever.

If this fails, something the observation inputs depend on changed (this module, numpy, or RocketSim).
Don't regenerate the reference; find the change. Policies trained on v1 inputs depend on it.
"""

from pathlib import Path

import numpy as np

import pytest

from playstyle import ttb_v1, versions

REF = Path(__file__).parent / "fixtures" / "ttb_v1_reference.npz"


def test_reference_is_v1():
    assert str(np.load(REF)["version"]) == ttb_v1.VERSION == "ttb-v1"


def test_reproduces_reference_exactly():
    ref = np.load(REF)
    predictor = ttb_v1.BallPredictor()
    for i in range(len(ref["tick"])):
        ball = ref["ball"][i]
        slices = predictor.predict(ball[0], ball[1], ball[2], int(ref["tick"][i]))
        np.testing.assert_array_equal(slices, ref["slices"][i], err_msg=f"ball prediction, step {i}")
        ttb = ttb_v1.estimate_ttb(ref["pos"][i], ref["vel"][i], ref["fwd"][i], ref["boost"][i],
                                  ref["demoed"][i], slices, ref["on_ground"][i])
        np.testing.assert_array_equal(ttb, ref["ttb"][i], err_msg=f"TTB, step {i}")


def test_cache_tolerances():
    """Recorded play never lands near the tolerances, so pin them directly: a ball nudged just inside them
    reuses the cached path, one nudged just outside gets a fresh prediction."""
    predictor = ttb_v1.BallPredictor()
    pos, vel, ang = np.array([0.0, 0.0, 500.0]), np.array([800.0, 300.0, 200.0]), np.zeros(3)
    predictor.predict(pos, vel, ang, 0)
    cached_pos, cached_vel = predictor._cache_pos[2].astype(float), predictor._cache_vel[2].astype(float)
    cached = predictor._cache_pos[2:2 + predictor._n_out:predictor._stride].copy()
    for dp, dv, reuse in ((0.09, 0, True), (0.11, 0, False), (0, 0.49, True), (0, 0.51, False)):
        p = ttb_v1.BallPredictor()
        p.predict(pos, vel, ang, 0)
        nudged_pos, nudged_vel = cached_pos + [dp, 0, 0], cached_vel + [dv, 0, 0]
        out = p.predict(nudged_pos, nudged_vel, ang, 8)
        fresh = ttb_v1.BallPredictor().predict(nudged_pos, nudged_vel, ang)
        np.testing.assert_array_equal(out, cached if reuse else fresh, err_msg=f"dp={dp} dv={dv}")


def test_checkpoint_versions(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="no versions.json"):
        versions.check(tmp_path)
    versions.write(tmp_path, "live")
    versions.check(tmp_path)
    assert versions.reward_versions("live")["ttb"].startswith("ttb-live-")
    assert versions.reward_versions("v1") == {"ttb": "ttb-v1"}
    monkeypatch.setitem(versions.OBS_VERSIONS, "ttb", "ttb-v2")
    with pytest.raises(RuntimeError, match="trained on"):
        versions.check(tmp_path)


def test_demoed_car_is_unreachable():
    ref = np.load(REF)
    ttb = ttb_v1.estimate_ttb(ref["pos"][0], ref["vel"][0], ref["fwd"][0], ref["boost"][0],
                              np.ones(4, bool), ref["slices"][0], ref["on_ground"][0])
    assert np.all(np.isinf(ttb))
