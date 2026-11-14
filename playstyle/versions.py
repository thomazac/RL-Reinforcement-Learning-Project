"""Versions of everything that shapes the policy's inputs and rewards, recorded with every checkpoint.

A checkpoint may only be loaded with the same observation versions it was trained on. Reward versions may
change mid-training; they are recorded so each run says which one it used.
"""

import hashlib
import json
from pathlib import Path

from . import profile_v1, ttb_v1
from .obs import OBS_LAYOUT

OBS_VERSIONS = {"layout": OBS_LAYOUT, "ttb": ttb_v1.VERSION, "profile": profile_v1.VERSION}
REWARD_TTB = {"v1": ttb_v1.VERSION, "live": "ttb-live"}
FILENAME = "versions.json"
_LIVE_SOURCES = ("ttb.py", "ball_prediction.py")


def reward_versions(reward_ttb):
    """The live estimator keeps changing, so it is identified by a hash of its source."""
    if reward_ttb == "v1":
        return {"ttb": REWARD_TTB["v1"]}
    digest = hashlib.sha1(b"".join(Path(__file__).with_name(f).read_bytes() for f in _LIVE_SOURCES)).hexdigest()
    return {"ttb": f"{REWARD_TTB['live']}-{digest[:10]}"}


def write(checkpoint_dir, reward_ttb):
    Path(checkpoint_dir, FILENAME).write_text(json.dumps(
        {"obs_versions": OBS_VERSIONS, "reward_versions": reward_versions(reward_ttb)}, indent=2))


def check(checkpoint_dir):
    """Raise unless the checkpoint was trained on the current observation versions."""
    path = Path(checkpoint_dir, FILENAME)
    if not path.exists():
        raise RuntimeError(f"{checkpoint_dir} has no {FILENAME}; refusing to guess which inputs it was trained on")
    saved = json.loads(path.read_text())["obs_versions"]
    if saved != OBS_VERSIONS:
        raise RuntimeError(f"{checkpoint_dir} was trained on {saved}, current inputs are {OBS_VERSIONS}")
