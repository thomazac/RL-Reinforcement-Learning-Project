"""Versions of everything that shapes the policy's inputs, recorded with every checkpoint.

A checkpoint may only be loaded with the same versions it was trained on.
"""

import json
from pathlib import Path

from . import ttb_v1

OBS_VERSIONS = {"ttb": ttb_v1.VERSION}
FILENAME = "obs_versions.json"


def write(checkpoint_dir):
    Path(checkpoint_dir, FILENAME).write_text(json.dumps(OBS_VERSIONS, indent=2))


def check(checkpoint_dir):
    """Raise unless the checkpoint was trained on the current observation versions."""
    path = Path(checkpoint_dir, FILENAME)
    if not path.exists():
        raise RuntimeError(f"{checkpoint_dir} has no {FILENAME}; refusing to guess which inputs it was trained on")
    saved = json.loads(path.read_text())
    if saved != OBS_VERSIONS:
        raise RuntimeError(f"{checkpoint_dir} was trained on {saved}, current inputs are {OBS_VERSIONS}")
