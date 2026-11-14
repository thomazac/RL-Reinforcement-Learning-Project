"""Time the observation builder and profile update on the recorded reference states (microseconds per env step)."""

import lzma
import pickle
import time

from playstyle import profile_v1
from playstyle.obs import EntityObs


def main(repeats=20):
    with lzma.open("tests/fixtures/obs_v1_reference.pkl.xz", "rb") as f:
        rows = pickle.load(f)["rows"]
    builder, tracker = EntityObs(), profile_v1.ProfileTracker()
    agents = sorted(rows[0]["state"].cars)
    builder.reset(agents, rows[0]["state"], rows[0]["info"])
    best_obs = best_profile = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        for r in rows:
            builder.build_obs(agents, r["state"], r["info"])
        best_obs = min(best_obs, time.perf_counter() - t)
        tracker.reset(rows[0]["state"])
        t = time.perf_counter()
        for r in rows:
            tracker.update(r["state"], r["info"]["obs_ttb"], 8 / 120)
        best_profile = min(best_profile, time.perf_counter() - t)
    n = len(rows)
    print(f"observations {best_obs / n * 1e6:.0f} us/step   profile {best_profile / n * 1e6:.0f} us/step   (4 agents)")


if __name__ == "__main__":
    main()
