"""Human baselines from the GC1 2s replays, with the same definitions the bots are measured by.

Per player per 5 minutes of play: saves and shots by this project's definitions (playstyle/rewards.py: a touch
that stops a ball predicted into the own net within 1 s / puts the ball on course for the opponent net within 3 s,
from the ball path predicted just before and just after the touch), next to Rocket League's own counts from the
replay header. Crowding over live play: mean cars within NEAR of the ball, and the share of time all four are.

    python scripts/human_baselines.py
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_replay_states import usable_frames
from playstyle.ball_prediction import BallPredictor
from playstyle.rewards import SAVE_HORIZON, SHOT_HORIZON, enters_net
from validate_ttb import player_index

FPS = 30
AROUND = 0.1  # s: the ball path is predicted this long before and after a touch
NEAR = 1000.0


def replay(path, predictor):
    import subtr_actor as sa

    meta, arr = sa.get_ndarray_with_info_from_replay_filepath(
        path, global_feature_adders=["BallRigidBody", "CurrentTime"], fps=FPS)
    rm = meta["replay_meta"]
    headers = dict(rm["all_headers"])
    if len(rm["team_zero"]) != 2 or len(rm["team_one"]) != 2 or "TotalSecondsPlayed" not in headers:
        return None
    players = rm["team_zero"] + rm["team_one"]
    team = [0, 0, 1, 1]
    time, ball = arr[:, 12], arr[:, 0:12]
    cars = arr[:, 13:].reshape(len(arr), 4, 14)
    live = usable_frames(time, ball[:, 0:3], ball[:, 6:9], cars[:, :, 0:3], pre_goal=0.0)
    alive = ~np.all(cars[:, :, 0:3] == 0, axis=2)
    near = (np.linalg.norm(cars[:, :, 0:3] - ball[:, None, 0:3], axis=2) < NEAR) & alive
    counts = near[live].sum(axis=1)
    both = (near[live][:, 0] & near[live][:, 1]).sum() + (near[live][:, 2] & near[live][:, 3]).sum()  # team-frames

    def predicted(t):
        b = ball[min(np.searchsorted(time, t), len(time) - 1)]
        return predictor.predict(b[0:3], b[6:9], b[9:12] / 100)

    saves = shots = 0
    remote_ids = [p["remote_id"] for p in players]
    for e in sa.get_stats_events(path)["events"]:
        m = e["meta"]
        if m["stream"] != "touch":
            continue
        p = player_index(remote_ids, m["primary_player"])
        if p is None:
            continue
        t = m["timing"].get("start_time", m["timing"].get("time"))
        before, after = predicted(t - AROUND), predicted(t + AROUND)
        saves += enters_net(before, team[p], SAVE_HORIZON) and not enters_net(after, team[p], SHOT_HORIZON)
        shots += not enters_net(before, 1 - team[p], SHOT_HORIZON) and enters_net(after, 1 - team[p], SHOT_HORIZON)
    stats = headers.get("PlayerStats", [])
    return dict(minutes=headers["TotalSecondsPlayed"] / 60, saves=saves, shots=shots,
                rl_saves=sum(s.get("Saves", 0) for s in stats), rl_shots=sum(s.get("Shots", 0) for s in stats),
                near_sum=float(counts.sum()), all_four=float((counts == 4).sum()), both_near=float(both),
                frames=len(counts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default="data/replays/*.replay")
    args = ap.parse_args()
    predictor, rows = BallPredictor(), []
    for f in sorted(glob.glob(args.replays)):
        try:
            r = replay(f, predictor)
        except Exception as e:  # a few replays fail to parse
            print(f"skip {Path(f).name}: {e}")
            continue
        if r is None:
            print(f"skip {Path(f).name}: not a 2v2 game or no game length in the header")
            continue
        rows.append(r)
    total = {k: sum(r[k] for r in rows) for k in rows[0]}
    per5 = 5 / (4 * total["minutes"])
    print(f"Humans, {len(rows)} GC1 2s replays, {total['minutes']:.0f} game-minutes; per player per 5 minutes:")
    print(f"  saves  (our definition) {total['saves'] * per5:5.2f}   (Rocket League's count {total['rl_saves'] * per5:5.2f})")
    print(f"  shots  (our definition) {total['shots'] * per5:5.2f}   (Rocket League's count {total['rl_shots'] * per5:5.2f})")
    print(f"Crowding over live play: cars within {NEAR:.0f} uu of the ball mean {total['near_sum'] / total['frames']:.2f} "
          f"of 4; all four {total['all_four'] / total['frames']:.1%} of the time; both teammates of a team "
          f"{total['both_near'] / (2 * total['frames']):.1%} of the time (per team)")


if __name__ == "__main__":
    main()
