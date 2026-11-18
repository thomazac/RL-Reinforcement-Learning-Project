"""Are the bots hitting the ball or farming contact? Compares a checkpoint with the GC1 2s replays.

Bot touches are counted two ways: contact steps (what the Phase 1a reward and the run log count; a car
pushing the ball scores every step) and distinct touches (a run of consecutive contact steps by one car
counts once, which is how the replay touch events count a carry). Hit strength is the ball speed change
across a distinct touch, the same quantity the replay events report.

Also reports clumping, goals, how episodes end, and the reward-side ball prediction cache: how often it
recomputes and what a recompute costs.

    python scripts/touch_check.py --checkpoint <folder> [--minutes 30]
"""

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from playstyle.ball_prediction import TICK_RATE
from playstyle.env import NO_TOUCH_SECONDS, TICK_SKIP, make_env
from playstyle.rewards import Phase1bReward

DT = TICK_SKIP / TICK_RATE
WEAK = 500.0  # uu/s: below this ball speed change a touch is a tap or push
NEAR = 1000.0  # uu: "at the ball"


def describe(speed_changes):
    s = np.asarray(speed_changes)
    return (f"median {np.median(s):5.0f} uu/s, p25 {np.percentile(s, 25):5.0f}, p75 {np.percentile(s, 75):5.0f}, "
            f"weak (<{WEAK:.0f}) {np.mean(s < WEAK):5.1%}")


def human(pattern):
    import subtr_actor as sa

    minutes, touches, goals, changes, n_replays = 0.0, 0, 0, [], 0
    for f in sorted(glob.glob(pattern)):
        try:
            headers = dict(sa.get_replay_meta(f)["replay_meta"]["all_headers"])
            ev = [e for e in sa.get_stats_events(f)["events"] if e["meta"]["stream"] == "touch"]
        except Exception as e:  # a few replays fail to parse
            print(f"skip {Path(f).name}: {e}")
            continue
        if headers.get("TeamSize") != 2 or "TotalSecondsPlayed" not in headers:
            print(f"skip {Path(f).name}: no team size or game length in the header")
            continue
        minutes += headers["TotalSecondsPlayed"] / 60
        goals += headers.get("Team0Score", 0) + headers.get("Team1Score", 0)
        touches += len(ev)
        changes += [e["payload"]["payload"]["ball_speed_change"] for e in ev]
        n_replays += 1
    print(f"Humans, {n_replays} GC1 2s replays, {minutes:.0f} game-minutes")
    print(f"  touches per player per minute  {touches / 4 / minutes:5.2f}")
    print(f"  goals per 5 minutes            {goals / minutes * 5:5.2f}")
    print(f"  hit strength                   {describe(changes)}")
    return touches / 4 / minutes


def bots(checkpoint, minutes, seed):
    import torch
    from watch import load_policy

    actor = load_policy(checkpoint, "cuda" if torch.cuda.is_available() else "cpu")
    # Clean eval: whole matches from kickoff (no practice starts), scored with the Phase 1b reward for its events.
    env = make_env(coordination=True, reward_fn=Phase1bReward(), seed=seed)
    names = env.reward_fn.names
    fires = np.zeros(len(names))
    predictor = env.shared_info_provider.obs_predictor
    calls = {"hit": [], "miss": []}
    original = predictor.predict

    def timed(*a, **kw):
        before = predictor._cache_pos
        start = time.perf_counter()
        out = original(*a, **kw)
        calls["miss" if predictor._cache_pos is not before else "hit"].append(time.perf_counter() - start)
        return out

    predictor.predict = timed
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs = env.reset()
    agents = list(env.agents)
    steps, contact_steps, distinct, changes = 0, 0, 0, []
    in_contact = {a: False for a in agents}
    start_vel = {}
    near_counts, all_near, goals = [], 0, 0
    ends = {"match over": 0, "no-touch timeout": 0, "max length": 0}
    since_touch = 0.0
    prev_ball_vel = np.array(env.state.ball.linear_velocity)
    total = int(minutes * 60 / DT)
    while steps < total:
        with torch.no_grad():
            chosen, _ = actor.get_action(None, [obs[a] for a in agents])
        obs, rewards, term, trunc = env.step({a: np.asarray(c).reshape(1) for a, c in zip(agents, chosen)})
        fires += sum((rewards[a][1:] != 0) for a in agents)
        state = env.state
        ball, ball_vel = state.ball.position, np.array(state.ball.linear_velocity)
        steps += 1
        touched_any = False
        for a in agents:
            car = state.cars[a]
            touching = car.ball_touches > 0
            touched_any |= touching
            contact_steps += touching
            if touching and not in_contact[a]:
                distinct += 1
                start_vel[a] = prev_ball_vel
            elif in_contact[a] and not touching:
                changes.append(float(np.linalg.norm(prev_ball_vel - start_vel.pop(a))))
            in_contact[a] = touching
        near = sum(np.linalg.norm(state.cars[a].physics.position - ball) < NEAR
                   for a in agents if not state.cars[a].is_demoed)
        near_counts.append(near)
        all_near += near == 4
        goals += bool(state.goal_scored)
        since_touch = 0.0 if touched_any else since_touch + DT
        prev_ball_vel = ball_vel
        if any(term.values()) or any(trunc.values()):
            if any(term.values()):
                ends["match over"] += 1
            elif since_touch >= NO_TOUCH_SECONDS - DT:
                ends["no-touch timeout"] += 1
            else:
                ends["max length"] += 1
            obs = env.reset()
            in_contact = {a: False for a in agents}
            start_vel, since_touch = {}, 0.0
            prev_ball_vel = np.array(env.state.ball.linear_velocity)

    sim_min = steps * DT / 60
    n_ep = sum(ends.values())
    print(f"Bots, {Path(checkpoint).name}, {sim_min:.0f} sim-minutes, stochastic policy")
    print(f"  contact steps per car per minute {contact_steps / 4 / sim_min:6.2f}   (the Phase 1a reward and run log)")
    print(f"  distinct touches per car per min {distinct / 4 / sim_min:6.2f}   (comparable to the replay count)")
    print(f"  contact steps per distinct touch {contact_steps / max(distinct, 1):6.2f}")
    print(f"  hit strength                     {describe(changes)}")
    print(f"  cars within {NEAR:.0f} uu of the ball   mean {np.mean(near_counts):.2f} of 4; all four {all_near / steps:.1%} of the time"
          f"   (human GC1 live play: 1.04, 0.3%)")
    print(f"  goals per 5 minutes              {goals / sim_min * 5:6.2f}")
    rate = dict(zip(names, fires / (steps * len(agents))))
    per5 = 5 * 60 / DT
    print(f"  saves per player per 5 minutes   {rate['save'] * per5:6.2f}   (human GC1, same definition: 1.28)")
    print(f"  shots per player per 5 minutes   {rate['shot'] * per5:6.2f}   (human GC1, same definition: 2.80)")
    print(f"  double-commit share of time      {2 * rate['double_commit']:6.2%}   empty net {rate['nobody_home']:.1%}")
    print(f"  both teammates near the ball     {rate['both_near']:6.1%}   of the time, per team (human GC1 live play: 3.9%)")
    print(f"  episode ends ({n_ep})            " + ", ".join(f"{k} {v / max(n_ep, 1):.0%}" for k, v in ends.items()))
    hit, miss = np.array(calls["hit"]), np.array(calls["miss"])
    n = len(hit) + len(miss)
    print(f"  ball prediction cache            recompute on {len(miss) / n:.1%} of steps; "
          f"hit {np.median(hit) * 1e6:.0f} us, recompute {np.median(miss) * 1e6:.0f} us (medians); "
          f"recompute share of predict time {miss.sum() / (hit.sum() + miss.sum()):.0%}")
    return distinct / 4 / sim_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--replays", default="data/replays/*.replay")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    h = human(args.replays)
    print()
    b = bots(args.checkpoint, args.minutes, args.seed)
    print(f"\nBot distinct touches are {b / h:.1f}x the human rate")


if __name__ == "__main__":
    main()
