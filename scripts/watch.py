"""Top-down viewer with the TTB and role overlay.

    python scripts/watch.py                                  # live window, scripted bots
    python scripts/watch.py --policy random
    python scripts/watch.py --save runs/roles.gif --seconds 20
    python scripts/watch.py --no-draw --seconds 600           # role-change stats only
    python scripts/watch.py --blue-mode chaser                # watch a deliberately bad team
    python scripts/watch.py --checkpoint runs/train/<run>/checkpoints/ppo/<run>-<id>/<step id> --save game.mp4 --seconds 180

Labels show role (LOCK when the first man is committed), TTB and height. Marker size grows with
height. Red marks show coordination events: double-commit on the penalized car, nobody challenging
on the first man, nobody home next to the goal.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playstyle.ball_prediction import TICK_RATE
from playstyle.env import TICK_SKIP, build_env, make_env
from playstyle.roles import FIRST, MIN_HOLD
from playstyle.scripted import MODES, RoleBot

FIELD_X, FIELD_Y, GOAL_HALF_W = 4096, 5120, 893
COLORS = ("tab:blue", "tab:orange")
DT = TICK_SKIP / TICK_RATE


def car_triangle(pos, fwd, size=170):
    f = fwd[:2] / max(np.linalg.norm(fwd[:2]), 1e-6)
    r = np.array([f[1], -f[0]])
    base = pos[:2] - f * size * 0.6
    return [pos[:2] + f * size, base + r * size * 0.5, base - r * size * 0.5]


def draw(ax, state, info, t):
    from matplotlib.patches import Circle, Polygon, Rectangle

    ax.clear()
    ax.set_xlim(-FIELD_X - 300, FIELD_X + 300)
    ax.set_ylim(-FIELD_Y - 1000, FIELD_Y + 1000)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((-FIELD_X, -FIELD_Y), 2 * FIELD_X, 2 * FIELD_Y, fill=False, color="0.6"))
    ax.axhline(0, color="0.85", lw=1)
    for team, y in ((0, -FIELD_Y), (1, FIELD_Y)):
        ax.add_patch(Rectangle((-GOAL_HALF_W, y if team else y - 880), 2 * GOAL_HALF_W, 880, alpha=0.25, color=COLORS[team]))

    slices = info["ball_slices"]
    ball = state.ball.position
    ax.plot(slices[:, 0], slices[:, 1], ":", color="0.4", lw=1)
    # Marker size grows with height so roles can be checked against aerial balls.
    ax.add_patch(Circle(ball[:2], 92.75 * (1 + ball[2] / 700), color="0.2"))
    ax.text(ball[0] + 150, ball[1] - 300, f"z {ball[2]:.0f}", fontsize=7, color="0.2")

    coordination = info.get("coordination", {})
    for agent, car in state.cars.items():
        color = COLORS[car.team_num]
        if car.is_demoed:
            continue
        pos, ttb = car.physics.position, info["ttb"][agent]
        ax.add_patch(Polygon(car_triangle(pos, car.physics.forward, size=170 * (1 + pos[2] / 700)), color=color))
        first = info["roles"][agent] == FIRST
        locked = first and info.get("role_locked", {}).get(car.team_num, False)
        if first:
            ax.add_patch(Circle(pos[:2], 330, fill=False, color=color, lw=2.5, ls="-" if locked else "--"))
            k = min(round(ttb / 0.1), len(slices) - 1) if np.isfinite(ttb) else None
            if k is not None:
                ax.plot([pos[0], slices[k, 0]], [pos[1], slices[k, 1]], "--", color=color, lw=1)
        role = ("1st LOCK" if locked else "1st") if first else "2nd"
        ttb_text = f"{ttb:.2f}s" if np.isfinite(ttb) else "--"
        ax.text(pos[0] + 250, pos[1] + 250, f"{role} {ttb_text} z{pos[2]:.0f}", fontsize=7, color=color,
                weight="bold" if first else "normal")
        ev = coordination.get(car.team_num, {})
        if ev.get("double_commit") == agent:
            ax.plot(*pos[:2], "x", color="red", ms=14, mew=3)
            ax.text(pos[0] + 250, pos[1] - 150, "DOUBLE COMMIT", fontsize=7, color="red", weight="bold")
        elif ev.get("excused") and info["roles"][agent] != FIRST:
            ax.text(pos[0] + 250, pos[1] - 150, f"dc excused: {ev['excused']}", fontsize=7, color="0.45")
        if ev.get("nobody_challenging") == agent:
            ax.text(pos[0] + 250, pos[1] - 300, "NOBODY CHALLENGING", fontsize=7, color="red", weight="bold")

    for team, ev in coordination.items():
        if ev.get("nobody_home"):
            y = -FIELD_Y - 650 if team == 0 else FIELD_Y + 450
            ax.text(-1300, y, "NOBODY HOME", fontsize=9, color="red", weight="bold")
    ax.set_title(f"t = {t:6.1f}s{'   KICKOFF' if info['kickoff'] else ''}", fontsize=10)


class RoleStats:
    """Counts first-man changes and how long each holder kept the role."""

    def __init__(self):
        self.holder, self.since, self.holds, self.changes = {}, {}, [], 0

    def update(self, state, roles, t, reset):
        for team in (0, 1):
            first = next((a for a, c in state.cars.items() if c.team_num == team and roles[a] == FIRST), None)
            if reset or team not in self.holder:
                self.holder[team], self.since[team] = first, t
            elif first != self.holder[team]:
                self.holds.append(t - self.since[team])
                self.changes += 1
                self.holder[team], self.since[team] = first, t

    def summary(self, seconds):
        h = np.array(self.holds) if self.holds else np.array([np.nan])
        return (f"{self.changes} first-man changes in {seconds / 60:.1f} sim-minutes "
                f"({self.changes / max(seconds / 60, 1e-9) / 2:.1f} per team per minute); "
                f"median hold {np.nanmedian(h):.2f}s; holds under MIN_HOLD ({MIN_HOLD}s): "
                f"{np.mean(h < MIN_HOLD) * 100:.0f}% (only touch, demo or moving-away releases may do this)")


def load_policy(checkpoint, device):
    """The actor from a training checkpoint folder (the one holding ppo_learner/). Network width is read from
    the saved weights; the checkpoint must have been trained on the current observation versions."""
    import torch

    from playstyle import versions
    from playstyle.network import EntityActor

    learner = Path(checkpoint) / "ppo_learner"
    versions.check(learner)
    state = torch.load(learner / "actor.pt", map_location=device)
    d_model, d_ff = state["net.trunk.kind"].shape[1], state["net.trunk.blocks.0.ff.0.weight"].shape[0]
    actor = EntityActor(device=device, graphs=False, d_model=d_model, d_ff=d_ff)
    actor.load_state_dict(state)
    return actor.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=("scripted", "random"), default="scripted")
    ap.add_argument("--checkpoint", help="play all four cars with a trained policy from this checkpoint folder")
    ap.add_argument("--deterministic", action="store_true", help="with --checkpoint: always the likeliest action")
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--save", help="write the video to this file (.gif, or .mp4 through ffmpeg) instead of a window")
    ap.add_argument("--no-draw", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blue-mode", choices=MODES, default="normal")
    ap.add_argument("--orange-mode", choices=MODES, default="normal")
    args = ap.parse_args()

    actor = None
    if args.checkpoint:
        import torch

        actor = load_policy(args.checkpoint, "cuda" if torch.cuda.is_available() else "cpu")
    # Trained policies play the env they trained in. Roles are switched on for the overlay; they only add
    # shared-info entries for the reward and overlay, which the observation never reads.
    env = build_env(coordination=True) if actor is not None else make_env()
    bot = RoleBot(seed=args.seed, team_modes={0: args.blue_mode, 1: args.orange_mode})
    rng = np.random.default_rng(args.seed)
    obs = env.reset()
    stats, frames, touches, goals = RoleStats(), [], 0, 0
    mp4 = args.save and args.save.endswith(".mp4")

    fig = ax = writer = None
    if not args.no_draw:
        import matplotlib
        if args.save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 6.5))
        if mp4:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=round(1 / DT))
            writer.setup(fig, args.save, dpi=120)
        elif not args.save:
            plt.ion()
            plt.show()

    t, reset = 0.0, True
    while t < args.seconds:
        stats.update(env.state, env.shared_info["roles"], t, reset)
        if ax is not None:
            draw(ax, env.state, env.shared_info, t)
            if writer is not None:
                writer.grab_frame()
            elif args.save:
                fig.canvas.draw()
                frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            else:
                start = time.perf_counter()
                plt.pause(max(1e-3, DT - (time.perf_counter() - start)))
                if not plt.fignum_exists(fig.number):
                    break
        if actor is not None:
            import torch

            with torch.no_grad():
                chosen, _ = actor.get_action(None, [obs[a] for a in env.agents], deterministic=args.deterministic)
            actions = {a: np.asarray(c).reshape(1) for a, c in zip(env.agents, chosen)}
        elif args.policy == "scripted":
            actions = bot.act(env.state, env.shared_info)
        else:
            actions = {a: np.array([rng.integers(90)]) for a in env.agents}
        obs, _, term, trunc = env.step(actions)
        touches += sum(c.ball_touches > 0 for c in env.state.cars.values())
        goals += bool(env.state.goal_scored)
        t += DT
        reset = any(term.values()) or any(trunc.values())
        if reset:
            obs = env.reset()

    print(stats.summary(t))
    print(f"{touches / len(env.agents) / (t / 60):.2f} touches per car per minute, {goals} goals in {t / 60:.1f} sim-minutes")
    if writer is not None:
        writer.finish()
        print(f"saved {args.save}")
        return
    if args.save:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames[::2]]
        imgs[0].save(args.save, save_all=True, append_images=imgs[1:], duration=int(2 * DT * 1000), loop=0)
        print(f"saved {len(imgs)} frames to {args.save}")


if __name__ == "__main__":
    main()
