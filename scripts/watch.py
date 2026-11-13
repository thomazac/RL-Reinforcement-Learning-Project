"""Top-down viewer with the TTB and role overlay.

    python scripts/watch.py                                  # live window, scripted bots
    python scripts/watch.py --policy random
    python scripts/watch.py --save runs/roles.gif --seconds 20
    python scripts/watch.py --no-draw --seconds 600           # role-change stats only
"""

import argparse
import time

import numpy as np

from playstyle.ball_prediction import TICK_RATE
from playstyle.env import TICK_SKIP, make_env
from playstyle.roles import FIRST, MIN_HOLD
from playstyle.scripted import RoleBot

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
    ax.plot(slices[:, 0], slices[:, 1], ":", color="0.4", lw=1)
    ax.add_patch(Circle(state.ball.position[:2], 92.75, color="0.2"))

    for agent, car in state.cars.items():
        color = COLORS[car.team_num]
        if car.is_demoed:
            continue
        pos, ttb = car.physics.position, info["ttb"][agent]
        ax.add_patch(Polygon(car_triangle(pos, car.physics.forward), color=color))
        first = info["roles"][agent] == FIRST
        if first:
            ax.add_patch(Circle(pos[:2], 330, fill=False, color=color, lw=2.5))
            k = min(round(ttb / 0.1), len(slices) - 1) if np.isfinite(ttb) else None
            if k is not None:
                ax.plot([pos[0], slices[k, 0]], [pos[1], slices[k, 1]], "--", color=color, lw=1)
        label = f"{'1st' if first else '2nd'} {ttb:.2f}s" if np.isfinite(ttb) else f"{'1st' if first else '2nd'} --"
        ax.text(pos[0] + 250, pos[1] + 250, label, fontsize=8, color=color, weight="bold" if first else "normal")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=("scripted", "random"), default="scripted")
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--save", help="write a GIF instead of opening a window")
    ap.add_argument("--no-draw", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = make_env()
    bot, rng = RoleBot(seed=args.seed), np.random.default_rng(args.seed)
    env.reset()
    stats, frames = RoleStats(), []

    fig = ax = None
    if not args.no_draw:
        import matplotlib
        if args.save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 6.5))
        if not args.save:
            plt.ion()
            plt.show()

    t, reset = 0.0, True
    while t < args.seconds:
        stats.update(env.state, env.shared_info["roles"], t, reset)
        if ax is not None:
            draw(ax, env.state, env.shared_info, t)
            if args.save:
                fig.canvas.draw()
                frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            else:
                start = time.perf_counter()
                plt.pause(max(1e-3, DT - (time.perf_counter() - start)))
                if not plt.fignum_exists(fig.number):
                    break
        if args.policy == "scripted":
            actions = bot.act(env.state, env.shared_info)
        else:
            actions = {a: np.array([rng.integers(90)]) for a in env.agents}
        _, _, term, trunc = env.step(actions)
        t += DT
        reset = any(term.values()) or any(trunc.values())
        if reset:
            env.reset()

    print(stats.summary(t))
    if args.save:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames[::2]]
        imgs[0].save(args.save, save_all=True, append_images=imgs[1:], duration=int(2 * DT * 1000), loop=0)
        print(f"saved {len(imgs)} frames to {args.save}")


if __name__ == "__main__":
    main()
