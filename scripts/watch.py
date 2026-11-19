"""Game viewer. Default: a 3D perspective view of the arena with simple shapes (cars as oriented boxes, the ball
with its predicted path and ground shadow, role and TTB labels). --view 2d: top-down with the full TTB, role
and coordination overlay.

    python scripts/watch.py                                  # live window, scripted bots
    python scripts/watch.py --view 2d                        # top-down overlay
    python scripts/watch.py --policy random
    python scripts/watch.py --save runs/roles.gif --seconds 20
    python scripts/watch.py --no-draw --seconds 600           # role-change stats only
    python scripts/watch.py --blue-mode chaser                # watch a deliberately bad team
    python scripts/watch.py --checkpoint runs/train/<run>/checkpoints/ppo/<run>-<id>/<step id> --save game.mp4 --seconds 180

2D: labels show role (LOCK when the first man is committed), TTB and height. Marker size grows with
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
from playstyle.match import new_match
from playstyle.roles import FIRST, MIN_HOLD
from playstyle.scripted import MODES, RoleBot

FIELD_X, FIELD_Y, GOAL_HALF_W = 4096, 5120, 893
FULL_GAME_CAP = 900.0  # s of play: a full match is 5 minutes plus kickoffs and overtime
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


class View3D:
    """Perspective view of the whole arena with simple shapes: wireframe field, translucent goals, cars as
    oriented boxes (darker nose face), the ball as a low-poly sphere with its predicted path and a ground
    shadow. Cars and ball are drawn 2.2x real size so they stay visible at full-field distance."""

    SCALE = 2.2
    CAR_HALF = np.array([118.0, 84.0, 36.0]) / 2  # Octane hitbox: length, width, height
    BALL_RADIUS = 92.75
    CEILING, GOAL_DEPTH, GOAL_HEIGHT = 2044, 880, 642
    BOX_FACES = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))

    def __init__(self, fig, elev=28, azim=-62):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self.Poly = Poly3DCollection
        ax = self.ax = fig.add_subplot(projection="3d")
        fig.subplots_adjust(0, 0, 1, 1)
        ax.set_axis_off()
        ax.set_xlim(-FIELD_X, FIELD_X)
        ax.set_ylim(-FIELD_Y - self.GOAL_DEPTH, FIELD_Y + self.GOAL_DEPTH)
        ax.set_zlim(0, self.CEILING)
        ax.set_box_aspect((2 * FIELD_X, 2 * (FIELD_Y + self.GOAL_DEPTH), self.CEILING), zoom=1.08)
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type("persp", focal_length=0.5)
        # Draw in a fixed order (floor, goals, then cars and ball, labels last). matplotlib's own depth sort
        # uses each shape's average depth, which puts the whole floor in front of anything near midfield.
        ax.computed_zorder = False

        x, y, z = FIELD_X, FIELD_Y, self.CEILING
        ax.add_collection3d(Poly3DCollection([[(-x, -y, 0), (x, -y, 0), (x, y, 0), (-x, y, 0)]],
                                             facecolor="#e4ece0", edgecolor="0.55", lw=1, zorder=0))
        ax.plot([-x, x], [0, 0], [0, 0], color="0.7", lw=0.8, zorder=1)
        ax.plot([-x, x, x, -x, -x], [-y, -y, y, y, -y], [z] * 5, color="0.82", lw=0.6, zorder=1)
        for cx, cy in ((-x, -y), (x, -y), (x, y), (-x, y)):
            ax.plot([cx, cx], [cy, cy], [0, z], color="0.82", lw=0.6, zorder=1)
        for team, sign in ((0, -1), (1, 1)):
            box = self._box((0, sign * (y + self.GOAL_DEPTH / 2), self.GOAL_HEIGHT / 2), np.eye(3),
                            (self.GOAL_DEPTH / 2, GOAL_HALF_W, self.GOAL_HEIGHT / 2), swap_xy=True)
            ax.add_collection3d(Poly3DCollection(box, facecolor=COLORS[team], edgecolor=COLORS[team],
                                                 alpha=0.18, lw=0.8, zorder=1))

        lat, lon = np.linspace(0, np.pi, 7), np.linspace(0, 2 * np.pi, 11)
        point = lambda a, b: (np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a))
        self.sphere = np.array([[point(lat[i], lon[j]), point(lat[i + 1], lon[j]), point(lat[i + 1], lon[j + 1]),
                                 point(lat[i], lon[j + 1])] for i in range(6) for j in range(10)])
        self.ball = Poly3DCollection([], facecolor="0.3", edgecolor="0.15", lw=0.3, zorder=3)
        ax.add_collection3d(self.ball)
        self.ball_label = ax.text(0, 0, 0, "", fontsize=7, color="0.2", zorder=4)
        self.path, = ax.plot([], [], [], ":", color="0.35", lw=1, zorder=2)
        self.shadow, = ax.plot([], [], [], "o", color="0.55", ms=4, zorder=2)
        self.drop, = ax.plot([], [], [], "-", color="0.6", lw=0.7, zorder=2)
        self.cars, self.labels = {}, {}
        self.title = ax.text2D(0.02, 0.97, "", transform=ax.transAxes, fontsize=10)

    @staticmethod
    def _box(center, rot, half, swap_xy=False):
        """The six faces of a box: half-sizes along rot's columns (forward, right, up)."""
        h = np.array([half[1], half[0], half[2]]) if swap_xy else np.asarray(half)
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
        world = np.asarray(center) + corners @ np.asarray(rot).T
        return [world[list(f)] for f in View3D.BOX_FACES]

    def update(self, state, info, t):
        ax = self.ax
        ball = np.asarray(state.ball.position, dtype=float)
        self.ball.set_verts(list(ball + self.sphere * self.BALL_RADIUS * self.SCALE))
        self.ball_label.set_position_3d(ball + (0, 0, 260))
        self.ball_label.set_text(f"z {ball[2]:.0f}")
        slices = np.asarray(info["ball_slices"])[:, :3]
        self.path.set_data_3d(slices[:, 0], slices[:, 1], slices[:, 2])
        self.shadow.set_data_3d([ball[0]], [ball[1]], [0])
        self.drop.set_data_3d([ball[0], ball[0]], [ball[1], ball[1]], [0, ball[2]])

        roles = info.get("roles", {})
        for agent, car in state.cars.items():
            if agent not in self.cars:
                nose = tuple(0.55 * c for c in matplotlib_rgb(COLORS[car.team_num]))
                colors = [COLORS[car.team_num]] * 6
                colors[3] = nose  # +forward face
                self.cars[agent] = self.Poly([], facecolors=colors, edgecolor="0.1", lw=0.4, zorder=3)
                ax.add_collection3d(self.cars[agent])
                self.labels[agent] = ax.text(0, 0, 0, "", fontsize=7, color=COLORS[car.team_num], zorder=4)
            if car.is_demoed:
                self.cars[agent].set_verts([])
                self.labels[agent].set_text("")
                continue
            phys = car.physics
            rot = np.column_stack([phys.forward, phys.right, phys.up])
            self.cars[agent].set_verts(self._box(phys.position, rot, self.CAR_HALF * self.SCALE))
            ttb = info["ttb"][agent]
            role = ("1st" if roles.get(agent) == FIRST else "2nd") if roles else ""
            self.labels[agent].set_position_3d(np.asarray(phys.position) + (0, 0, 220))
            self.labels[agent].set_text(f"{role} {ttb:.2f}s" if np.isfinite(ttb) else role)

        match = info.get("match", {})
        clock = match.get("time_remaining")
        clock_text = f"{int(clock // 60)}:{int(clock % 60):02d}{' OT' if match.get('overtime') else ''}   " \
            if clock is not None else ""
        self.title.set_text(f"{clock_text}score {match.get('score_diff', 0):+d} (blue)   t = {t:5.1f}s"
                            f"{'   KICKOFF' if info['kickoff'] else ''}")


def matplotlib_rgb(name):
    from matplotlib.colors import to_rgb

    return to_rgb(name)


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
    """The actor from a training checkpoint folder (the one holding ppo_learner/), or from a run folder holding
    numbered checkpoint folders, in which case the newest is used. Network width is read from the saved weights;
    the checkpoint must have been trained on the current observation versions."""
    import torch

    from playstyle import versions
    from playstyle.network import EntityActor

    checkpoint = Path(checkpoint)
    if not (checkpoint / "ppo_learner").exists():
        numbered = [p for p in checkpoint.iterdir() if p.name.isdigit() and (p / "ppo_learner").exists()]
        if not numbered:
            raise SystemExit(f"{checkpoint} holds neither ppo_learner/ nor numbered checkpoint folders")
        checkpoint = max(numbered, key=lambda p: int(p.name))
        print(f"using newest checkpoint {checkpoint.name}")
    learner = checkpoint / "ppo_learner"
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
    ap.add_argument("--full-game", action="store_true",
                    help="one whole match from kickoff: 5:00 on the clock at 0-0, until it ends (overtime included); "
                         "--seconds is ignored, apart from a 15-minute safety cap")
    ap.add_argument("--save", help="write the video to this file (.gif, or .mp4 through ffmpeg) instead of a window")
    ap.add_argument("--no-draw", action="store_true")
    ap.add_argument("--view", choices=("3d", "2d"), default="3d",
                    help="3d: perspective arena view; 2d: top-down with the full coordination overlay")
    ap.add_argument("--drills", help="with --checkpoint: play only these practice starts, comma-separated "
                                     "(defend_shot, aerial), back to back, each titled with its kind")
    ap.add_argument("--drill-level", type=float, default=0.0, help="curriculum level for --drills (0 easy .. 1 hard)")
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
    if args.drills:
        kinds = args.drills.split(",")
        env = build_env(phase="nx", start_weights={k: 1 / len(kinds) for k in kinds})
        starts = env.state_mutator.mutators[1]
        starts.curriculum.level = starts.aerial_curriculum.level = args.drill_level
    else:
        env = build_env(coordination=True) if actor is not None else make_env()
    bot = RoleBot(seed=args.seed, team_modes={0: args.blue_mode, 1: args.orange_mode})
    rng = np.random.default_rng(args.seed)
    obs = env.reset()
    if args.full_game:  # episodes normally start from a sampled score and clock
        env.shared_info["match"] = new_match()
        obs = env.obs_builder.build_obs(env.agents, env.state, env.shared_info)
        args.seconds = FULL_GAME_CAP
    stats, frames, touches, goals = RoleStats(), [], 0, 0
    mp4 = args.save and args.save.endswith(".mp4")

    fig = ax = writer = None
    if not args.no_draw:
        import matplotlib
        if args.save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if args.view == "3d":
            fig = plt.figure(figsize=(9, 6))
            view = View3D(fig)
            ax = view.ax
        else:
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
            if args.drills:
                fig.suptitle(f"{env.shared_info['start']} drill, level {args.drill_level:g}", x=0.98, y=0.97,
                             ha="right", fontsize=11)
            if args.view == "3d":
                view.update(env.state, env.shared_info, t)
            else:
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
        if args.full_game:
            if any(term.values()):  # the match is over; truncations (no-touch timeout) are training-only
                break
            reset = False
            continue
        reset = any(term.values()) or any(trunc.values())
        if reset:
            obs = env.reset()

    print(stats.summary(t))
    print(f"{touches / len(env.agents) / (t / 60):.2f} touches per car per minute, {goals} goals in {t / 60:.1f} sim-minutes")
    if args.full_game:
        m = env.shared_info["match"]
        print(f"final score blue {m['blue']} - {m['orange']} orange{' (overtime)' if m['overtime'] else ''}"
              f"{'' if m['over'] else ' (stopped at the safety cap)'}")
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
