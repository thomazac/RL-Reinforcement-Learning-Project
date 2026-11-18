"""Episode start states for Phase 1b on: kickoffs, human replay moments, random realistic states and coordination
tests, mixed by MixedStart. Every non-kickoff state passes valid_state before use.

Replay states come from data/replay_states.npz (scripts/extract_replay_states.py). Replays carry positions,
rotations, velocities and boost but no jump or flip state, so cars that are not resting on a surface start with
their jump, double jump and flip used up (the conservative choice: no free flip).
"""

import numpy as np
from rlgym.api import DoneCondition, StateMutator

from .ball_prediction import TICK_RATE
from .rewards import SHOT_HORIZON, enters_net

FIELD_X, FIELD_Y, CEILING = 4096.0, 5120.0, 2044.0
CORNER = 8064.0  # the 45-degree corner walls: |x| + |y| = CORNER
# Floor and ceiling curve into the walls. Measured: a flat car on the floor is pushed on its first step when its
# center is 80-240 uu from a side wall, matching a radius of 256 (its hitbox corners enter the curve below ~205).
EDGE_RADIUS = 256.0
GOAL_HALF_W, GOAL_H, GOAL_DEPTH = 893.0, 642.0, 880.0
BALL_RADIUS = 92.75
CAR_REST_Z = 17.0
ON_SURFACE = 40.0  # car origin this close to a surface, roof facing away from it: resting on it
# RocketSim's Octane hitbox: half extents along forward, right, up, and the box center's offset from the car origin.
HITBOX_HALF = np.array([120.507, 86.6994, 38.6591]) / 2
HITBOX_OFFSET = np.array([13.8757, 0.0, 20.755])
CAR_GAP = 2 * float(np.linalg.norm(HITBOX_HALF)) + 5  # hitbox centers at least this far apart (bounding spheres)
CONTACT_GAP = 5.0  # uu of clearance between the ball and any hitbox
MAX_CAR_SPIN = 5.5  # rad/s

# Shares of episode starts. defend_shot episodes are short practice (PracticeEnd, ~30 steps; at most 90), whole-match
# episodes run ~2,700 steps (clocks start anywhere up to 5 minutes, plus overtime). Measured at 35% of episodes,
# practice was only ~0.6% of steps (0.8 practice shots per 5 minutes of play); 80% of episodes gives ~5-6% of steps,
# with whole matches still ~95% of training. The whole-match starts keep their relative shares.
START_WEIGHTS = {"kickoff": 0.055, "replay": 0.077, "random": 0.034, "coord_equidistant": 0.017,
                 "coord_counter": 0.017, "defend_shot": 0.80}
PRACTICE_SECONDS = 6.0
GRAVITY = 650.0  # uu/s^2


def surfaces(pos):
    """(distance, outward normal) to the floor, ceiling, side walls and back walls."""
    x, y, z = pos
    return [(z, (0, 0, 1)), (CEILING - z, (0, 0, -1)), (FIELD_X - x, (-1, 0, 0)), (FIELD_X + x, (1, 0, 0)),
            (FIELD_Y - y, (0, -1, 0)), (FIELD_Y + y, (0, 1, 0))]


def resting(pos, up):
    """Whether a car at pos with roof direction up is resting on one of the flat surfaces."""
    return any(d < ON_SURFACE and float(np.dot(up, n)) > 0.8 for d, n in surfaces(pos))


def hitbox_corners(pos, rot):
    """The 8 corners of a car's hitbox; rot has columns forward, right, up."""
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    return np.asarray(pos) + (HITBOX_OFFSET + signs * HITBOX_HALF) @ np.asarray(rot).T


def inside_arena(points, margin=0.0):
    """All points at least margin inside the arena: the flat walls, floor, ceiling and corner walls, the rounded
    edges where floor and ceiling meet the walls (radius EDGE_RADIUS; none along the goal mouths), or inside a goal."""
    x, y, z = np.abs(points[:, 0]), np.abs(points[:, 1]), points[:, 2]
    in_field = (x <= FIELD_X - margin) & (y <= FIELD_Y - margin) & (x + y <= CORNER - margin * np.sqrt(2))
    in_goal = (x <= GOAL_HALF_W - margin) & (y > FIELD_Y - margin) & (y <= FIELD_Y + GOAL_DEPTH - margin) \
        & (z >= margin) & (z <= GOAL_H - margin)
    ok = (in_field | in_goal) & (z >= margin) & (z <= CEILING - margin)
    r = EDGE_RADIUS
    walls = [FIELD_X - x, np.where(x <= GOAL_HALF_W, np.inf, FIELD_Y - y), (CORNER - x - y) / np.sqrt(2)]
    for h in (z, CEILING - z):
        for w in walls:
            near = (h < r) & (w < r)
            ok &= ~near | ((r - h) ** 2 + (r - w) ** 2 <= (r - margin) ** 2)
    return bool(np.all(ok | in_goal))


def ball_box_distance(ball_pos, pos, rot):
    """Distance from the ball center to a car's hitbox (0 if inside)."""
    local = np.asarray(rot).T @ (np.asarray(ball_pos) - np.asarray(pos)) - HITBOX_OFFSET
    return float(np.linalg.norm(local - np.clip(local, -HITBOX_HALF, HITBOX_HALF)))


def valid_state(ball_pos, car_pos, car_rot):
    """Nothing inside walls or the corners, the ball in play (not in a goal), no car overlapping another or the ball.
    Hitboxes are RocketSim's Octane. car_pos: (n, 3); car_rot: (n, 3, 3)."""
    ball = np.asarray(ball_pos, dtype=np.float64)
    if abs(ball[1]) > FIELD_Y - BALL_RADIUS + 1 or not inside_arena(ball[None], BALL_RADIUS - 1):
        return False
    centers = []
    for pos, rot in zip(car_pos, car_rot):
        if not inside_arena(hitbox_corners(pos, rot)):
            return False
        if ball_box_distance(ball, pos, rot) < BALL_RADIUS + CONTACT_GAP:
            return False
        centers.append(np.asarray(pos) + np.asarray(rot) @ HITBOX_OFFSET)
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if np.linalg.norm(centers[i] - centers[j]) < CAR_GAP:
                return False
    return True


def yaw_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def mirror(ball, cars):
    """Rotate the whole state 180 degrees about the vertical axis and swap the teams' car slots, so blue and orange
    trade places. ball: dict, cars: list of dicts (blue cars first)."""
    flip = np.diag([-1.0, -1.0, 1.0])

    def turn(d):
        out = dict(d)
        for k in ("pos", "vel", "ang_vel"):
            out[k] = flip @ d[k]
        if "rot" in d:
            out["rot"] = flip @ d["rot"]
        return out

    half = len(cars) // 2
    turned = [turn(c) for c in cars]
    return turn(ball), turned[half:] + turned[:half]


def apply_state(state, ball, cars):
    """Write ball and cars (blue cars first, then orange, each a dict with pos, rot (3x3, columns forward, right,
    up), vel, ang_vel, boost, airborne) into an RLGym GameState."""
    state.ball.position = np.asarray(ball["pos"], dtype=np.float32)
    state.ball.linear_velocity = np.asarray(ball["vel"], dtype=np.float32)
    state.ball.angular_velocity = np.asarray(ball["ang_vel"], dtype=np.float32)
    order = sorted(state.cars, key=lambda a: (state.cars[a].team_num, a))
    for agent, c in zip(order, cars):
        car = state.cars[agent]
        car.physics.position = np.asarray(c["pos"], dtype=np.float32)
        car.physics.rotation_mtx = np.asarray(c["rot"], dtype=np.float32)
        car.physics.linear_velocity = np.asarray(c["vel"], dtype=np.float32)
        spin = np.asarray(c["ang_vel"], dtype=np.float64)
        car.physics.angular_velocity = (spin * min(1.0, MAX_CAR_SPIN / max(np.linalg.norm(spin), 1e-6))).astype(np.float32)
        car.boost_amount = float(c["boost"])
        airborne = bool(c["airborne"])
        car.on_ground = not airborne
        car.has_jumped = car.has_double_jumped = car.has_flipped = airborne
        car.is_jumping = False
        car.demo_respawn_timer = 0.0


class ReplayStates:
    """Moments from human replays. Games are drawn evenly, then a moment within the game, so long games and games
    with more usable moments do not dominate. Half the draws are mirrored (teams swapped)."""

    def __init__(self, path):
        data = np.load(path)
        self.ball, self.cars, self.airborne, self.game = (data[k] for k in ("ball", "cars", "airborne", "game"))
        self.by_game = [np.flatnonzero(self.game == g) for g in np.unique(self.game)]

    def sample(self, rng):
        frames = self.by_game[rng.integers(len(self.by_game))]
        i = frames[rng.integers(len(frames))]
        b, c = self.ball[i], self.cars[i]
        ball = {"pos": b[0:3], "vel": b[3:6], "ang_vel": b[6:9]}
        cars = [{"pos": c[k, 0:3], "rot": c[k, 3:12].reshape(3, 3), "vel": c[k, 12:15], "ang_vel": c[k, 15:18],
                 "boost": c[k, 18], "airborne": self.airborne[i, k]} for k in range(len(c))]
        if rng.random() < 0.5:
            ball, cars = mirror(ball, cars)
        return ball, cars


def ground_car(pos, yaw, speed, boost):
    rot = yaw_matrix(yaw)
    return {"pos": np.array([pos[0], pos[1], CAR_REST_Z]), "rot": rot, "vel": rot[:, 0] * speed,
            "ang_vel": np.zeros(3), "boost": boost, "airborne": False}


def random_state(rng):
    """Cars on the ground anywhere, random heading, speed and boost; the ball anywhere in play, moving."""
    ball = {"pos": np.array([rng.uniform(-3500, 3500), rng.uniform(-4500, 4500), rng.uniform(BALL_RADIUS, 1200)]),
            "vel": rng.uniform(-1200, 1200, 3), "ang_vel": np.zeros(3)}
    cars = [ground_car((rng.uniform(-3800, 3800), rng.uniform(-4800, 4800)), rng.uniform(-np.pi, np.pi),
                       rng.uniform(0, 1500), rng.uniform(0, 100)) for _ in range(4)]
    return ball, cars


def _facing(pos, target):
    d = np.asarray(target[:2]) - np.asarray(pos[:2])
    return float(np.arctan2(d[1], d[0]))


def coord_equidistant(rng):
    """Both cars of one team at the same distance from a loose ball, on their own side of it, driving at it: one
    should go, the other should not. Opponents start further away."""
    ball_xy = np.array([rng.uniform(-3000, 3000), rng.uniform(-2500, 2500)])
    ball = {"pos": np.array([*ball_xy, BALL_RADIUS]), "vel": np.array([*rng.uniform(-300, 300, 2), 0.0]),
            "ang_vel": np.zeros(3)}
    dist, speed = rng.uniform(1500, 2800), rng.uniform(500, 1200)
    # Team 0 (blue) defends -y: its cars sit at -y of the ball, spread by +-(20..60) degrees.
    spread = np.radians(rng.uniform(20, 60))
    cars = []
    for side in (-1, 1):
        angle = -np.pi / 2 + side * spread
        pos = ball_xy + dist * np.array([np.cos(angle), np.sin(angle)])
        cars.append(ground_car(pos, _facing(pos, ball_xy), speed, rng.uniform(20, 100)))
    for _ in range(2):
        angle = np.pi / 2 + rng.uniform(-1.0, 1.0)
        pos = ball_xy + rng.uniform(dist + 1000, dist + 2500) * np.array([np.cos(angle), np.sin(angle)])
        cars.append(ground_car(pos, _facing(pos, ball_xy), rng.uniform(0, 800), rng.uniform(0, 100)))
    ball_state, car_states = ball, cars
    return mirror(ball_state, car_states) if rng.random() < 0.5 else (ball_state, car_states)


def coord_counter(rng):
    """A counter-attack: both cars of one team caught upfield while an opponent carries the ball toward their net.
    Someone has to get back."""
    ball_xy = np.array([rng.uniform(-2500, 2500), rng.uniform(-1500, 1000)])
    ball_vel = np.array([rng.uniform(-300, 300), -rng.uniform(1000, 1700), 0.0])
    ball = {"pos": np.array([*ball_xy, BALL_RADIUS]), "vel": ball_vel, "ang_vel": np.zeros(3)}
    heading = _facing((0, 0), ball_vel)
    cars = []
    for _ in range(2):  # blue, caught in orange's half
        pos = (rng.uniform(-3500, 3500), rng.uniform(1500, 4300))
        cars.append(ground_car(pos, rng.uniform(-np.pi, np.pi), rng.uniform(0, 1200), rng.uniform(0, 100)))
    behind = ball_xy - rng.uniform(300, 600) * ball_vel[:2] / np.linalg.norm(ball_vel[:2])
    cars.append(ground_car(behind, heading, float(np.linalg.norm(ball_vel[:2])), rng.uniform(20, 100)))
    pos = (rng.uniform(-3000, 3000), rng.uniform(0, 3000))
    cars.append(ground_car(pos, _facing(pos, ball_xy), rng.uniform(500, 1500), rng.uniform(0, 100)))
    return mirror(ball, cars) if rng.random() < 0.5 else (ball, cars)


def _lerp(easy, hard, level):
    return easy + (hard - easy) * level


def defend_shot(rng, level=1.0):
    """A shot already on its way into one team's net: the ball on a ballistic path into the goal mouth. One defender
    near the net facing the ball, the other rotating back from upfield; the shooter just behind the ball, the other
    attacker upfield. Bots can only learn saves by facing shots.

    level (0..1, ShotCurriculum) sets the difficulty: at 0 the shot takes 1.8-2.5 s, crosses the line 110-200 uu up
    and is aimed within 150 uu of the keeper; at 1 it takes 0.8-2 s, crosses 150-550 up, anywhere in the goal mouth."""
    keeper_pos = (rng.uniform(-800, 800), rng.uniform(-5000, -4300))
    pos = np.array([rng.uniform(-2500, 2500), rng.uniform(-4200, -1500),
                    rng.uniform(BALL_RADIUS, _lerp(150, 400, level))])
    spread = _lerp(150, 700, level)
    aim_x = (1 - level) * keeper_pos[0] + rng.uniform(-spread, spread)
    target = np.array([np.clip(aim_x, -750, 750), -FIELD_Y - 200,  # blue (team 0) defends -y
                       rng.uniform(_lerp(110, 150, level), _lerp(200, 550, level))])
    flight = rng.uniform(_lerp(1.8, 0.8, level), _lerp(2.5, 2.0, level))
    vel = (target - pos) / flight
    vel[2] += 0.5 * GRAVITY * flight
    ball = {"pos": pos, "vel": vel, "ang_vel": np.zeros(3)}
    keeper = ground_car(keeper_pos, _facing(keeper_pos, pos), rng.uniform(0, 500), rng.uniform(0, 100))
    back_pos = (rng.uniform(-3500, 3500), pos[1] + rng.uniform(-500, 2500))
    back = ground_car(back_pos, -np.pi / 2 + rng.uniform(-0.6, 0.6), rng.uniform(500, 1500), rng.uniform(0, 100))
    heading = vel[:2] / np.linalg.norm(vel[:2])
    # On easy (slow) shots a shooter following through at full speed catches his own shot (23% of level-0 shots),
    # so he starts slower and further back there.
    shooter_pos = pos[:2] - rng.uniform(_lerp(700, 250, level), _lerp(1000, 450, level)) * heading
    shooter = ground_car(shooter_pos, float(np.arctan2(heading[1], heading[0])),
                         rng.uniform(_lerp(300, 1000, level), _lerp(800, 1800, level)), rng.uniform(0, 100))
    support_pos = (rng.uniform(-3000, 3000), pos[1] + rng.uniform(1000, 3500))
    support = ground_car(support_pos, -np.pi / 2 + rng.uniform(-0.8, 0.8), rng.uniform(500, 1500), rng.uniform(0, 100))
    cars = [keeper, back, shooter, support]
    return mirror(ball, cars) if rng.random() < 0.5 else (ball, cars)


GENERATORS = {"random": random_state, "coord_equidistant": coord_equidistant, "coord_counter": coord_counter,
              "defend_shot": defend_shot}


def state_is_valid(ball, cars):
    return valid_state(np.asarray(ball["pos"], dtype=np.float64), np.array([c["pos"] for c in cars], dtype=np.float64),
                       np.array([c["rot"] for c in cars], dtype=np.float64))


class ShotCurriculum:
    """Adaptive difficulty for defend_shot, one per env process (no coordination between processes needed): starts
    easy and steps up by STEP each time more than PASS_RATE of the last WINDOW practice shots were defended. A save
    rate near zero gives PPO almost no successful examples; this keeps it in the range where it can learn."""

    STEP, PASS_RATE, WINDOW = 0.1, 0.30, 40

    def __init__(self, level=0.0):
        self.level = level
        self.results = []

    def record(self, defended):
        self.results.append(bool(defended))
        if len(self.results) >= self.WINDOW:
            if np.mean(self.results[-self.WINDOW:]) > self.PASS_RATE and self.level < 1.0:
                self.level = min(1.0, round(self.level + self.STEP, 6))
                self.results.clear()
            else:
                self.results = self.results[-self.WINDOW:]


class PracticeEnd(DoneCondition):
    """Truncation for defend_shot episodes, which are save practice rather than matches: they end once the play
    resolves (a goal, or a defender touched the ball and it is no longer on course for the net) or after
    PRACTICE_SECONDS. Truncated, not terminated, so the value bootstraps and nothing learns that a game ends there.
    Other starts are whole matches and never end here. Outcomes go to the curriculum: defended counts as a success,
    a goal as a failure; a shot that times out without either is not counted."""

    def __init__(self, curriculum=None):
        self.curriculum = curriculum

    def reset(self, agents, initial_state, shared_info):
        self.active = shared_info.get("start") == "defend_shot"
        self.start_tick = initial_state.tick_count
        self.defending = 0 if initial_state.ball.linear_velocity[1] < 0 else 1

    def is_done(self, agents, state, shared_info):
        done = False
        if self.active:
            touched = any(c.ball_touches > 0 for c in state.cars.values() if c.team_num == self.defending)
            defended = touched and not enters_net(np.asarray(shared_info["ball_slices"]), self.defending, SHOT_HORIZON)
            conceded = bool(state.goal_scored) and state.scoring_team != self.defending
            done = defended or bool(state.goal_scored) \
                or (state.tick_count - self.start_tick) / TICK_RATE >= PRACTICE_SECONDS
            if done:
                if self.curriculum is not None and (defended or conceded):
                    self.curriculum.record(defended)
                self.active = False
        return {a: done for a in agents}


class MixedStart(StateMutator):
    """Picks each episode's start: kickoff (through the given kickoff mutator), a replay moment, a random
    realistic state or a coordination test, by START_WEIGHTS. Generated states are redrawn until valid (replay
    states were validated at extraction). The choice goes to shared_info["start"]."""

    def __init__(self, kickoff, replay_path=None, seed=None, weights=None, curriculum=None):
        self.kickoff = kickoff
        self.curriculum = curriculum or ShotCurriculum(level=1.0)
        self.rng = np.random.default_rng(seed)
        self.replays = ReplayStates(replay_path) if replay_path else None
        w = dict(START_WEIGHTS if weights is None else weights)
        if self.replays is None:
            w.pop("replay", None)
        self.kinds, p = list(w), np.array(list(w.values()), dtype=np.float64)
        self.p = p / p.sum()

    def apply(self, state, shared_info):
        kind = self.kinds[self.rng.choice(len(self.kinds), p=self.p)]
        shared_info["start"] = kind
        if kind == "kickoff":
            self.kickoff.apply(state, shared_info)
            return
        if kind == "defend_shot":
            shared_info["practice_level"] = self.curriculum.level
        for _ in range(100):
            if kind == "replay":
                ball, cars = self.replays.sample(self.rng)
            elif kind == "defend_shot":
                ball, cars = defend_shot(self.rng, self.curriculum.level)
            else:
                ball, cars = GENERATORS[kind](self.rng)
            if state_is_valid(ball, cars):
                apply_state(state, ball, cars)
                return
        shared_info["start"] = "kickoff"  # no valid draw: fall back rather than start inside a wall
        self.kickoff.apply(state, shared_info)
