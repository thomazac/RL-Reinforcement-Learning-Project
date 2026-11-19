"""The exported RLBot bot builds the training observation: a simulated match is replayed to the bot as RLBot packets
(converted from the RocketSim state), and the bot's observation for its car must match the env's."""

import importlib.util
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("rlbot")
from rlbot import flat
from rlgym_compat.common_values import BOOST_LOCATIONS
from rlgym_compat.math import quat_to_euler, rotation_to_quaternion

from playstyle import versions
from playstyle.env import build_env
from playstyle.network import EntityActor

AGENTS = ("blue-0", "blue-1", "orange-0", "orange-1")
PLAYER_IDS = {a: 100 + i for i, a in enumerate(AGENTS)}


def vec(v):
    return flat.Vector3(*map(float, v))


def physics(p):
    pitch, yaw, roll = quat_to_euler(rotation_to_quaternion(np.asarray(p.rotation_mtx, dtype=np.float64)))
    return flat.Physics(vec(p.position), flat.Rotator(float(pitch), float(yaw), float(roll)),
                        vec(p.linear_velocity), vec(p.angular_velocity))


def packet(state, match):
    tick = int(state.tick_count)
    players = []
    for a in AGENTS:
        c = state.cars[a]
        air = flat.AirState.OnGround if c.on_ground else flat.AirState.InAir
        players.append(flat.PlayerInfo(
            physics=physics(c.physics), team=c.team_num, player_id=PLAYER_IDS[a], boost=float(c.boost_amount),
            air_state=air, has_jumped=bool(c.has_jumped), has_double_jumped=bool(c.has_double_jumped),
            has_dodged=bool(c.has_flipped), dodge_elapsed=float(c.flip_time),
            dodge_timeout=-1.0 if c.on_ground or not c.has_jumped else float(1.25 - c.air_time_since_jump),
            demolished_timeout=float(c.demo_respawn_timer) if c.is_demoed else -1.0,
            latest_touch=flat.Touch(game_seconds=tick / 120) if c.ball_touches > 0 else None,
            hitbox=flat.BoxShape(118.00738, 84.19941, 36.159073), is_bot=True, name=a))
    return flat.GamePacket(
        players=players, balls=[flat.BallInfo(physics=physics(state.ball))],
        boost_pads=[flat.BoostPadState(t == 0, float(t)) for t in state.boost_pad_timers],
        match_info=flat.MatchInfo(frame_num=tick, seconds_elapsed=tick / 120, match_phase=flat.MatchPhase.Active,
                                  game_time_remaining=float(match["time_remaining"]), is_overtime=match["overtime"]),
        teams=[flat.TeamInfo(team_index=0, score=match["blue"]), flat.TeamInfo(team_index=1, score=match["orange"])])


@pytest.fixture(scope="module")
def bot_module(tmp_path_factory):
    """A bot exported from a fresh (untrained) actor, imported from the export folder like RLBot would run it."""
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    from export_rlbot import export

    ckpt = tmp_path_factory.mktemp("ckpt") / "ppo_learner"
    ckpt.mkdir()
    torch.save(EntityActor(device="cpu", graphs=False).state_dict(), ckpt / "actor.pt")
    versions.write(ckpt, "v1")
    out = export(ckpt.parent, tmp_path_factory.mktemp("export") / "bot", "test bot")
    assert (out / "bot.toml").read_text().startswith("[settings]")
    spec = importlib.util.spec_from_file_location("exported_bot", out / "bot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("me", ["blue-0", "orange-1"])
def test_bot_observation_matches_training(bot_module, me):
    env = build_env(phase="1a")  # kickoff starts: no practice episode ends the replay early
    obs = env.reset()
    bot = bot_module.PlaystyleBot.__new__(bot_module.PlaystyleBot)
    bot.field_info = flat.FieldInfo(boost_pads=[flat.BoostPad(vec(loc), loc[2] > 72) for loc in BOOST_LOCATIONS])
    bot.match_config = flat.MatchConfiguration()
    bot.player_id = PLAYER_IDS[me]
    bot.initialize()
    seen = {}
    build = bot.obs_builder.build_obs
    bot.obs_builder.build_obs = lambda agents, state, info: seen.setdefault("obs", build(agents, state, info))

    rng = np.random.default_rng(0)
    for step in range(60):
        seen.clear()
        controls = bot.get_output(packet(env.state, env.shared_info["match"]))
        np.testing.assert_allclose(seen["obs"][me], obs[me], atol=2e-3, err_msg=f"step {step}")
        action = int(np.flatnonzero((bot.table == bot.prev_action).all(axis=1))[0])
        assert isinstance(controls, flat.ControllerState)
        actions = {a: np.array([action if a == me else rng.integers(len(bot.table))]) for a in env.agents}
        obs, _, term, trunc = env.step(actions)
        assert not any(term.values()) and not any(trunc.values())
