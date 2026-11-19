"""RLBot v5 bot for a checkpoint exported with scripts/export_rlbot.py.

The export folder holds this file, bot.toml, the actor weights (model/) and a snapshot of the playstyle package, so
the bot builds exactly the observation its network was trained on. Each packet updates an rlgym-compat game state;
every TICK_SKIP ticks the bot runs the training pipeline (TTBRoleProvider, then EntityObs) for its own car, takes the
network's most likely action from the training lookup table and holds it until the next decision.

Cars are named as in training (blue-0, blue-1, orange-0, orange-1, in packet order within each team), which fixes the
order of the teammate and opponent blocks in the observation. Score, clock and overtime come from the packet.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch
from rlbot import flat
from rlbot.managers import Bot
from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym_compat import GameState

from playstyle import versions
from playstyle.env import TICK_SKIP
from playstyle.network import EntityActor
from playstyle.obs import EntityObs, MATCH_SECONDS
from playstyle.shared_info import TTBRoleProvider


def load_actor(model_dir):
    versions.check(model_dir)
    state = torch.load(model_dir / "actor.pt", map_location="cpu")
    d_model, d_ff = state["net.trunk.kind"].shape[1], state["net.trunk.blocks.0.ff.0.weight"].shape[0]
    actor = EntityActor(device="cpu", graphs=False, d_model=d_model, d_ff=d_ff)
    actor.load_state_dict(state)
    return actor.eval()


def training_names(packet):
    """player_id -> training agent id: blue-0, blue-1, orange-0, orange-1 in packet order within each team."""
    count, names = [0, 0], {}
    for p in packet.players:
        names[p.player_id] = f"{('blue', 'orange')[p.team]}-{count[p.team]}"
        count[p.team] += 1
    return names


class PlaystyleBot(Bot):
    def initialize(self):
        torch.set_num_threads(1)
        self.actor = load_actor(HERE / "model")
        self.table = LookupTableAction.make_lookup_table()
        self.obs_builder = EntityObs()
        # rlgym-compat keys its cars by player_id; the training names are applied to a view of its state.
        self.game = GameState.create_compat_game_state(self.field_info, self.match_config)
        self.provider = TTBRoleProvider(coordination=False)
        self.info = self.provider.create({})
        self.controls = flat.ControllerState()
        self.prev_action = np.zeros(8, dtype=np.float32)
        self.last_tick = None

    def view(self, packet):
        names = training_names(packet)
        g = self.game
        return SimpleNamespace(cars={names[pid]: car for pid, car in g.cars.items()}, ball=g.ball,
                               boost_pad_timers=g.boost_pad_timers, tick_count=g.tick_count,
                               goal_scored=g.goal_scored), names

    def get_output(self, packet: flat.GamePacket) -> flat.ControllerState:
        if not packet.balls:
            return self.controls
        self.game.update(packet)
        tick = packet.match_info.frame_num
        if self.last_tick is not None and tick - self.last_tick < TICK_SKIP:
            return self.controls
        state, names = self.view(packet)
        me = next(names[p.player_id] for p in packet.players if p.player_id == self.player_id)
        if self.last_tick is None:
            self.provider.set_state(list(state.cars), state, self.info)
        else:
            self.provider.step(list(state.cars), state, self.info)
        self.last_tick = tick

        m = packet.match_info
        blue, orange = (t.score for t in sorted(packet.teams, key=lambda t: t.team_index))
        remaining = MATCH_SECONDS if m.is_unlimited_time else min(float(m.game_time_remaining), MATCH_SECONDS)
        self.info["match"] = {"score_diff": blue - orange, "time_remaining": remaining, "overtime": m.is_overtime}
        self.info["prev_action"] = {me: self.prev_action}

        obs = self.obs_builder.build_obs([me], state, self.info)[me]
        with torch.no_grad():
            action, _ = self.actor.get_action(None, [obs], deterministic=True)
        throttle, steer, pitch, yaw, roll, jump, boost, handbrake = self.prev_action = \
            self.table[int(action[0][0])].astype(np.float32)
        self.game.reset_car_ball_touches()
        self.controls = flat.ControllerState(throttle=throttle, steer=steer, pitch=pitch, yaw=yaw, roll=roll,
                                             jump=bool(jump), boost=bool(boost), handbrake=bool(handbrake))
        return self.controls


if __name__ == "__main__":
    PlaystyleBot().run()
