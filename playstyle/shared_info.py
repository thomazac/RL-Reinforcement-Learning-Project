"""Per-tick TTB, role and coordination pipeline, shared by the observation builder and reward through shared_info."""

from rlgym.api import SharedInfoProvider

from .ball_prediction import TICK_RATE, BallPredictor
from .coordination import CoordinationTracker
from .roles import RoleTracker, is_kickoff
from .ttb import ttb_from_state


class TTBRoleProvider(SharedInfoProvider):
    """Writes shared_info["ttb"], ["roles"], ["role_locked"], ["coordination"], ["ball_slices"] and ["kickoff"]."""

    def __init__(self):
        self.predictor = BallPredictor()
        self.roles = RoleTracker()
        self.coordination = CoordinationTracker()
        self._last_tick = 0

    def create(self, shared_info):
        return shared_info

    def set_state(self, agents, initial_state, shared_info):
        ttb, slices = ttb_from_state(initial_state, self.predictor)
        self._last_tick = initial_state.tick_count
        self.coordination.reset()
        roles = self.roles.reset(initial_state, ttb)
        return self._write(shared_info, initial_state, ttb, slices, roles, 0.0)

    def step(self, agents, state, shared_info):
        ttb, slices = ttb_from_state(state, self.predictor)
        dt = (state.tick_count - self._last_tick) / TICK_RATE
        self._last_tick = state.tick_count
        return self._write(shared_info, state, ttb, slices, self.roles.update(state, ttb, dt), dt)

    def _write(self, shared_info, state, ttb, slices, roles, dt):
        shared_info.update(
            ttb=ttb, roles=roles, ball_slices=slices, kickoff=is_kickoff(state),
            role_locked={team: tr.locked for team, tr in self.roles.teams.items()},
            coordination=self.coordination.update(state, ttb, roles, slices, dt),
        )
        return shared_info
