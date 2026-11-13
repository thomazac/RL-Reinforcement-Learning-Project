"""Per-tick TTB, role and coordination pipeline, shared by the observation builder and reward through shared_info."""

from rlgym.api import SharedInfoProvider

from . import ttb, ttb_v1
from .ball_prediction import BallPredictor
from .coordination import CoordinationTracker
from .roles import RoleTracker, is_kickoff
from .versions import OBS_VERSIONS, REWARD_TTB, reward_versions


class TTBRoleProvider(SharedInfoProvider):
    """Always writes shared_info["obs_ttb"] and ["obs_ball_slices"] (frozen v1, observation side),
    ["ttb"] and ["ball_slices"] (reward side), ["kickoff"], ["obs_versions"] and ["reward_versions"].
    With coordination on, also ["roles"], ["role_locked"] and ["coordination"].

    reward_ttb="v1" reuses the frozen estimate for the reward side; "live" runs playstyle/ttb.py separately.
    Phase 1a has no coordination terms, so its workers skip roles and coordination.
    """

    def __init__(self, reward_ttb="v1", coordination=True):
        if reward_ttb not in REWARD_TTB:
            raise ValueError(f"reward_ttb must be one of {list(REWARD_TTB)}")
        self.reward_ttb, self.use_coordination = reward_ttb, coordination
        self.obs_predictor = ttb_v1.BallPredictor()
        self.predictor = BallPredictor() if reward_ttb == "live" else None
        self.roles = RoleTracker()
        self.coordination = CoordinationTracker()
        self._last_tick = 0

    def create(self, shared_info):
        shared_info["obs_versions"] = dict(OBS_VERSIONS)
        shared_info["reward_versions"] = reward_versions(self.reward_ttb)
        return shared_info

    def set_state(self, agents, initial_state, shared_info):
        self._last_tick = initial_state.tick_count
        self.coordination.reset()
        return self._write(shared_info, initial_state, None)

    def step(self, agents, state, shared_info):
        dt = (state.tick_count - self._last_tick) / ttb_v1.TICK_RATE
        self._last_tick = state.tick_count
        return self._write(shared_info, state, dt)

    def _write(self, shared_info, state, dt):
        """dt None means a fresh episode."""
        obs_ttb, obs_slices = ttb_v1.ttb_from_state(state, self.obs_predictor)
        if self.predictor is None:
            reward_ttb, slices = obs_ttb, obs_slices
        else:
            reward_ttb, slices = ttb.ttb_from_state(state, self.predictor)
        shared_info.update(obs_ttb=obs_ttb, obs_ball_slices=obs_slices, ttb=reward_ttb, ball_slices=slices,
                           kickoff=is_kickoff(state))
        if self.use_coordination:
            roles = self.roles.reset(state, reward_ttb) if dt is None else self.roles.update(state, reward_ttb, dt)
            shared_info.update(
                roles=roles, role_locked={team: tr.locked for team, tr in self.roles.teams.items()},
                coordination=self.coordination.update(state, reward_ttb, roles, slices, dt or 0.0),
            )
        return shared_info
