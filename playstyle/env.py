import random

from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import AnyCondition, NoTouchTimeoutCondition, TimeoutCondition
from rlgym.rocket_league.reward_functions import GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, MutatorSequence

from .match import MatchEngine, MatchOverCondition, RandomMatchStart, SeededKickoff
from .obs import EntityObs, RecordingActionParser
from .rewards import Phase1bReward, ShapedReward
from .shared_info import TTBRoleProvider

TICK_SKIP = 8
NO_TOUCH_SECONDS = 30
MAX_EPISODE_SECONDS = 600  # a 5-minute clock that only runs after kickoffs, plus overtime


def make_env(team_size=2, obs_builder=None, reward_fn=None, reward_ttb="v1", coordination=True, seed=None):
    """seed fixes what the environment itself draws: kickoff spawn points, starting score and clock. It cannot
    make a run repeat exactly: RocketSim's contact physics and demo respawn points differ between identical
    runs (scripts/seed_divergence.py, scripts/rocketsim_determinism.py)."""
    kickoff = SeededKickoff(seed)  # one stream for episode starts and mid-match kickoffs
    return RLGym(
        state_mutator=MutatorSequence(FixedTeamSizeMutator(team_size, team_size), kickoff, RandomMatchStart(seed)),
        obs_builder=obs_builder or EntityObs(),
        action_parser=RecordingActionParser(RepeatAction(LookupTableAction(), repeats=TICK_SKIP)),
        reward_fn=reward_fn or GoalReward(),
        termination_cond=MatchOverCondition(),
        truncation_cond=AnyCondition(NoTouchTimeoutCondition(NO_TOUCH_SECONDS), TimeoutCondition(MAX_EPISODE_SECONDS)),
        transition_engine=MatchEngine(RocketSimEngine(), kickoff),
        shared_info_provider=TTBRoleProvider(reward_ttb=reward_ttb, coordination=coordination),
    )


REWARDS = {"1a": ShapedReward, "1b": Phase1bReward}


def build_env(reward_ttb="v1", coordination=False, weights=None, team_spirit=0.0, opponent_scale=1.0, phase="1a"):
    """The env one RLGym-Learn process runs; the defaults are Phase 1a (playstyle.rewards.PHASE_1A, no
    coordination, team spirit 0). Phase "1b" uses playstyle.rewards.Phase1bReward and always turns coordination on,
    which its penalties read. RLGym-Learn seeds Python's global random with a different number in each
    process before calling this, so drawing the env seed from it gives every process its own kickoffs and
    starting scores, repeatable from the run's seed. This module stays free of torch so env processes start
    fast and stay small. Use functools.partial to pass other settings; RLGym-Learn calls it with no arguments."""
    return make_env(reward_ttb=reward_ttb, coordination=coordination or phase == "1b", seed=random.getrandbits(32),
                    reward_fn=REWARDS[phase](weights, team_spirit, opponent_scale))
