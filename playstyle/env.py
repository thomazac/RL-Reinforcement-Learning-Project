from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import AnyCondition, NoTouchTimeoutCondition, TimeoutCondition
from rlgym.rocket_league.reward_functions import GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, MutatorSequence

from .match import MatchEngine, MatchOverCondition, RandomMatchStart, SeededKickoff
from .obs import EntityObs, RecordingActionParser
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
