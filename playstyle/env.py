from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import AnyCondition, GoalCondition, NoTouchTimeoutCondition, TimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence

from .shared_info import TTBRoleProvider

TICK_SKIP = 8


def make_env(team_size=2, obs_builder=None, reward_fn=None):
    return RLGym(
        state_mutator=MutatorSequence(FixedTeamSizeMutator(team_size, team_size), KickoffMutator()),
        obs_builder=obs_builder or DefaultObs(zero_padding=None),
        action_parser=RepeatAction(LookupTableAction(), repeats=TICK_SKIP),
        reward_fn=reward_fn or GoalReward(),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(NoTouchTimeoutCondition(30), TimeoutCondition(300)),
        transition_engine=RocketSimEngine(),
        shared_info_provider=TTBRoleProvider(),
    )
