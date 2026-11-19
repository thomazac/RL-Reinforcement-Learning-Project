import random
from pathlib import Path

from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import AnyCondition, NoTouchTimeoutCondition, TimeoutCondition
from rlgym.rocket_league.reward_functions import GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, MutatorSequence

from .match import MatchEngine, MatchOverCondition, RandomMatchStart, SeededKickoff
from .obs import EntityObs, RecordingActionParser
from .rewards import Phase1bReward, ShapedReward
from .rewards_nx import NextoStyleReward
from .shared_info import TTBRoleProvider
from .states import MixedStart, PracticeEnd, ShotCurriculum

TICK_SKIP = 8
NO_TOUCH_SECONDS = 30
MAX_EPISODE_SECONDS = 600  # a 5-minute clock that only runs after kickoffs, plus overtime


REPLAY_STATES = Path(__file__).resolve().parents[1] / "data" / "replay_states.npz"


def make_env(team_size=2, obs_builder=None, reward_fn=None, reward_ttb="v1", coordination=True, seed=None,
             mixed_starts=False, start_weights=None):
    """seed fixes what the environment itself draws: kickoff spawn points, start states, starting score and clock.
    It cannot make a run repeat exactly: RocketSim's contact physics and demo respawn points differ between
    identical runs (scripts/seed_divergence.py, scripts/rocketsim_determinism.py).

    mixed_starts: episodes start from playstyle.states.MixedStart (kickoffs, replay moments, random states,
    coordination tests) instead of always a kickoff. Kickoffs after goals are unchanged."""
    kickoff = SeededKickoff(seed)  # one stream for episode starts and mid-match kickoffs
    start = kickoff
    truncation = [NoTouchTimeoutCondition(NO_TOUCH_SECONDS), TimeoutCondition(MAX_EPISODE_SECONDS)]
    if mixed_starts:
        # Shared: starts set the levels, practice endings feed them outcomes.
        curriculum, aerial_curriculum = ShotCurriculum(level=0.0), ShotCurriculum(level=0.0)
        start = MixedStart(kickoff, REPLAY_STATES if REPLAY_STATES.exists() else None,
                           seed=None if seed is None else [seed, 1],  # its own stream, apart from RandomMatchStart's
                           weights=start_weights, curriculum=curriculum, aerial_curriculum=aerial_curriculum)
        truncation.append(PracticeEnd(curriculum, aerial_curriculum))
    return RLGym(
        state_mutator=MutatorSequence(FixedTeamSizeMutator(team_size, team_size), start, RandomMatchStart(seed)),
        obs_builder=obs_builder or EntityObs(),
        action_parser=RecordingActionParser(RepeatAction(LookupTableAction(), repeats=TICK_SKIP)),
        reward_fn=reward_fn or GoalReward(),
        termination_cond=MatchOverCondition(),
        truncation_cond=AnyCondition(*truncation),
        transition_engine=MatchEngine(RocketSimEngine(), kickoff),
        shared_info_provider=TTBRoleProvider(reward_ttb=reward_ttb, coordination=coordination),
    )


REWARDS = {"1a": ShapedReward, "1b": Phase1bReward, "nx": NextoStyleReward}


def build_env(reward_ttb="v1", coordination=False, weights=None, team_spirit=0.0, opponent_scale=1.0, phase="1a",
              start_weights=None, gamma=0.99):
    """The env one RLGym-Learn process runs; the defaults are Phase 1a (playstyle.rewards.PHASE_1A, no
    coordination, team spirit 0). Phase "1b" uses playstyle.rewards.Phase1bReward and "nx"
    playstyle.rewards_nx.NextoStyleReward; both always turn coordination on, which their penalties read. "nx" also
    takes the run's gamma, which its potentials need. RLGym-Learn seeds Python's global random with a different
    number in each process before calling this, so drawing the env seed from it gives every process its own kickoffs
    and starting scores, repeatable from the run's seed. This module stays free of torch so env processes start
    fast and stay small. Use functools.partial to pass other settings; RLGym-Learn calls it with no arguments."""
    later = phase in ("1b", "nx")
    extra = {"gamma": gamma} if phase == "nx" else {}
    return make_env(reward_ttb=reward_ttb, coordination=coordination or later, seed=random.getrandbits(32),
                    reward_fn=REWARDS[phase](weights, team_spirit, opponent_scale, **extra), mixed_starts=later,
                    start_weights=start_weights)
