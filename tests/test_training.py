import json
import random

import numpy as np
import pytest
import torch
from rlgym_learn_algos.ppo import (DerivedPPOLearnerConfig, GAETrajectoryProcessor, GAETrajectoryProcessorConfigModel,
                                   PPOLearner, PPOLearnerConfigModel, Trajectory)
from rlgym_learn_algos.ppo.trajectory_processor import DerivedTrajectoryProcessorConfig
from torch.nn.utils import parameters_to_vector

from playstyle import versions
from playstyle.env import build_env
from playstyle.network import N_ACTIONS
from playstyle.obs import OBS_DIM
from playstyle.training import (KICKOFF, REWARD_CLIP, SELF_DEMOED, ClippedGAETrajectoryProcessor, Learner, factories,
                                observation_jumps)

BATCH = 64


class FixedBatches:
    """Stands in for the experience buffer: the same batches, in the same order, every time."""

    def __init__(self, n=3 * BATCH, seed=0):
        rng = np.random.default_rng(seed)
        self.data = (["blue-0"] * n,
                     rng.normal(0, 0.5, (n, OBS_DIM)).astype(np.float32),
                     rng.integers(0, N_ACTIONS, (n, 1)),
                     torch.tensor(rng.normal(-4.5, 0.05, n), dtype=torch.float32),
                     torch.tensor(rng.normal(0, 1, n), dtype=torch.float32),
                     torch.tensor(rng.normal(0, 1, n), dtype=torch.float32))

    def get_all_batches_shuffled(self, batch_size):
        for lo in range(0, len(self.data[0]) - batch_size + 1, batch_size):
            yield tuple(x[lo:lo + batch_size] for x in self.data)


def loaded(learner, device="cpu", checkpoint=None):
    torch.manual_seed(0)  # same starting weights for every learner
    learner.load(DerivedPPOLearnerConfig(
        learner_config=PPOLearnerConfigModel(n_epochs=2, batch_size=BATCH, n_minibatches=2, device=device),
        agent_controller_name="test", obs_space=("real", OBS_DIM), action_space=("discrete", N_ACTIONS),
        checkpoint_load_folder=None if checkpoint is None else str(checkpoint)))
    return learner


def weights(learner):
    return parameters_to_vector([*learner.actor.parameters(), *learner.critic.parameters()]).detach().cpu()


def test_float32_update_is_the_rlgym_learn_update():
    ours = loaded(Learner(*factories(128), precision="float32"))
    theirs = loaded(PPOLearner(*factories(128)))
    start = weights(ours)
    assert torch.equal(start, weights(theirs))
    a, b = ours.learn(FixedBatches()), theirs.learn(FixedBatches())
    assert a.cumulative_model_updates == b.cumulative_model_updates == 6
    assert (weights(ours) - start).abs().max() > 1e-4  # the update moved the weights
    assert torch.allclose(weights(ours), weights(theirs), atol=1e-6)
    for field in ("actor_entropy", "kl_divergence", "critic_loss", "sb3_clip_fraction",
                  "actor_update_magnitude", "critic_update_magnitude"):
        assert getattr(a, field) == pytest.approx(getattr(b, field), rel=1e-4, abs=1e-6), field


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mixed precision needs a GPU")
def test_float16_update_stays_close_to_float32():
    half = loaded(Learner(*factories(128), precision="float16"), device="cuda")
    full = loaded(Learner(*factories(128), precision="float32"), device="cuda")
    start = weights(full)
    a, b = half.learn(FixedBatches()), full.learn(FixedBatches())
    assert a.cumulative_model_updates == b.cumulative_model_updates == 6
    assert a.actor_entropy == pytest.approx(b.actor_entropy, rel=1e-3)
    assert a.critic_loss == pytest.approx(b.critic_loss, rel=2e-2)
    assert a.kl_divergence == pytest.approx(b.kl_divergence, rel=0.1, abs=1e-6)
    moved = (weights(full) - start).norm()
    # Measured 0.16: Adam turns small gradient differences into full-size steps on weights whose gradient is
    # near zero. A broken float16 path (no gradient unscaling, skipped steps) gives 1 or more.
    assert (weights(half) - weights(full)).norm() < 0.5 * moved
    assert weights(half).dtype == torch.float32  # weights themselves stay in float32


class RecordingLoss(torch.nn.MSELoss):
    def __init__(self):
        super().__init__()
        self.targets = []

    def forward(self, predicted, target):
        self.targets.append(target.detach().float().cpu().view(-1).clone())
        return super().forward(predicted, target)


DEVICES = ["cpu"] + (["cuda"] * torch.cuda.is_available())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("learner_class", ["ours_float32", "ours_float16", "rlgym_learn"])
def test_critic_targets_are_values_plus_advantages(learner_class, device):
    """rlgym-learn-algos issue #30: non-blocking copies let the targets be built before the copies finished,
    leaving the critic with targets equal to its own predictions (critic loss 0, critic never moves)."""
    if learner_class == "ours_float16" and device == "cpu":
        pytest.skip("float16 runs only on the GPU")
    learner = {"ours_float32": lambda: Learner(*factories(128), precision="float32"),
               "ours_float16": lambda: Learner(*factories(128), precision="float16"),
               "rlgym_learn": lambda: PPOLearner(*factories(128))}[learner_class]()
    loaded(learner, device=device)
    learner.critic_loss_fn = RecordingLoss()
    batches = FixedBatches()
    data = learner.learn(batches)

    expected = []
    for _ in range(2):  # epochs
        for _, _, _, _, values, advantages in batches.get_all_batches_shuffled(BATCH):
            expected += list((values + advantages).split(BATCH // 2))
    assert len(learner.critic_loss_fn.targets) == len(expected) == 12
    for got, want in zip(learner.critic_loss_fn.targets, expected):
        assert torch.allclose(got, want, atol=1e-6)
    assert data.critic_loss > 0.1
    assert data.critic_update_magnitude > 0


def hand_gae(rewards, values, final_value, truncated, gamma=0.99, lmbda=0.95):
    following = np.append(values[1:], final_value if truncated else 0.0)
    deltas = rewards + gamma * following - values
    advantages, running = np.zeros_like(values), 0.0
    for i in reversed(range(len(values))):
        running = deltas[i] + gamma * lmbda * running
        advantages[i] = running
    return advantages


def test_truncated_trajectories_bootstrap_from_their_final_observation():
    """GAE as RLGym-Learn computes it: a truncated trajectory continues from the value of its final observation
    (the one before the reset); a terminated one stops there."""
    processor = GAETrajectoryProcessor()
    processor.load(DerivedTrajectoryProcessorConfig(
        GAETrajectoryProcessorConfigModel(standardize_returns=False), "test", torch.float32, torch.device("cpu")))
    rng = np.random.default_rng(0)
    trajectories, expected = [], []
    for i, truncated in enumerate((True, False, True, False)):
        n = 20 + 7 * i
        rewards = rng.normal(0, 1, n)
        values = rng.normal(0, 1, n)
        final_value = 5.0  # large, so bootstrapping or not is plain in the result
        trajectories.append(Trajectory("blue-0", [np.zeros(OBS_DIM, np.float32)] * n, [np.zeros(1, np.int64)] * n,
                                       torch.zeros(n), list(rewards), torch.tensor(values, dtype=torch.float32),
                                       np.zeros(OBS_DIM, np.float32), torch.tensor(final_value), truncated))
        expected.append(hand_gae(rewards, values, final_value, truncated))
    (_, _, _, _, _, advantages), _ = processor.process_trajectories(trajectories)
    assert np.allclose(advantages.numpy(), np.concatenate(expected), atol=1e-4)


def gae_processor(cls, standardize=True):
    processor = cls()
    processor.load(DerivedTrajectoryProcessorConfig(
        GAETrajectoryProcessorConfigModel(standardize_returns=standardize), "test", torch.float32, torch.device("cpu")))
    return processor


def trajectory(rewards, values, final_value=0.0, truncated=False, obs=None):
    n = len(rewards)
    obs = [np.zeros(OBS_DIM, np.float32)] * (n + 1) if obs is None else obs
    return Trajectory("blue-0", list(obs[:n]), [np.zeros(1, np.int64)] * n, torch.zeros(n), list(rewards),
                      torch.tensor(values, dtype=torch.float32), obs[n], torch.tensor(final_value), truncated)


@pytest.mark.parametrize("cls, clip", [(GAETrajectoryProcessor, None), (ClippedGAETrajectoryProcessor, REWARD_CLIP)])
def test_reward_clip(cls, clip):
    """rlgym-learn-algos 0.2.6 does not clip; ours clips at +-10. The running return spread starts at exactly 1,
    so rewards of +-50 pass through unscaled and come out at +-50 or +-10."""
    rewards, values = np.zeros(30), np.zeros(30)
    rewards[[5, 20]] = 50.0, -50.0
    (_, _, _, _, _, advantages), stats = gae_processor(cls).process_trajectories([trajectory(rewards, values)])
    scaled = rewards if clip is None else np.clip(rewards, -clip, clip)
    assert np.allclose(advantages.numpy(), hand_gae(scaled, values, 0.0, False), atol=1e-3)
    if clip is not None:
        assert stats.reward_clip_share == pytest.approx(2 / 30)
        assert stats.largest_scaled_reward == pytest.approx(50.0)


def test_clipped_processor_matches_rlgym_learn_when_nothing_is_clipped():
    """Same advantages, returns and running return statistics as 0.2.6's processor, over several iterations
    (from the second on, rewards are divided by a spread other than 1)."""
    rng = np.random.default_rng(1)
    ours, theirs = gae_processor(ClippedGAETrajectoryProcessor), gae_processor(GAETrajectoryProcessor)
    for _ in range(3):
        batch = [trajectory(rng.normal(0, 0.3, n), rng.normal(0, 1, n), 0.5, bool(n % 2)) for n in (40, 57, 90)]
        (*_, a), stats = ours.process_trajectories(batch)
        (*_, b), data = theirs.process_trajectories(batch)
        assert np.allclose(a.numpy(), b.numpy(), atol=1e-5)
        assert stats.reward_clip_share == 0.0
        assert stats.average_return == pytest.approx(data.average_return)
        assert np.allclose(ours.return_stats.std, theirs.return_stats.std)
    assert ours.return_stats.std[0] != 1.0


def test_observation_jump_monitor():
    n = 40
    obs = np.zeros((n + 1, OBS_DIM), np.float32)
    obs[:, 0] = np.linspace(0, 1, n + 1)  # the car drives steadily: 0.025 per step
    obs[:, 124] = np.linspace(0.5, -0.5, n + 1)
    obs[20:, 124] = 0.0  # a goal: the ball goes back to the centre for a kickoff...
    obs[20:25, KICKOFF] = 1.0  # ...which the kickoff flag marks
    obs[30:33, SELF_DEMOED] = 1.0  # demoed, then respawned somewhere else
    obs[33:, 1] = 0.7
    assert observation_jumps([trajectory(np.zeros(n), np.zeros(n), obs=list(obs))]) == (pytest.approx(0.025, rel=1e-5), 0)

    broken = obs.copy()
    broken[10, 125] = 3.0  # a garbage read mid-run
    broken[12, 50] = np.nan
    largest, nonfinite = observation_jumps([trajectory(np.zeros(n), np.zeros(n), obs=list(broken))])
    assert largest == pytest.approx(3.0) and nonfinite == 1

    late = obs.copy()
    late[n, 2] = 0.9  # the final observation (what a truncated trajectory bootstraps from) is checked too
    assert observation_jumps([trajectory(np.zeros(n), np.zeros(n), obs=list(late))])[0] == pytest.approx(0.9)


def test_checkpoint_records_versions_and_only_loads_matching_observations(tmp_path):
    learner = loaded(Learner(*factories(128), precision="float32"))
    learner.learn(FixedBatches())
    learner.save_checkpoint(tmp_path)
    saved = json.loads((tmp_path / versions.FILENAME).read_text())
    assert saved["obs_versions"] == versions.OBS_VERSIONS
    assert saved["reward_versions"] == versions.reward_versions("v1")

    restored = loaded(Learner(*factories(128), precision="float32"), checkpoint=tmp_path)
    assert torch.equal(weights(restored), weights(learner))

    saved["obs_versions"]["layout"] = "entity-v0"
    (tmp_path / versions.FILENAME).write_text(json.dumps(saved))
    with pytest.raises(RuntimeError, match="entity-v0"):
        loaded(Learner(*factories(128), precision="float32"), checkpoint=tmp_path)

    (tmp_path / versions.FILENAME).unlink()
    with pytest.raises(RuntimeError, match=versions.FILENAME):
        loaded(Learner(*factories(128), precision="float32"), checkpoint=tmp_path)


def test_build_env_takes_its_seed_from_the_process_random_stream():
    def first_obs(seed):
        random.seed(seed)  # what RLGym-Learn does in each env process before building the env
        env = build_env()
        obs = env.reset()
        return np.concatenate([obs[a] for a in sorted(obs)])

    assert np.array_equal(first_obs(5), first_obs(5))
    assert any(not np.array_equal(first_obs(5), first_obs(s)) for s in (6, 7, 8))
