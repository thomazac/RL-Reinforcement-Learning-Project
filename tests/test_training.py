import json
import random

import numpy as np
import pytest
import torch
from rlgym_learn_algos.ppo import PPOLearnerConfigModel
from rlgym_learn_algos.ppo.ppo_learner import DerivedPPOLearnerConfig, PPOLearner
from torch.nn.utils import parameters_to_vector

from playstyle import versions
from playstyle.env import build_env
from playstyle.network import N_ACTIONS
from playstyle.obs import OBS_DIM
from playstyle.training import Learner, factories

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
        agent_controller_name=None, obs_space=("real", OBS_DIM), action_space=("discrete", N_ACTIONS),
        checkpoint_load_folder=None if checkpoint is None else str(checkpoint)))
    return learner


def weights(learner):
    return parameters_to_vector(learner.actor_critic.parameters()).detach().cpu()


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
