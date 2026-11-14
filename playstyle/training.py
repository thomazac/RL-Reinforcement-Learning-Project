"""Wiring between this project and RLGym-Learn 1.0.5 / rlgym-learn-algos 0.2.6: the attention actor and critic, a
PPO learner with a faster update, and the run configuration. The env each process builds is
playstyle.env.build_env. (RLGym-Learn 2.0.0's env processes do not start on Windows, rlgym-learn issue #35.)

How RLGym-Learn spends an iteration (read from its source):
  - One loop in the main process collects env steps, runs the actor on them and sends actions back. Once
    timesteps_per_iteration steps are in, the same loop runs the update. Env processes sit waiting for their
    next action meanwhile, so nothing is collected during the update.
  - Every epoch goes through the whole experience buffer, not just the new steps. With a buffer of 300,000 and
    100,000 new steps per iteration, each step is trained on 3 times per epoch.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
from rlgym_learn import (BaseConfigModel, LearningCoordinatorConfigModel, ProcessConfigModel, PyAnySerdeType,
                         SerdeTypesModel)
from rlgym_learn_algos.ppo import (
    ExperienceBufferConfigModel,
    GAETrajectoryProcessor,
    GAETrajectoryProcessorConfigModel,
    NumpyExperienceBuffer,
    PPOAgentController,
    PPOAgentControllerConfigModel,
    PPOData,
    PPOLearner,
    PPOLearnerConfigModel,
)
from torch.nn.utils import clip_grad_norm_, parameters_to_vector

from . import versions
from .network import EntityActor, EntityCritic

# Token width -> network settings. Width is fixed for a run and everything trained from it.
NETWORKS = {128: dict(d_model=128, n_heads=4, d_ff=256), 256: dict(d_model=256, n_heads=4, d_ff=512)}
CONTROLLER = "ppo"  # the agent controller's name in the run config and checkpoint folders
MAX_GRAD_NORM = 0.5  # fixed in rlgym-learn-algos 0.2.6, applied to actor and critic separately


def serde_types():
    """How agent ids, observations, actions and rewards cross between the env processes and the learner."""
    space = PyAnySerdeType.TUPLE((PyAnySerdeType.STRING(), PyAnySerdeType.INT()))
    return SerdeTypesModel(
        agent_id_serde_type=PyAnySerdeType.STRING(),  # "blue-0", "orange-1", ...
        obs_serde_type=PyAnySerdeType.NUMPY(np.float32),
        action_serde_type=PyAnySerdeType.NUMPY(np.int64),
        reward_serde_type=PyAnySerdeType.FLOAT(),
        obs_space_serde_type=space,
        action_space_serde_type=space,
    )


@dataclass
class PPOStats(PPOData):
    first_minibatch_ratio: float = 1.0  # mean new/old probability ratio before this update moved the policy


class Learner(PPOLearner):
    """rlgym-learn-algos 0.2.6's PPO update with three changes.

    precision="float16": the forward and backward passes run in half precision (autocast with gradient
    scaling); weights, losses and the optimizer stay in float32. Collection still runs in float32, so the old
    and new log-probabilities differ by float16 rounding (scripts/float16_check.py). precision="float32" gives
    the same weights as PPOLearner, bit for bit (tests/test_training.py).

    first_minibatch_ratio: the mean probability ratio on the first minibatch of each update, before any step.
    It should be very close to 1; a drift as the policy sharpens points at the float16 mismatch first.

    Checkpoints record the observation and reward versions (playstyle/versions.py) next to the weights, and a
    checkpoint trained on different observation versions will not load.
    """

    def __init__(self, actor_factory, critic_factory, precision="float16", reward_ttb="v1"):
        super().__init__(actor_factory, critic_factory)
        assert precision in ("float16", "float32")
        self.precision = precision
        self.reward_ttb = reward_ttb
        self.scaler = None

    def save_checkpoint(self, folder_path):
        super().save_checkpoint(folder_path)
        versions.write(folder_path, self.reward_ttb)

    def _load_from_checkpoint(self):
        versions.check(self.config.checkpoint_load_folder)
        super()._load_from_checkpoint()

    def learn(self, exp):
        cfg = self.config.learner_config
        device = cfg.device
        half = self.precision == "float16" and device.type == "cuda"
        if self.scaler is None:
            self.scaler = torch.amp.GradScaler(device.type, enabled=half)
        scaler = self.scaler
        actor_before = parameters_to_vector(self.actor.parameters())
        critic_before = parameters_to_vector(self.critic.parameters())
        totals = torch.zeros(4, device=device)  # entropy, KL, value loss, clip fraction
        first_ratio = None
        n_batches = 0

        start = time.time()
        for _ in range(cfg.n_epochs):
            for agent_ids, obs, actions, old_log_probs, values, advantages in exp.get_all_batches_shuffled(cfg.batch_size):
                targets = (values + advantages).to(device)
                advantages = advantages.to(device)
                if cfg.advantage_normalization:
                    std, mean = torch.std_mean(advantages)
                    advantages = (advantages - mean) / (std + 1e-8)
                old_log_probs = old_log_probs.to(device)
                obs = torch.as_tensor(obs, dtype=torch.float32).to(device)

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                for lo in range(0, cfg.batch_size, self.minibatch_size):  # gradients add up over minibatches
                    hi = min(lo + self.minibatch_size, cfg.batch_size)
                    share = (hi - lo) / cfg.batch_size
                    with torch.autocast(device.type, dtype=torch.float16, enabled=half):
                        predicted = self.critic(agent_ids[lo:hi], obs[lo:hi])
                        log_probs, entropy = self.actor.get_backprop_data(agent_ids[lo:hi], obs[lo:hi], actions[lo:hi])
                    predicted = predicted.float().view_as(targets[lo:hi])
                    log_ratio = log_probs.float().view_as(old_log_probs[lo:hi]) - old_log_probs[lo:hi]
                    ratio = torch.exp(log_ratio)
                    clipped = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
                    actor_loss = -torch.min(ratio * advantages[lo:hi], clipped * advantages[lo:hi]).mean() * share
                    value_loss = self.critic_loss_fn(predicted, targets[lo:hi]) * share
                    entropy = entropy.float() * share
                    scaler.scale(actor_loss - entropy * cfg.ent_coef).backward()
                    scaler.scale(value_loss).backward()
                    with torch.no_grad():
                        if first_ratio is None:
                            first_ratio = ratio.mean()
                        totals += torch.stack([entropy, ((ratio - 1) - log_ratio).mean() * share, value_loss,
                                               ((ratio - 1).abs() > cfg.clip_range).float().mean() * share])

                scaler.unscale_(self.actor_optimizer)
                scaler.unscale_(self.critic_optimizer)
                clip_grad_norm_(self.critic.parameters(), max_norm=MAX_GRAD_NORM)
                clip_grad_norm_(self.actor.parameters(), max_norm=MAX_GRAD_NORM)
                scaler.step(self.actor_optimizer)
                scaler.step(self.critic_optimizer)
                scaler.update()
                n_batches += 1

        actor_change = (actor_before - parameters_to_vector(self.actor.parameters())).norm().item()
        critic_change = (critic_before - parameters_to_vector(self.critic.parameters())).norm().item()
        entropy, kl, value_loss, clip_fraction = (totals / max(n_batches, 1)).tolist()  # waits for the GPU
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.cumulative_model_updates += n_batches
        return PPOStats((time.time() - start) / max(n_batches, 1), self.cumulative_model_updates, entropy, kl,
                        value_loss, clip_fraction, actor_change, critic_change,
                        1.0 if first_ratio is None else first_ratio.item())


def factories(width):
    """The two functions RLGym-Learn calls to build the actor and the critic."""

    def actor_factory(obs_space, action_space, device):
        return EntityActor(device=device, **NETWORKS[width])

    def critic_factory(obs_space, device):
        return EntityCritic(device=device, **NETWORKS[width])

    return actor_factory, critic_factory


def make_agent_controller(width=128, precision="float16", reward_ttb="v1", metrics_logger=None):
    controller = PPOAgentController(*factories(width), NumpyExperienceBuffer(GAETrajectoryProcessor()),
                                    metrics_logger=metrics_logger)
    controller.learner = Learner(*factories(width), precision=precision, reward_ttb=reward_ttb)
    return controller


def make_config(n_proc=16, timesteps_per_iteration=100_000, buffer_size=300_000, batch_size=100_000, n_minibatches=2,
                n_epochs=1, lr=2e-4, ent_coef=0.01, gamma=0.99, timestep_limit=5_000_000_000, device="cuda:0",
                save_folder="checkpoints", run_name="run", save_every_ts=5_000_000, seed=123, inference_procs=None):
    """Defaults: 100,000 steps per iteration, a 300,000-step buffer, batches of 100,000 as two minibatches of
    50,000, 1 epoch. Every epoch covers the whole buffer, so each step is trained on 3 times. (The design
    doc's 2 epochs would make that 6.) Reuse is a training setting and can change mid-run.

    inference_procs: how many env processes must have stepped before the actor runs on their observations
    (RLGym-Learn's default is 45% of them). Each actor call costs about the same whatever its size, so larger
    batches mean fewer calls, at the price of processes waiting for the rest."""
    process = ProcessConfigModel(n_proc=n_proc) if inference_procs is None else \
        ProcessConfigModel(n_proc=n_proc, min_process_steps_per_inference=inference_procs)
    return LearningCoordinatorConfigModel(
        base_config=BaseConfigModel(serde_types=serde_types(), timestep_limit=timestep_limit, random_seed=seed),
        process_config=process,
        agent_controllers_config={CONTROLLER: PPOAgentControllerConfigModel(
            timesteps_per_iteration=timesteps_per_iteration,
            save_every_ts=save_every_ts,
            run_name=run_name,
            random_seed=seed,
            learner_config=PPOLearnerConfigModel(
                n_epochs=n_epochs, batch_size=batch_size, n_minibatches=n_minibatches, ent_coef=ent_coef,
                actor_lr=lr, critic_lr=lr, device=device),
            experience_buffer_config=ExperienceBufferConfigModel(
                max_size=buffer_size, device="cpu",
                trajectory_processor_config=GAETrajectoryProcessorConfigModel(gamma=gamma).model_dump()),
        )},
        agent_controllers_save_folder=save_folder,
    )
