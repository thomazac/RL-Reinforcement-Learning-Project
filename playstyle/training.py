"""Wiring between this project and RLGym-Learn: the attention actor and critic, a PPO learner with a faster
update, and the run configuration. The env each process builds is playstyle.env.build_env.

How RLGym-Learn 2.0 spends an iteration:
  - One loop in the main process collects env steps, runs the actor on them and sends actions back. Once
    timesteps_per_iteration steps are in, the same loop runs the update. Env processes sit waiting for their
    next action meanwhile, so nothing is collected during the update. (Read from its source; not yet seen
    running, see scripts/throughput.py.)
  - Every epoch goes through the whole experience buffer, not just the new steps. With a buffer of 300,000 and
    100,000 new steps per iteration, each step is trained on 3 times per epoch (measured, scripts/learner_cost.py).
"""

import time

import numpy as np
import torch
from rlgym_learn import BaseConfigModel, LearningCoordinatorConfigModel, ProcessConfigModel, SerdeTypesModel
from rlgym_learn.pyany_serde import PyAnySerdeType
from rlgym_learn_algos.ppo import (
    ExperienceBufferConfigModel,
    GAETrajectoryProcessor,
    GAETrajectoryProcessorConfigModel,
    NumpyExperienceBuffer,
    PPOAgentController,
    PPOAgentControllerConfigModel,
    PPOLearnerConfigModel,
    SeparateActorCritic,
)
from rlgym_learn_algos.ppo.ppo_learner import PPOData, PPOLearner

from . import versions
from .network import EntityActor, EntityCritic

# Token width -> network settings. Width is fixed for a run and everything trained from it.
NETWORKS = {128: dict(d_model=128, n_heads=4, d_ff=256), 256: dict(d_model=256, n_heads=4, d_ff=512)}


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


class Learner(PPOLearner):
    """RLGym-Learn's PPO update with two changes.

    precision="float16": the forward and backward passes run in half precision (autocast with gradient
    scaling); weights, losses and the optimizer stay in float32. The PPO passes take a bit under half the
    time on an RTX 3090 (scripts/learner_cost.py). Collection still runs in float32, so the old and new
    log-probabilities differ by float16 rounding (scripts/float16_check.py). precision="float32" gives the
    same weights as PPOLearner, bit for bit (tests/test_training.py).

    Checkpoints record the observation and reward versions (playstyle/versions.py) next to the weights, and a
    checkpoint trained on different observation versions will not load.
    """

    def __init__(self, actor_critic_factory, optimizers_factory, precision="float16", reward_ttb="v1"):
        super().__init__(actor_critic_factory, optimizers_factory)
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
        actor_before = self.actor_critic.get_actor_parameter_vector()
        critic_before = self.actor_critic.get_critic_parameter_vector()
        totals = torch.zeros(4, device=device)  # entropy, KL, value loss, clip fraction
        n_batches = 0

        start = time.time()
        for _ in range(cfg.n_epochs):
            for agent_ids, obs, actions, old_log_probs, values, advantages in exp.get_all_batches_shuffled(cfg.batch_size):
                obs = torch.as_tensor(obs, dtype=torch.float32).to(device, non_blocking=True)
                actions = torch.as_tensor(actions).to(device, non_blocking=True)
                old_log_probs = old_log_probs.to(device, non_blocking=True)
                advantages = advantages.to(device, non_blocking=True)
                targets = values.to(device, non_blocking=True) + advantages
                if cfg.advantage_standardization:
                    std, mean = torch.std_mean(advantages)
                    advantages = (advantages - mean) / (std + 1e-8)

                for optimizer in self.optimizers:
                    optimizer.zero_grad()
                for lo in range(0, cfg.batch_size, self.minibatch_size):  # gradients add up over minibatches
                    hi = min(lo + self.minibatch_size, cfg.batch_size)
                    share = (hi - lo) / cfg.batch_size
                    with torch.autocast(device.type, dtype=torch.float16, enabled=half):
                        log_probs, entropy, predicted = self.actor_critic.get_backprop_data(
                            agent_ids[lo:hi], obs[lo:hi], actions[lo:hi])
                    log_ratio = log_probs.float().view(-1) - old_log_probs[lo:hi]
                    ratio = torch.exp(log_ratio)
                    clipped = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
                    actor_loss = -torch.min(ratio * advantages[lo:hi], clipped * advantages[lo:hi]).mean() * share
                    value_loss = self.critic_loss_fn(predicted.float().view(-1), targets[lo:hi]) * share
                    entropy = entropy.float() * share
                    scaler.scale(actor_loss - entropy * cfg.ent_coef + value_loss).backward()
                    with torch.no_grad():
                        totals += torch.stack([entropy, ((ratio - 1) - log_ratio).mean() * share, value_loss,
                                               ((ratio - 1).abs() > cfg.clip_range).float().mean() * share])

                for optimizer in self.optimizers:
                    scaler.unscale_(optimizer)
                if cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=cfg.max_grad_norm)
                for optimizer in self.optimizers:
                    scaler.step(optimizer)
                scaler.update()
                n_batches += 1

        actor_change = (actor_before - self.actor_critic.get_actor_parameter_vector()).norm().item()
        critic_change = (critic_before - self.actor_critic.get_critic_parameter_vector()).norm().item()
        entropy, kl, value_loss, clip_fraction = (totals / max(n_batches, 1)).tolist()  # waits for the GPU
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        self.cumulative_model_updates += n_batches
        return PPOData((time.time() - start) / max(n_batches, 1), self.cumulative_model_updates,
                       entropy, kl, value_loss, clip_fraction, actor_change, critic_change)


def factories(width):
    """The two functions RLGym-Learn calls to build the networks and their optimizers."""

    def actor_critic_factory(obs_space, action_space, dtype, device, name):
        return SeparateActorCritic(EntityActor(device=device, **NETWORKS[width]),
                                   EntityCritic(device=device, **NETWORKS[width]))

    def optimizers_factory(actor_critic, group_kwargs, name):
        return [torch.optim.Adam(actor_critic.actor.parameters(), **group_kwargs["actor"]),
                torch.optim.Adam(actor_critic.critic.parameters(), **group_kwargs["critic"])]

    return actor_critic_factory, optimizers_factory


def make_agent_controller(width=128, precision="float16", reward_ttb="v1", metrics_logger=None):
    controller = PPOAgentController(*factories(width), NumpyExperienceBuffer(GAETrajectoryProcessor()),
                                    metrics_logger=metrics_logger)
    controller.learner = Learner(*factories(width), precision=precision, reward_ttb=reward_ttb)
    return controller


def make_config(n_proc=16, timesteps_per_iteration=100_000, buffer_size=300_000, batch_size=100_000, n_minibatches=2,
                n_epochs=1, lr=2e-4, ent_coef=0.01, gamma=0.99, timestep_limit=5_000_000_000, device="cuda:0",
                save_folder="checkpoints", run_name="run", save_every_ts=5_000_000, seed=123):
    """Defaults: 100,000 steps per iteration, a 300,000-step buffer, batches of 100,000 as two minibatches of
    50,000, 1 epoch. Every epoch covers the whole buffer, so each step is trained on 3 times. (The design
    doc's 2 epochs would make that 6.) Reuse is a training setting and can change mid-run."""
    return LearningCoordinatorConfigModel(
        base_config=BaseConfigModel(serde_types=serde_types(), timestep_limit=timestep_limit, random_seed=seed),
        process_config=ProcessConfigModel(n_proc=n_proc),
        agent_controller_config=PPOAgentControllerConfigModel(
            timesteps_per_iteration=timesteps_per_iteration,
            save_every_ts=save_every_ts,
            run_name=run_name,
            random_seed=seed,
            learner_config=PPOLearnerConfigModel(
                n_epochs=n_epochs, batch_size=batch_size, n_minibatches=n_minibatches, ent_coef=ent_coef,
                optimizer_named_parameter_group_kwargs={"actor": {"lr": lr}, "critic": {"lr": lr}}, device=device),
            experience_buffer_config=ExperienceBufferConfigModel(
                max_size=buffer_size, device="cpu",
                trajectory_processor_config=GAETrajectoryProcessorConfigModel(gamma=gamma)),
        ),
        agent_controller_save_folder=save_folder,
    )
