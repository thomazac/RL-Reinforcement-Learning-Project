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

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import psutil
import torch
from rlgym_learn import (BaseConfigModel, LearningCoordinatorConfigModel, ProcessConfigModel, PyAnySerdeType,
                         SerdeTypesModel)
from rlgym_learn_algos.ppo import (
    ExperienceBufferConfigModel,
    GAETrajectoryProcessor,
    GAETrajectoryProcessorConfigModel,
    GAETrajectoryProcessorData,
    NumpyExperienceBuffer,
    PPOAgentController,
    PPOAgentControllerConfigModel,
    PPOData,
    PPOLearner,
    PPOLearnerConfigModel,
    PPOMetricsLogger,
    Trajectory,
)
from torch.nn.utils import clip_grad_norm_, parameters_to_vector

from . import versions
from .ball_prediction import TICK_RATE
from .env import TICK_SKIP
from .network import EntityActor, EntityCritic
from .rewards import PHASE_1A

STEPS_PER_SECOND = TICK_RATE / TICK_SKIP  # agent decisions per second of game time

# Token width -> network settings. Width is fixed for a run and everything trained from it.
NETWORKS = {128: dict(d_model=128, n_heads=4, d_ff=256), 256: dict(d_model=256, n_heads=4, d_ff=512)}
WIDTH = 256  # locked for Phase 1a onward: cost 17.6-19.6% of overall throughput against 128, under the 20% line
CONTROLLER = "ppo"  # the agent controller's name in the run config and checkpoint folders
MAX_GRAD_NORM = 0.5  # fixed in rlgym-learn-algos 0.2.6, applied to actor and critic separately


def serde_types():
    """How agent ids, observations, actions and rewards cross between the env processes and the learner."""
    space = PyAnySerdeType.TUPLE((PyAnySerdeType.STRING(), PyAnySerdeType.INT()))
    return SerdeTypesModel(
        agent_id_serde_type=PyAnySerdeType.STRING(),  # "blue-0", "orange-1", ...
        obs_serde_type=PyAnySerdeType.NUMPY(np.float32),
        action_serde_type=PyAnySerdeType.NUMPY(np.int64),
        reward_serde_type=PyAnySerdeType.NUMPY(np.float32),  # [reward, raw components...], playstyle/rewards.py
        obs_space_serde_type=space,
        action_space_serde_type=space,
    )


REWARD_CLIP = 10.0  # on rewards divided by the running return spread, as in RLGym-PPO
CLIP_SHARE_WARNING = 0.01

# Observation indices for the jump monitor (playstyle/obs.py): the observer's car and the ball position on the
# 1/5120 scale, the observer's demoed flag and the kickoff flag.
SELF_POS, BALL_POS, SELF_DEMOED, KICKOFF = slice(0, 3), slice(124, 127), 28, 185
JUMP_WARNING = 0.15  # one 8-tick step moves a car or the ball at most about 0.1 on that scale


def observation_jumps(trajectories):
    """Largest one-step position change of the observer's car or the ball within any trajectory (final
    observation included), and the number of non-finite observation values. Steps that legitimately teleport
    are left out: into a kickoff after a goal, and while the observer is demoed or respawning. A garbage read
    from the env processes would show up as a large jump or a non-finite value."""
    largest, nonfinite = 0.0, 0
    for t in trajectories:
        obs = np.stack(t.obs_list + [t.final_obs])
        nonfinite += int((~np.isfinite(obs)).sum())
        if len(obs) < 2:
            continue
        pos = np.concatenate([obs[:, SELF_POS], obs[:, BALL_POS]], axis=1)
        jump = np.abs(np.diff(pos, axis=0)).max(axis=1)
        demoed = obs[:, SELF_DEMOED] > 0.5
        into_kickoff = (obs[1:, KICKOFF] > 0.5) & (obs[:-1, KICKOFF] < 0.5)
        keep = ~(demoed[1:] | demoed[:-1] | into_kickoff)
        if keep.any():
            largest = max(largest, float(np.nan_to_num(jump[keep], nan=np.inf).max()))
    return largest, nonfinite


@dataclass
class RunStats(GAETrajectoryProcessorData):
    reward_clip_share: float = 0.0  # share of steps whose scaled reward hit the clip
    largest_scaled_reward: float = 0.0  # largest |reward / return spread| before clipping
    largest_obs_jump: float = 0.0
    nonfinite_obs: int = 0
    component_means: dict = field(default_factory=dict)  # raw reward component -> mean per agent-step


class ClippedGAETrajectoryProcessor(GAETrajectoryProcessor):
    """rlgym-learn-algos 0.2.6's GAE with the scaled rewards clipped at +-REWARD_CLIP (0.2.6 divides rewards by
    the running return spread but does not clip). Returns, and so the return statistics, stay in raw reward
    units as in 0.2.6. Also reports the observation-jump monitor, since every trajectory passes through here.

    Rewards may be plain numbers or vectors [reward, raw components...] (playstyle/rewards.py); PPO trains on the
    first entry, and the others are averaged per component for the log."""

    def __init__(self, component_names=tuple(PHASE_1A)):
        super().__init__()
        self.component_names = list(component_names)

    def process_trajectories(self, trajectories):
        rust = self.rust_gae_trajectory_processor
        spread = float(self.return_stats.std[0]) if self.standardize_returns else 1.0
        rewards = [np.asarray(t.reward_list, dtype=np.float64) for t in trajectories]
        totals = [r[:, 0] if r.ndim == 2 else r for r in rewards]
        parts = [r[:, 1:] for r in rewards if r.ndim == 2]

        def with_rewards(t, r):
            return Trajectory(t.agent_id, t.obs_list, t.action_list, t.log_probs, r.tolist(), t.val_preds,
                              t.final_obs, t.final_val_pred, t.truncated)

        ids, obs, actions, log_probs, values, _, returns, avg_reward, avg_return = \
            rust.process_trajectories([with_rewards(t, r) for t, r in zip(trajectories, totals)], spread)
        scaled = [r / spread for r in totals]
        clipped = [with_rewards(t, np.clip(s, -REWARD_CLIP, REWARD_CLIP)) for t, s in zip(trajectories, scaled)]
        advantages = rust.process_trajectories(clipped, 1.0)[5]
        if self.standardize_returns:
            for sample in returns[:min(self.max_returns_per_stats_increment, len(returns))]:
                self.return_stats.update(sample)

        all_scaled = np.concatenate(scaled) if scaled else np.zeros(1)
        largest_jump, nonfinite = observation_jumps(trajectories)
        means = np.concatenate(parts).mean(axis=0) if parts else []
        stats = RunStats(average_reward=avg_reward, average_undiscounted_episodic_return=avg_return,
                         average_return=returns.mean(), return_standard_deviation=returns.std(),
                         reward_clip_share=float((np.abs(all_scaled) > REWARD_CLIP).mean()),
                         largest_scaled_reward=float(np.abs(all_scaled).max()),
                         largest_obs_jump=largest_jump, nonfinite_obs=nonfinite,
                         component_means={n: float(m) for n, m in zip(self.component_names, means)})
        return ((ids, obs, actions, log_probs.to(device=self.device), values.to(device=self.device),
                 torch.from_numpy(advantages).to(device=self.device)), stats)


class ExperienceBuffer(NumpyExperienceBuffer):
    """NumpyExperienceBuffer that also restores the trajectory processor's running return statistics from a
    checkpoint. rlgym-learn-algos 0.2.6 saves them but never passes the checkpoint folder on when loading, so a
    resumed run would restart reward scaling from a spread of 1 (found by scripts/train.py's resume check)."""

    def load(self, config):
        super().load(config)
        if config.checkpoint_load_folder is not None:
            self.trajectory_processor.checkpoint_load_folder = config.checkpoint_load_folder
            self.trajectory_processor._load_from_checkpoint()


def resources():
    """Memory of this process, of the env processes (its children) and of the GPU, in GB."""
    me = psutil.Process()
    children = sum(c.memory_info().rss for c in me.children(recursive=True) if c.is_running())
    return {"Main Process GB": me.memory_info().rss / 2**30, "Env Processes GB": children / 2**30,
            "GPU Allocated GB": torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
            "GPU Reserved GB": torch.cuda.memory_reserved() / 2**30 if torch.cuda.is_available() else 0.0}


class RunLogger(PPOMetricsLogger):
    """RLGym-Learn's PPO metrics plus this project's: per-component reward means (raw and weighted), touches
    per minute, the monitors and memory. Printed each iteration and appended as one JSON line per iteration to
    `path` (if given). Prints a warning when a monitor crosses its threshold."""

    def __init__(self, path=None, weights=None):
        super().__init__()
        self.path = path
        self.weights = dict(PHASE_1A if weights is None else weights)
        self.warnings = []

    def collect_agent_metrics(self, data):
        super().collect_agent_metrics(data)
        tp, ppo = data.trajectory_processor_data, data.ppo_data
        means = tp.component_means
        self.agent_metrics["Reward Components"] = {
            **{f"{n} (raw mean)": m for n, m in means.items()},
            **{f"{n} (weighted mean)": m * self.weights.get(n, 0.0) for n, m in means.items()},
        }
        if "touch" in means:  # touches per car per minute of game time
            self.agent_metrics["Reward Components"]["Touches per Minute"] = means["touch"] * STEPS_PER_SECOND * 60
        self.agent_metrics["Resources"] = resources()
        self.agent_metrics["Monitors"] = {
            "Reward Clip Share": tp.reward_clip_share,
            "Largest Scaled Reward": tp.largest_scaled_reward,
            "Largest Observation Jump": tp.largest_obs_jump,
            "Non-finite Observation Values": tp.nonfinite_obs,
            "First Minibatch Ratio": ppo.first_minibatch_ratio,
        }
        self.warnings = []
        if tp.reward_clip_share > CLIP_SHARE_WARNING:
            self.warnings.append(f"{tp.reward_clip_share:.1%} of scaled rewards hit the +-{REWARD_CLIP:g} clip")
        if tp.largest_obs_jump > JUMP_WARNING or tp.nonfinite_obs:
            self.warnings.append(f"observation jump {tp.largest_obs_jump:.3f}, {tp.nonfinite_obs} non-finite values")

    def report_metrics(self):
        super().report_metrics()
        for w in self.warnings:
            print(f"WARNING: {w}")
        if self.path is not None:
            with open(self.path, "a") as f:
                f.write(json.dumps({**self.get_metrics(), "Warnings": self.warnings}, default=float) + "\n")


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
    checkpoint trained on different observation versions will not load. They also keep the float16 gradient
    scale, which PPOLearner's checkpoints leave out.
    """

    SCALER_FILE = "grad_scaler.pt"

    def __init__(self, actor_factory, critic_factory, precision="float16", reward_ttb="v1"):
        super().__init__(actor_factory, critic_factory)
        assert precision in ("float16", "float32")
        self.precision = precision
        self.reward_ttb = reward_ttb
        self.scaler = None

    def load(self, config):
        device = config.learner_config.device
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.precision == "float16" and device.type == "cuda")
        super().load(config)

    def save_checkpoint(self, folder_path):
        super().save_checkpoint(folder_path)
        versions.write(folder_path, self.reward_ttb)
        torch.save(self.scaler.state_dict(), os.path.join(folder_path, self.SCALER_FILE))

    def _load_from_checkpoint(self):
        folder = self.config.checkpoint_load_folder
        versions.check(folder)
        super()._load_from_checkpoint()
        if os.path.exists(os.path.join(folder, self.SCALER_FILE)):
            self.scaler.load_state_dict(torch.load(os.path.join(folder, self.SCALER_FILE)))

    def learn(self, exp):
        cfg = self.config.learner_config
        device = cfg.device
        half = self.scaler.is_enabled()
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


def make_agent_controller(width=WIDTH, precision="float16", reward_ttb="v1", metrics_logger=None):
    controller = PPOAgentController(*factories(width), ExperienceBuffer(ClippedGAETrajectoryProcessor()),
                                    metrics_logger=metrics_logger)
    controller.learner = Learner(*factories(width), precision=precision, reward_ttb=reward_ttb)
    return controller


def make_config(n_proc=32, timesteps_per_iteration=100_000, buffer_size=300_000, batch_size=100_000, n_minibatches=2,
                n_epochs=1, lr=2e-4, ent_coef=0.01, gamma=0.99, timestep_limit=5_000_000_000, device="cuda:0",
                save_folder="checkpoints", run_name="run", save_every_ts=5_000_000, seed=123, inference_procs=None):
    """Defaults: 100,000 steps per iteration, a 300,000-step buffer, batches of 100,000 as two minibatches of
    50,000, 1 epoch. Every epoch covers the whole buffer, so each step is trained on 3 times. (The design
    doc's 2 epochs would make that 6.) Reuse is a training setting and can change mid-run.

    inference_procs: how many env processes must have stepped before the actor runs on their observations.
    Default: half of them, so one half steps while the other is served. Measured 20-30% more collection than
    waiting for all at 8-20 processes, either width (scripts/throughput.py). Total throughput flattens past
    about 20 processes; 32 measured best, by a few percent."""
    process = ProcessConfigModel(n_proc=n_proc, min_process_steps_per_inference=inference_procs or max(1, n_proc // 2))
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
