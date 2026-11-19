"""Attention actor and critic for the entity-v1 observation (playstyle/obs.py).

The observation is cut into six tokens: the four car blocks (self, teammate, two opponents), the ball and the
game state. One shared encoder reads every car block. Two pre-norm self-attention layers mix the tokens; the
head reads the observer's own token and the mean of all six.

  token kinds   a learned vector per kind (self, teammate, opponent, ball, game) is added to each token. Both
                opponents get the same one and nothing else tells them apart, so swapping the two opponent
                blocks cannot change the output.
  profile       the teammate profile (last 6 values of the observation) is not part of the game token. It goes
                through its own linear layer and is added to the teammate token, after the car encoder.
  no dropout    PPO compares action probabilities between collection and update; dropout would add noise to both.
  policy head   the last layer starts near zero, so the first policy is close to uniform over the 90 actions.

Actor and critic are separate networks of the same shape (EntityActor, EntityCritic), in the form
rlgym_learn_algos' PPO expects.
"""

from collections.abc import Sequence

import numpy as np
import torch
from rlgym_learn_algos.ppo.actor import Actor
from rlgym_learn_algos.ppo.critic import Critic
from torch import nn
from torch.nn import functional as F

from . import profile_v1
from .obs import BALL_DIM, CAR_DIM, GLOBAL_DIM, OBS_DIM

N_CARS = 4
PROFILE_DIM = len(profile_v1.FEATURES)
GAME_DIM = GLOBAL_DIM - PROFILE_DIM
SELF, TEAMMATE, OPPONENT, BALL, GAME = range(5)
TOKEN_KINDS = (SELF, TEAMMATE, OPPONENT, OPPONENT, BALL, GAME)
N_ACTIONS = 90  # LookupTableAction


def _encoder(n_in, d_model):
    return nn.Sequential(nn.Linear(n_in, d_model), nn.ReLU(), nn.Linear(d_model, d_model))


class SelfAttention(nn.Module):
    """Multi-head self-attention over [batch, tokens, d_model]. Written out rather than nn.MultiheadAttention,
    which takes a different code path in eval mode and so gives slightly different numbers than in training."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        return self.out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d))


class AttentionBlock(nn.Module):
    """Pre-norm: normalize, attend (or feed forward), add back. No dropout."""

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = SelfAttention(d_model, n_heads)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = x + self.attention(self.norm_attention(x))
        return x + self.ff(self.norm_ff(x))


class EntityTrunk(nn.Module):
    """Observation [batch, OBS_DIM] -> features [batch, 2 * d_model]."""

    def __init__(self, d_model=128, n_heads=4, n_layers=2, d_ff=256):
        super().__init__()
        self.car = _encoder(CAR_DIM, d_model)
        self.ball = _encoder(BALL_DIM, d_model)
        self.game = _encoder(GAME_DIM, d_model)
        self.profile = nn.Linear(PROFILE_DIM, d_model)
        self.kind = nn.Parameter(torch.randn(5, d_model) * 0.02)
        self.register_buffer("kind_of_token", torch.tensor(TOKEN_KINDS), persistent=False)
        self.blocks = nn.ModuleList(AttentionBlock(d_model, n_heads, d_ff) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.out_features = 2 * d_model

    def tokens(self, obs):
        """The six tokens before attention: [batch, 6, d_model]."""
        cars, ball, game, profile = obs.split((N_CARS * CAR_DIM, BALL_DIM, GAME_DIM, PROFILE_DIM), dim=1)
        tokens = torch.cat([self.car(cars.unflatten(1, (N_CARS, CAR_DIM))),
                            self.ball(ball).unsqueeze(1), self.game(game).unsqueeze(1)], dim=1)
        teammate_only = F.pad(self.profile(profile).unsqueeze(1), (0, 0, TEAMMATE, len(TOKEN_KINDS) - 1 - TEAMMATE))
        return tokens + self.kind[self.kind_of_token] + teammate_only

    def forward(self, obs):
        x = self.tokens(obs)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return torch.cat([x[:, SELF], x.mean(dim=1)], dim=1)


class EntityNet(nn.Module):
    """Trunk, hidden layers, and a linear output."""

    def __init__(self, n_out, hidden=(512, 512), final_gain=None, **trunk):
        super().__init__()
        self.trunk = EntityTrunk(**trunk)
        layers, n = [], self.trunk.out_features
        for size in hidden:
            layers += [nn.Linear(n, size), nn.ReLU()]
            n = size
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(n, n_out)
        if final_gain is not None:
            nn.init.orthogonal_(self.out.weight, gain=final_gain)
            nn.init.zeros_(self.out.bias)

    def forward(self, obs):
        return self.out(self.hidden(self.trunk(obs)))


def _as_tensor(obs_list, device):
    if isinstance(obs_list, torch.Tensor):
        return obs_list
    return torch.as_tensor(np.asarray(obs_list), dtype=torch.float32, device=device)


def sample(log_probs):
    """One action per row, drawn from the categorical distribution (Gumbel-max), with its log-probability.
    Uses only fixed-shape operations, so it can run inside a CUDA graph."""
    noise = -torch.log(-torch.log(torch.rand_like(log_probs).clamp_(1e-20, 1.0 - 1e-7)))
    action = (log_probs + noise).argmax(dim=-1, keepdim=True)
    return action, log_probs.gather(-1, action).squeeze(-1)


class GraphedSampler:
    """Collection-time sampling through CUDA graphs. At this network size each GPU call costs about the same
    whatever its batch size (kernel launches dominate), so each batch is padded to the next fixed size and
    replayed as one captured graph. The graphs read the actor's weights in place, so they stay valid as the
    optimizer updates them. Only actions and log-probabilities come back to the CPU, through pinned memory."""

    SIZES = (16, 32, 48, 64, 96, 128, 192, 256)

    def __init__(self, actor):
        self.actor = actor
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs = {}
        self.obs_in = torch.empty((self.SIZES[-1], OBS_DIM), dtype=torch.float32, pin_memory=True)
        self.actions_out = torch.empty((self.SIZES[-1], 1), dtype=torch.int64, pin_memory=True)
        self.log_probs_out = torch.empty(self.SIZES[-1], dtype=torch.float32, pin_memory=True)

    def _forward(self, obs):
        return sample(F.log_softmax(self.actor.logits(obs), dim=-1))

    def _capture(self, size):
        obs = torch.zeros((size, OBS_DIM), device=self.actor.device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(side):
            for _ in range(3):
                self._forward(obs)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        # thread_local: CUDA calls from other threads (the overlapped update) do not invalidate the capture.
        with torch.no_grad(), torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
            action, log_prob = self._forward(obs)
        return graph, obs, action, log_prob

    def capture_all(self):
        """Capture every size now. Lazily, a size first seen while the update runs on another thread is captured
        concurrently with it, which failed a resumed run (cudaErrorStreamCaptureInvalidated)."""
        for size in self.SIZES:
            if size not in self.graphs:
                self.graphs[size] = self._capture(size)

    def __call__(self, obs_list):
        n = len(obs_list)
        if n > self.SIZES[-1]:
            parts = [self(obs_list[i:i + self.SIZES[-1]]) for i in range(0, n, self.SIZES[-1])]
            return np.concatenate([a for a, _ in parts]), torch.cat([lp for _, lp in parts])
        size = next(s for s in self.SIZES if s >= n)
        if size not in self.graphs:
            self.graphs[size] = self._capture(size)
        graph, obs, action, log_prob = self.graphs[size]
        np.stack(obs_list, out=self.obs_in.numpy()[:n])
        obs[:n].copy_(self.obs_in[:n], non_blocking=True)
        graph.replay()
        self.actions_out[:n].copy_(action[:n], non_blocking=True)
        self.log_probs_out[:n].copy_(log_prob[:n], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return self.actions_out[:n].numpy().copy(), self.log_probs_out[:n].clone()


class EntityActor(Actor):
    def __init__(self, device="cpu", graphs=True, **net):
        super().__init__()
        self.device = torch.device(device)
        self.net = EntityNet(N_ACTIONS, final_gain=0.01, **net).to(self.device)
        self.sampler = GraphedSampler(self) if graphs and self.device.type == "cuda" else None

    def logits(self, obs_list):
        return self.net(_as_tensor(obs_list, self.device))

    def get_action(self, agent_id_list, obs_list, **kwargs):
        if self.sampler is not None and not kwargs.get("deterministic"):
            return self.sampler(obs_list)
        log_probs = F.log_softmax(self.logits(obs_list), dim=-1)
        if kwargs.get("deterministic"):
            action = log_probs.argmax(dim=-1, keepdim=True)
            return action.cpu().numpy(), torch.zeros(len(action))
        action, log_prob = sample(log_probs)
        return action.cpu().numpy(), log_prob.cpu()

    def get_backprop_data(self, agent_id_list, obs_list, action_list, **kwargs):
        log_probs = F.log_softmax(self.logits(obs_list), dim=-1)
        actions = action_list if isinstance(action_list, torch.Tensor) else \
            torch.as_tensor(np.asarray(action_list), device=self.device)
        entropy = -(log_probs * log_probs.exp()).sum(dim=-1)
        return log_probs.gather(-1, actions.long().view(-1, 1)), entropy.mean()


class EntityCritic(Critic):
    def __init__(self, device="cpu", **net):
        super().__init__()
        self.device = torch.device(device)
        self.net = EntityNet(1, **net).to(self.device)

    def forward(self, agent_id_list: Sequence, obs_list):
        return self.net(_as_tensor(obs_list, self.device))


assert N_CARS * CAR_DIM + BALL_DIM + GAME_DIM + PROFILE_DIM == OBS_DIM
