"""What the float16 update (playstyle.training.Learner) changes compared with float32, on real observations.

  1. Log-probabilities. Collection runs in float32 and the update in float16, so the same weights give
     slightly different log-probabilities, and PPO's probability ratio is off by that much before the policy
     has moved at all. The error grows as the policy gets less uniform, so the final layer is scaled up to
     imitate later training (a stand-in: these are not trained weights).
  2. Gradients. Float16 against float32 on the same 50,000-step minibatch, next to what dropping half the
     minibatch does (ordinary sampling noise).

    python scripts/float16_check.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import cosine_similarity, mse_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from learner_cost import load_steps

from playstyle.network import EntityActor, EntityCritic
from playstyle.training import NETWORKS

N = 50_000


def main():
    pieces = load_steps(100_000, 1500)
    obs = torch.tensor(np.concatenate([p["obs"] for p in pieces])[:N]).cuda()
    actions = torch.tensor(np.concatenate([p["actions"] for p in pieces])[:N]).cuda()
    torch.manual_seed(1)
    advantages, targets = torch.randn(N).cuda(), torch.randn(N).cuda() * 0.3

    print("log-probability of the sampled action, float16 minus float32 (PPO clips the ratio at +-0.2):")
    for width in NETWORKS:
        for gain in (0.01, 1.0, 3.0, 10.0):
            torch.manual_seed(0)
            actor = EntityActor(device="cuda", **NETWORKS[width])
            torch.nn.init.orthogonal_(actor.net.out.weight, gain=gain)
            with torch.no_grad():
                exact = torch.log_softmax(actor.logits(obs), -1)
                with torch.autocast("cuda", dtype=torch.float16):
                    half = torch.log_softmax(actor.logits(obs).float(), -1)
                sampled = torch.multinomial(exact.exp(), 1)
                error = (half - exact).gather(-1, sampled)
                entropy = -(exact * exact.exp()).sum(-1).mean().item()
            print(f"  width {width}, final layer x{gain:<5g} policy entropy {entropy:.2f}:  "
                  f"typical {error.pow(2).mean().sqrt().item():.1e}, largest {error.abs().max().item():.1e}")

    print("gradient of the PPO loss on one minibatch:")
    for width in NETWORKS:
        torch.manual_seed(0)
        actor = EntityActor(device="cuda", **NETWORKS[width])
        critic = EntityCritic(device="cuda", **NETWORKS[width])
        params = list(actor.parameters()) + list(critic.parameters())
        with torch.no_grad():
            old = torch.log_softmax(actor.logits(obs), -1).gather(-1, actions).view(-1)

        def gradient(half, rows=slice(None)):
            for p in params:
                p.grad = None
            scaler = torch.amp.GradScaler("cuda", enabled=half)
            with torch.autocast("cuda", dtype=torch.float16, enabled=half):
                log_probs, entropy = actor.get_backprop_data(None, obs[rows], actions[rows])
                values = critic(None, obs[rows])
            ratio = torch.exp(log_probs.float().view(-1) - old[rows])
            loss = (-torch.min(ratio * advantages[rows], ratio.clamp(0.8, 1.2) * advantages[rows]).mean()
                    - 0.01 * entropy.float() + mse_loss(values.float().view(-1), targets[rows]))
            scaler.scale(loss).backward()
            return torch.cat([p.grad.flatten() for p in params]) / scaler.get_scale()

        full, half, fewer = gradient(False), gradient(True), gradient(False, slice(0, N // 2))
        print(f"  width {width}: float16 differs from float32 by {((half - full).norm() / full.norm()).item():.2%} "
              f"of its length (cosine {cosine_similarity(full, half, dim=0).item():.6f}); "
              f"float32 on half the minibatch differs by {((fewer - full).norm() / full.norm()).item():.1%}")


if __name__ == "__main__":
    main()
