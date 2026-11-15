import lzma
import math
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from playstyle.network import (GAME_DIM, N_ACTIONS, OPPONENT, PROFILE_DIM, SELF, TEAMMATE, TOKEN_KINDS, EntityActor,
                               EntityCritic, sample)
from playstyle.obs import BALL_DIM, CAR_DIM, OBS_DIM

REF = Path(__file__).parent / "fixtures" / "obs_v1_reference.pkl.xz"
CARS = 4 * CAR_DIM
BLOCK = {name: slice(i * CAR_DIM, (i + 1) * CAR_DIM) for i, name in enumerate(("self", "mate", "opp0", "opp1"))}
PROFILE = slice(OBS_DIM - PROFILE_DIM, OBS_DIM)


@pytest.fixture(scope="module")
def obs():
    """Real observations: every 10th recorded step and all hand-built states, four observers each."""
    with lzma.open(REF, "rb") as f:
        ref = pickle.load(f)
    rows = ref["rows"][::10] + ref["hand_rows"]
    return torch.tensor(np.array([o for r in rows for o in r["obs"].values()]))


@pytest.fixture(scope="module")
def nets():
    """Actor and critic with the small policy output scaled up, so comparisons aren't hidden by tiny logits."""
    torch.manual_seed(0)
    actor, critic = EntityActor(), EntityCritic()
    with torch.no_grad():
        actor.net.out.weight *= 100
    return actor.eval(), critic.eval()


def swapped(obs, a, b):
    out = obs.clone()
    out[:, BLOCK[a]], out[:, BLOCK[b]] = obs[:, BLOCK[b]], obs[:, BLOCK[a]]
    return out


def test_output_shapes(obs, nets):
    actor, critic = nets
    with torch.no_grad():
        assert actor.logits(obs).shape == (len(obs), N_ACTIONS)
        assert critic(None, obs).shape == (len(obs), 1)
        assert actor.net.trunk.tokens(obs).shape == (len(obs), len(TOKEN_KINDS), 128)


def test_swapping_the_opponents_changes_nothing(obs, nets):
    actor, critic = nets
    other = swapped(obs, "opp0", "opp1")
    assert not torch.equal(other, obs)
    with torch.no_grad():
        logits, value = actor.logits(obs), critic(None, obs)
        assert logits.abs().max() > 0.5  # the comparison below is not between near-zero numbers
        torch.testing.assert_close(actor.logits(other), logits, atol=1e-5, rtol=0)
        torch.testing.assert_close(critic(None, other), value, atol=1e-5, rtol=0)


@pytest.mark.parametrize("a,b", [("self", "mate"), ("mate", "opp0"), ("self", "opp1")])
def test_other_swaps_do_change_the_output(obs, nets, a, b):
    actor, critic = nets
    other = swapped(obs, a, b)
    with torch.no_grad():
        assert (actor.logits(other) - actor.logits(obs)).abs().max() > 1e-2
        assert (critic(None, other) - critic(None, obs)).abs().max() > 1e-3


def test_profile_is_added_to_the_teammate_token_only(obs, nets):
    trunk = nets[0].net.trunk
    changed = obs.clone()
    changed[:, PROFILE] = torch.rand(len(obs), PROFILE_DIM)
    with torch.no_grad():
        before, after = trunk.tokens(obs), trunk.tokens(changed)
        delta = after - before
        others = [i for i in range(len(TOKEN_KINDS)) if i != TEAMMATE]
        assert torch.equal(after[:, others], before[:, others])
        expected = trunk.profile(changed[:, PROFILE]) - trunk.profile(obs[:, PROFILE])
        torch.testing.assert_close(delta[:, TEAMMATE], expected, atol=1e-5, rtol=0)
        # ...and the token itself is car encoding + kind + profile, not a concatenation
        token = trunk.car(obs[:, BLOCK["mate"]]) + trunk.kind[TEAMMATE] + trunk.profile(obs[:, PROFILE])
        torch.testing.assert_close(before[:, TEAMMATE], token, atol=1e-5, rtol=0)
        assert (nets[0].logits(changed) - nets[0].logits(obs)).abs().max() > 1e-3  # and it reaches the output


def test_game_token_reads_the_global_block_without_the_profile(obs, nets):
    trunk = nets[0].net.trunk
    assert trunk.game[0].in_features == GAME_DIM == 46
    changed = obs.clone()
    changed[:, CARS + BALL_DIM:OBS_DIM - PROFILE_DIM] += 0.5
    with torch.no_grad():
        moved = (trunk.tokens(changed) - trunk.tokens(obs)).abs().amax(dim=(0, 2))
    assert moved[-1] > 0 and torch.all(moved[:-1] == 0)


def test_one_car_encoder_and_one_kind_for_both_opponents(nets):
    trunk = nets[0].net.trunk
    assert TOKEN_KINDS[2] == TOKEN_KINDS[3] == OPPONENT and TOKEN_KINDS[0] == SELF
    assert trunk.car[0].in_features == CAR_DIM and trunk.kind.shape[0] == 5
    assert len(trunk.blocks) == 2


def test_no_dropout_and_pre_norm(obs, nets):
    actor, critic = nets
    for net in (actor, critic):
        assert not any(isinstance(m, (nn.Dropout, nn.MultiheadAttention)) for m in net.modules())
    with torch.no_grad():
        expected = actor.logits(obs)
        actor.train()
        torch.testing.assert_close(actor.logits(obs), expected, atol=0, rtol=0)  # training mode adds no noise
        actor.eval()
        # Pre-norm: the residual path is untouched, so with attention and feed-forward silenced a block is the identity.
        block = EntityActor().net.trunk.blocks[0]
        for layer in (block.attention.out, block.ff[-1]):
            nn.init.zeros_(layer.weight), nn.init.zeros_(layer.bias)
        x = torch.randn(3, 6, 128)
        assert torch.equal(block(x), x)


def test_first_policy_is_nearly_uniform(obs):
    torch.manual_seed(1)
    actor = EntityActor()
    with torch.no_grad():
        logits = actor.logits(obs)
        _, entropy = actor.get_backprop_data(None, obs, torch.zeros(len(obs), 1, dtype=torch.long))
    assert logits.abs().max() < 0.1
    assert entropy.item() == pytest.approx(math.log(N_ACTIONS), abs=1e-3)


def test_actor_interface(obs, nets):
    actor, _ = nets
    torch.manual_seed(2)
    with torch.no_grad():
        actions, log_probs = actor.get_action(None, [o.numpy() for o in obs[:32]])
        assert actions.shape == (32, 1) and actions.min() >= 0 and actions.max() < N_ACTIONS
        assert log_probs.shape == (32,) and log_probs.dtype == torch.float32
        again, entropy = actor.get_backprop_data(None, obs[:32], torch.as_tensor(actions))
        torch.testing.assert_close(again.view(-1), log_probs, atol=1e-6, rtol=0)
        assert again.shape == (32, 1) and entropy.dim() == 0
        best, _ = actor.get_action(None, obs[:32], deterministic=True)
        assert np.array_equal(best[:, 0], actor.logits(obs[:32]).argmax(dim=-1).numpy())


def test_sampling_follows_the_policy():
    torch.manual_seed(4)
    log_probs = torch.log_softmax(torch.randn(1, N_ACTIONS) * 2, dim=-1).expand(200_000, N_ACTIONS)
    actions, chosen = sample(log_probs)
    freq = torch.bincount(actions.view(-1), minlength=N_ACTIONS) / len(actions)
    assert (freq - log_probs[0].exp()).abs().max() < 0.005
    torch.testing.assert_close(chosen, log_probs.gather(-1, actions).view(-1))


def graph_vs_eager(width_kw, obs, sizes):
    torch.manual_seed(5)
    graphed = EntityActor(device="cuda", **width_kw)
    eager = EntityActor(device="cuda", graphs=False, **width_kw)
    eager.load_state_dict(graphed.state_dict())
    with torch.no_grad():
        graphed.net.out.weight *= 100  # sharper policy, so log-probabilities differ between actions
        eager.net.out.weight *= 100
    worst = 0.0
    with torch.no_grad():
        for n in sizes:
            actions, log_probs = graphed.get_action(None, [o.numpy() for o in obs[:n]])
            assert actions.shape == (n, 1) and log_probs.shape == (n,)
            exact = torch.log_softmax(eager.logits(obs[:n].cuda()), -1).gather(-1, torch.as_tensor(actions).cuda())
            worst = max(worst, (log_probs - exact.view(-1).cpu()).abs().max().item())
    return graphed, eager, worst


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("width_kw", [dict(), dict(d_model=256, d_ff=512)])
def test_graphed_sampling_matches_eager(obs, width_kw):
    sizes = (1, 12, 33, 80, 256, 300)  # padded to several graph sizes; 300 is split across two calls
    graphed, eager, worst = graph_vs_eager(width_kw, obs, sizes)
    assert worst < 1e-6

    # The graphs are captured by now; an optimizer step changes the weights in place and they must see it.
    opt = torch.optim.Adam(list(graphed.parameters()), lr=1e-2)
    loss, _ = graphed.get_backprop_data(None, obs[:64].cuda(), torch.zeros(64, 1, dtype=torch.long, device="cuda"))
    loss.mean().backward()
    opt.step()
    eager.load_state_dict(graphed.state_dict())
    with torch.no_grad():
        actions, log_probs = graphed.get_action(None, [o.numpy() for o in obs[:80]])
        exact = torch.log_softmax(eager.logits(obs[:80].cuda()), -1).gather(-1, torch.as_tensor(actions).cuda())
    assert (log_probs - exact.view(-1).cpu()).abs().max() < 1e-6


def test_every_parameter_gets_a_gradient(obs):
    torch.manual_seed(3)
    actor, critic = EntityActor(), EntityCritic()
    log_probs, entropy = actor.get_backprop_data(None, obs[:64], torch.randint(0, N_ACTIONS, (64, 1)))
    (log_probs.mean() + entropy + critic(None, obs[:64]).mean()).backward()
    for net in (actor, critic):
        for name, p in net.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, name
