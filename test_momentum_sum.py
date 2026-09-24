"""The paper momentum sum is opt-in. The shipped running average stays the default."""
import torch
from muon import muon_update


def test_default_is_the_running_average():
    torch.manual_seed(0)
    g = torch.randn(4, 4)
    beta = 0.95
    m = torch.zeros_like(g)
    muon_update(g.clone(), m, beta=beta, momentum_sum=False, ns_steps=1)
    expected = (1 - beta) * g
    assert torch.allclose(m.float(), expected.float(), atol=2e-2)
    assert not torch.allclose(m.float(), g.float(), atol=0.2)


def test_momentum_sum_matches_the_paper_recurrence():
    torch.manual_seed(1)
    g = torch.randn(4, 4)
    beta = 0.95
    m = torch.zeros_like(g)
    muon_update(g.clone(), m, beta=beta, momentum_sum=True, ns_steps=1)
    # B_1 = beta * 0 + G
    assert torch.allclose(m.float(), g.float(), atol=2e-2)
    g2 = torch.randn(4, 4)
    muon_update(g2.clone(), m, beta=beta, momentum_sum=True, ns_steps=1)
    # B_2 = beta * G + G2, within bf16
    expected = beta * g + g2
    assert torch.allclose(m.float(), expected.float(), atol=5e-2)


def test_default_flag_is_stable():
    torch.manual_seed(2)
    g = torch.randn(4, 4)
    a = muon_update(g.clone(), torch.zeros_like(g), momentum_sum=False, ns_steps=2)
    b = muon_update(g.clone(), torch.zeros_like(g), ns_steps=2)
    assert torch.equal(a, b)


if __name__ == "__main__":
    test_default_is_the_running_average()
    test_momentum_sum_matches_the_paper_recurrence()
    test_default_flag_is_stable()
    print("ok")
