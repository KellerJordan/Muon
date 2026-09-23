"""A parameter with no gradient must not be weight-decayed. See issue 54."""
import torch
from muon import SingleDeviceMuon, SingleDeviceMuonWithAuxAdam


def test_single_device_muon_leaves_missing_grad_untouched():
    torch.manual_seed(54)
    live = torch.nn.Parameter(torch.randn(4, 4))
    frozen = torch.nn.Parameter(torch.randn(4, 4))
    opt = SingleDeviceMuon([live, frozen], lr=0.02, weight_decay=0.1, momentum=0.95)
    live_before = live.detach().clone()
    frozen_before = frozen.detach().clone()
    live.grad = torch.randn_like(live)
    opt.step()
    assert torch.equal(frozen, frozen_before)
    assert "momentum_buffer" not in opt.state[frozen]
    assert not torch.equal(live, live_before)

    buf = opt.state[live]["momentum_buffer"].clone()
    live_after = live.detach().clone()
    live.grad = None
    opt.step()
    assert torch.equal(live, live_after)
    assert torch.equal(opt.state[live]["momentum_buffer"], buf)
    assert torch.equal(frozen, frozen_before)


def test_aux_adam_leaves_missing_grad_untouched():
    torch.manual_seed(54)
    muon_p = torch.nn.Parameter(torch.randn(4, 4))
    muon_frozen = torch.nn.Parameter(torch.randn(4, 4))
    adam_p = torch.nn.Parameter(torch.randn(4))
    adam_frozen = torch.nn.Parameter(torch.randn(4))
    opt = SingleDeviceMuonWithAuxAdam([
        dict(params=[muon_p, muon_frozen], lr=0.02, momentum=0.95, weight_decay=0.1, use_muon=True),
        dict(params=[adam_p, adam_frozen], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, use_muon=False),
    ])
    frozen_before = {p: p.detach().clone() for p in (muon_frozen, adam_frozen)}
    muon_before = muon_p.detach().clone()
    adam_before = adam_p.detach().clone()
    muon_p.grad = torch.randn_like(muon_p)
    adam_p.grad = torch.randn_like(adam_p)
    opt.step()
    for p, before in frozen_before.items():
        assert torch.equal(p, before)
        assert len(opt.state[p]) == 0
    assert not torch.equal(muon_p, muon_before)
    assert not torch.equal(adam_p, adam_before)
    assert opt.state[adam_p]["step"] == 1

    adam_buf = opt.state[adam_p]["exp_avg"].clone()
    adam_before = adam_p.detach().clone()
    adam_p.grad = None
    opt.step()
    assert torch.equal(adam_p, adam_before)
    assert torch.equal(opt.state[adam_p]["exp_avg"], adam_buf)
    assert opt.state[adam_p]["step"] == 1


if __name__ == "__main__":
    test_single_device_muon_leaves_missing_grad_untouched()
    test_aux_adam_leaves_missing_grad_untouched()
    print("ok")
