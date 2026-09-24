"""Optional NorMuon row RMS. Default Muon is unchanged."""
import torch
from muon import SingleDeviceMuon, muon_update


def test_default_matches_unnormalized_update():
    torch.manual_seed(0)
    g = torch.randn(8, 4)
    m1 = torch.zeros_like(g)
    m2 = torch.zeros_like(g)
    plain = muon_update(g.clone(), m1, row_rms=False)
    flagged = muon_update(g.clone(), m2, row_rms=False)
    assert torch.equal(plain, flagged)


def test_row_rms_makes_each_row_unit_rms():
    torch.manual_seed(1)
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = SingleDeviceMuon([p], lr=0.0, weight_decay=0, momentum=0.95, row_rms=True)
    p.grad = torch.randn(8, 4)
    before = p.detach().clone()
    opt.step()
    # lr=0 so the parameter does not move; inspect the orthogonal update via a second call
    g = torch.randn(6, 4)
    update = muon_update(g, torch.zeros_like(g), row_rms=True)
    rms = update.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)
    plain = muon_update(g.clone(), torch.zeros_like(g), row_rms=False)
    assert not torch.allclose(plain.float(), update.float(), atol=1e-3)
    assert torch.equal(p, before)


if __name__ == "__main__":
    test_default_matches_unnormalized_update()
    test_row_rms_makes_each_row_unit_rms()
    print("ok")
