"""CPU Newton-Schulz stays close to the bfloat16 quintic and stays finite."""
import torch
from muon import zeropower_via_newtonschulz5, SingleDeviceMuon


def bf16_quintic(G, steps):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def main():
    torch.manual_seed(0)
    g = torch.randn(256, 256)
    got = zeropower_via_newtonschulz5(g, steps=5)
    ref = bf16_quintic(g, steps=5).float()
    err = (got.float() - ref).abs().max().item()
    assert got.dtype == torch.float32
    assert torch.isfinite(got).all()
    assert err < 2e-2, err

    for device in ("cuda", "mps"):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        if device == "mps" and not torch.backends.mps.is_available():
            continue
        gm = torch.randn(64, 64, device=device)
        out = zeropower_via_newtonschulz5(gm, steps=2)
        assert out.dtype == torch.bfloat16, (device, out.dtype)
        assert torch.isfinite(out.float()).all()
        print("branch ok", device)

    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = SingleDeviceMuon([p], lr=0.02)
    before = p.detach().clone()
    p.pow(2).sum().backward()
    opt.step()
    assert torch.isfinite(p).all()
    assert not torch.equal(p, before)
    print("ok", err)


if __name__ == "__main__":
    main()
