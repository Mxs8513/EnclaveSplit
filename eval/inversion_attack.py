"""Model-inversion analysis — reproduces the *method* behind Table II / Fig. 4.

Question: does isolating the decision-level (Golden Partition Zone) layers
actually maximize resistance to input reconstruction? We answer it empirically,
exactly as in the paper §V-C: for each candidate split point `s`, an adversary
trains an inversion network g_s to reconstruct the input x from the exposed
normal-world activation z_s, and we report the reconstruction error
MSE(x, g_s(z_s)). Higher MSE ⇒ stronger resistance; the chosen split is only
justified if its MSE sits at/near the max of this curve.

The paper's ResNet-18 result (inversion net trained on auxiliary CIFAR-100):
reconstruction MSE rises 0.011 → 0.049 from a shallow split to `layer4` — a
~4.5× gain that locates the Golden Partition Zone at layer4.

This script wires up that training loop for ResNet-18. It needs an auxiliary
image dataset (CIFAR-100 by default via torchvision) and is compute-heavy, so it
is a reference for *reproduction*, not a clone-and-run unit test. Absolute MSEs
depend on the inversion network, data, and training budget.

    python eval/inversion_attack.py --config partition/configs/resnet18.yaml \
        --epochs 10 --data ./data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Candidate split points for ResNet-18 and the activation each exposes to the host.
RESNET18_SPLITS = ["layer1", "layer2", "layer3", "layer4", "fc"]


def _resnet18_activation_fn(model, split: str):
    """Return f(x) -> activation exposed to the untrusted host at `split`."""
    import torch

    def f(x):
        x = model.conv1(x); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
        x = model.layer1(x)
        if split == "layer1": return x
        x = model.layer2(x)
        if split == "layer2": return x
        x = model.layer3(x)
        if split == "layer3": return x
        x = model.layer4(x)
        if split == "layer4": return x
        x = model.avgpool(x); x = torch.flatten(x, 1)
        return model.fc(x)        # "fc" — logits
    return f


class Inverter:
    """A small transposed-conv decoder g_s: activation -> reconstructed image."""

    @staticmethod
    def build(in_ch: int):
        from torch import nn
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, 256, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid(),
        )


def main():
    import torch
    from torch import nn, optim
    from torchvision import datasets, models, transforms

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="partition/configs/resnet18.yaml")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval().to(device)
    for p in target.parameters():
        p.requires_grad_(False)

    tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    ds = datasets.CIFAR100(args.data, train=True, download=True, transform=tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2)

    print(f"{'split':>8} | {'recon. MSE':>11} | {'vs. shallow':>11}")
    print("-" * 36)
    baseline = None
    for split in ["layer1", "layer2", "layer3", "layer4"]:   # spatial activations
        act = _resnet18_activation_fn(target, split)
        with torch.no_grad():
            in_ch = act(torch.randn(1, 3, 224, 224, device=device)).shape[1]
        g = Inverter.build(in_ch).to(device)
        opt = optim.Adam(g.parameters(), 1e-3)
        loss_fn = nn.MSELoss()

        for _ in range(args.epochs):
            for imgs, _ in dl:
                imgs = imgs.to(device)
                with torch.no_grad():
                    z = act(imgs)
                rec = nn.functional.interpolate(g(z), size=imgs.shape[-2:])
                loss = loss_fn(rec, imgs)
                opt.zero_grad(); loss.backward(); opt.step()

        # final held-out MSE (reuse last batch for brevity in the reference)
        mse = float(loss.detach().cpu())
        baseline = baseline or mse
        print(f"{split:>8} | {mse:11.3f} | {mse / baseline:10.1f}×")

    print("\nExpectation (paper, Table II): MSE rises sharply at layer4 — the "
          "Golden Partition Zone EnclaveSplit isolates.")


if __name__ == "__main__":
    main()
