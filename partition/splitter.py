"""Golden Partition Zone splitter.

Splits a model into three stages along the EnclaveSplit boundary:

    head     — non-sensitive early layers           (untrusted normal world)
    enclave  — the Golden Partition Zone block       (attested enclave)
    tail     — any remaining non-sensitive layers    (untrusted normal world)

Only `enclave` ever sees cleartext weights for the decision-level layers. The
reference implements the torchvision ResNet-18 split concretely (layer4.1 +
avgpool + fc inside the enclave, matching partition/configs/resnet18.yaml);
other models follow the same pattern using the modules named in their config.

Runs on any CPU — no SGX/TrustZone required — so you can inspect and reproduce
the *partitioning method*. On real hardware, the `enclave` stage is the code
that the Open Enclave SDK loads into the secure world (see ../enclave).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import yaml

try:
    import torch
    from torch import nn
except ImportError:  # keep the module importable without torch for `--help`
    torch = None
    nn = object  # type: ignore


@dataclass
class SplitModel:
    """A model partitioned into head / enclave / tail callables."""
    name: str
    head: Callable          # runs in the normal world
    enclave: Callable       # runs in the (attested) enclave / secure stub
    tail: Callable          # runs in the normal world
    enclave_modules: list[str]

    def forward_native(self, x):
        """Reference forward with every stage in-process (the 'native' baseline)."""
        return self.tail(self.enclave(self.head(x)))


# ── Per-model builders ───────────────────────────────────────────────────────
# Each builder returns a SplitModel. Add new models here following the same
# pattern; the split boundary must match the model's YAML config.

def _build_resnet18(cfg: dict) -> "SplitModel":
    from torchvision import models

    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()

    def head(x):
        x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
        x = m.layer1(x); x = m.layer2(x); x = m.layer3(x)
        x = m.layer4[0](x)          # layer4.0 stays in the normal world
        return x

    def enclave(z):
        # Golden Partition Zone: layer4.1 + avgpool + fc (decision-level block).
        z = m.layer4[1](z)
        z = m.avgpool(z)
        z = torch.flatten(z, 1)
        return m.fc(z)

    def tail(z):
        return z                    # prediction comes straight out of the GPZ block

    return SplitModel("resnet18", head, enclave, tail, cfg["split"]["enclave_modules"])


def _build_mobilenetv2(cfg: dict) -> "SplitModel":
    from torchvision import models

    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).eval()
    boundary = 18                   # final 1x1 conv block index in m.features

    def head(x):
        for i in range(boundary):
            x = m.features[i](x)
        return x

    def enclave(z):
        z = m.features[boundary](z) # final 1x1 conv (GPZ)
        z = nn.functional.adaptive_avg_pool2d(z, 1)
        z = torch.flatten(z, 1)
        return m.classifier(z)

    def tail(z):
        return z

    return SplitModel("mobilenetv2", head, enclave, tail, cfg["split"]["enclave_modules"])


_BUILDERS = {
    "resnet18": _build_resnet18,
    "mobilenetv2": _build_mobilenetv2,
    # yolov8n / deeplabv3_mnetv3 / tinybert: same pattern using ultralytics /
    # torchvision.segmentation / transformers respectively — see their configs.
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build(config_path: str) -> "SplitModel":
    cfg = load_config(config_path)
    model = cfg["model"]
    if model not in _BUILDERS:
        raise NotImplementedError(
            f"The runnable reference implements {sorted(_BUILDERS)}. "
            f"'{model}' uses the same split pattern at "
            f"'{cfg['split']['enclave_from']}' (see {config_path}); wire up its "
            f"library (ultralytics / torchvision.segmentation / transformers) to run it."
        )
    return _BUILDERS[model](cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect / apply a Golden Partition Zone split.")
    ap.add_argument("--config", required=True, help="path to a partition/configs/*.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"Model:                 {cfg['model']}  ({cfg['task']})")
    print(f"Golden Partition Zone: {cfg['golden_partition_zone']}")
    print(f"Enclave-resident:      {cfg['split']['enclave_modules']}")
    print(f"Paper overhead:        {cfg.get('paper_overhead_pct', '—')}%  (zero accuracy loss)")

    if torch is None:
        print("\n[torch not installed — `pip install -r requirements.txt` to build the split model]")
        return

    sm = build(args.config)
    x = torch.randn(*cfg["input_shape"])
    z = sm.head(x)
    y = sm.tail(sm.enclave(z))
    print(f"\nHead output (sent to enclave): tuple{tuple(z.shape)}")
    print(f"Final output:                  tuple{tuple(y.shape)}")
    print("✓ split forward pass OK — only the GPZ block would run inside the enclave.")


if __name__ == "__main__":
    main()
