"""Convert the legacy T-bound checkpoint (checkpoints/best_model.pth) into a
T-free checkpoint so that benchmark_reconstruct.py (which expects the T-free
model) can load and run it.

Usage:
    python convert_to_tfree.py
        -> writes checkpoints/best_model_tfree.pth
"""
import sys, os
import torch

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)

from dd_madun_v3_tfree import DD_MADUNv3
# NOTE: do NOT import this from dd_madun_v3_train_tfree -- that script runs its
# entire pipeline (argparse + dataset building for every T + model construction)
# at import time, so importing it here used to build several GB of training data
# and never reach the conversion below. It now lives in warm_start.py.
from warm_start import warm_start_from_legacy
from ckpt_utils import resolve_model_cfg

SRC = os.path.join(_BASE, "checkpoints", "best_model.pth")
DST = os.path.join(_BASE, "checkpoints", "best_model_tfree.pth")

print(f"src: {SRC}")
print(f"dst: {DST}")

# The architecture is taken from the source checkpoint (cfg -> flat keys ->
# shapes inferred from the weights) instead of hard-coded, so converting a run
# trained with e.g. --nf 32 --nb 2 --K 6 works without editing this file.
src_ckpt = torch.load(SRC, map_location="cpu", weights_only=False)
cfg = resolve_model_cfg(src_ckpt)
legacy_T = cfg.get("legacy_T") or 8
print(f"  arch from checkpoint: K={cfg['K']} K1={cfg['K1']} ng={cfg['ng']} "
      f"nf={cfg['nf']} nb={cfg['nb']} (legacy T={legacy_T})")

model = DD_MADUNv3(T=legacy_T, H=cfg["H"], W=cfg["W"], K=cfg["K"], K1=cfg["K1"],
                   ng=cfg["ng"], nf=cfg["nf"], nb=cfg["nb"],
                   temporal=cfg["temporal"])
model = warm_start_from_legacy(model, SRC, legacy_T=legacy_T)

nparam = sum(p.numel() for p in model.parameters())
print(f"params: {nparam:,}")

torch.save(dict(model_state_dict=model.state_dict(),
                temporal=cfg["temporal"],
                T=legacy_T, H=cfg["H"], W=cfg["W"],
                K=cfg["K"], K1=cfg["K1"], ng=cfg["ng"],
                nf=cfg["nf"], nb=cfg["nb"],
                cfg=dict(K=cfg["K"], K1=cfg["K1"], ng=cfg["ng"],
                         nf=cfg["nf"], nb=cfg["nb"], temporal=cfg["temporal"],
                         T=legacy_T, H=cfg["H"], W=cfg["W"])), DST)
print(f"saved -> {DST}")
