"""Recover the DD-MADUN v3 architecture from a saved checkpoint.

Motivation: benchmark_reconstruct.py / convert_to_tfree.py used to hard-code
K=8, K1=3, ng=6, nf=96, nb=8. Any run trained with different --K/--K1/--nf/--nb
(e.g. the nf=32 nb=2 smoke tests) then died at load_state_dict with a shape
mismatch, or -- worse -- needed the constants edited by hand every time.

Resolution order for each key (later wins):
  1. defaults below
  2. ckpt["cfg"]            (written by dd_madun_v3_train_tfree.py)
  3. flat top-level keys    (written by convert_to_tfree.py)
  4. INFERRED from the state_dict itself

Step 4 matters because shape-critical keys (K, K1, nf, nb) must match the
stored tensors, and plenty of checkpoints in the wild -- e.g. the P4 legacy
checkpoint, which only stores epoch/model_state_dict/ema_psnr -- carry no cfg
at all. For those we read the shapes directly:
    K   = number of lrs.* parameters
    K2  = number of lsg.* parameters   ->  K1 = K - K2
    nf  = den.0.head.0.weight.shape[0]
    nb  = number of den.0.blocks.<i>.*
    legacy_T = den.0.head.0.weight.shape[1] - 1   (T-bound head is (nf, T+1, 3, 3);
              a T-free head is (nf, 2, 3, 3), so cin > 2 means legacy)
ng / temporal are NOT inferable (they do not change any tensor shape), so they
always come from cfg or the defaults.
"""
import torch

DEFAULT_CFG = dict(K=8, K1=3, ng=6, nf=96, nb=8, temporal="none", T=8, H=256, W=256)
_ARCH_KEYS = ("K", "K1", "ng", "nf", "nb", "temporal")


def _state_dict_of(ckpt):
    """Return the state dict inside `ckpt`, or None if it isn't a checkpoint."""
    if not isinstance(ckpt, dict) or not ckpt:
        return None
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        return sd if isinstance(sd, dict) else None
    # a bare state_dict: every value is a tensor
    if all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt
    return None


def infer_cfg_from_state_dict(sd):
    """Recover architecture keys from tensor shapes. See module docstring."""
    out = {}
    if not sd:
        return out

    def count(prefix):
        return sum(1 for k in sd if k.startswith(prefix + "."))

    K = count("lrs")
    K2 = count("lsg")
    if K:
        out["K"] = K
        if K2:
            out["K1"] = max(0, K - K2)

    hw = sd.get("den.0.head.0.weight", None)
    if hw is not None and getattr(hw, "dim", lambda: 0)() == 4:
        out["nf"] = int(hw.shape[0])
        cin = int(hw.shape[1])
        # T-bound head: (nf, T+1, 3, 3). T-free head: (nf, 2, 3, 3).
        out["legacy_T"] = cin - 1 if cin > 2 else None

    nb = len({k.split(".")[3] for k in sd if k.startswith("den.0.blocks.")})
    if nb:
        out["nb"] = nb
    return out


def resolve_model_cfg(ckpt, overrides=None):
    """Best-effort architecture dict for building a model that can load `ckpt`.

    Args:
        ckpt: a checkpoint dict, or a bare state_dict.
        overrides: explicit values (e.g. from CLI flags) that beat everything;
                   entries set to None are ignored so callers can pass
                   argparse defaults straight through.
    """
    cfg = dict(DEFAULT_CFG)
    if isinstance(ckpt, dict):
        c = ckpt.get("cfg", None)
        if isinstance(c, dict):
            cfg.update({k: v for k, v in c.items() if v is not None})
        for k in ("K", "K1", "ng", "nf", "nb", "temporal", "T", "H", "W"):
            v = ckpt.get(k, None)
            if v is not None:
                cfg[k] = v
    # Shapes win over metadata: a stale/wrong cfg would just fail at load time.
    cfg.update(infer_cfg_from_state_dict(_state_dict_of(ckpt)))
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def unresolved_fields(ckpt):
    """Keys that could NOT be recovered and therefore fell back to defaults.

    ng (DC gradient steps) and temporal ('none' vs 'gru') change runtime
    behaviour but no tensor shape, so they are impossible to infer -- if the
    checkpoint does not record them, the caller should say so out loud rather
    than silently running the wrong variant.
    """
    known = set()
    if isinstance(ckpt, dict):
        c = ckpt.get("cfg", None)
        if isinstance(c, dict):
            known |= {k for k, v in c.items() if v is not None}
        known |= {k for k in ("K", "K1", "ng", "nf", "nb", "temporal", "T", "H", "W")
                  if ckpt.get(k) is not None}
    return [k for k in ("ng", "temporal") if k not in known]


def describe_mismatch(model, sd):
    """Human-readable list of state_dict vs model mismatches (empty == clean)."""
    ms = model.state_dict()
    problems = []
    for k, v in ms.items():
        if k not in sd:
            problems.append(f"missing from checkpoint: {k} {tuple(v.shape)}")
        elif tuple(sd[k].shape) != tuple(v.shape):
            problems.append(f"shape mismatch: {k} ckpt={tuple(sd[k].shape)} "
                            f"model={tuple(v.shape)}")
    return problems
