"""Checkpoint shape adaptation: legacy T-bound DD-MADUN -> T-free DD-MADUN.

Lives in its own module (rather than inside dd_madun_v3_train_tfree.py) because
that training script executes its whole pipeline -- argparse, dataset building
for every T, model construction -- at module level. Importing it for a single
helper function therefore used to build ~6 GB of datasets and start training as
a side effect (convert_to_tfree.py hit exactly this and never finished).
"""
import torch


def warm_start_from_legacy(model, path, legacy_T):
    """Initialise a T-free model from a legacy T-bound checkpoint.

    head: (nf, T+1, 3, 3) -> (nf, 2, 3, 3)
          channel 0 = mean over the T image channels, channel 1 = noise map channel
    tail: (T, nf, 3, 3)   -> (1, nf, 3, 3)  mean over the T output channels (dim 0)
    Remaining weights (GRU gates, blocks, sf) have identical shapes and copy directly.
    """
    sd = torch.load(path, map_location="cpu", weights_only=False)
    old = sd.get("model_state_dict", sd)
    new = model.state_dict()
    n_img, n_tail = 0, 0
    for k in new:
        if k not in old:
            continue
        if old[k].shape == new[k].shape:
            new[k] = old[k]
        elif k.endswith("head.0.weight") and old[k].dim() == 4:
            # (nf, T+1, 3, 3) -> (nf, 2, 3, 3)
            new[k] = torch.stack([old[k][:, :legacy_T].mean(1), old[k][:, -1]], dim=1)
            n_img += 1
        elif k.endswith("tail.weight") and old[k].dim() == 4:
            # tail is (T, nf, 3, 3): the T axis is dim 0, NOT dim 1
            new[k] = old[k].mean(0, keepdim=True)
            n_tail += 1
    missing = model.load_state_dict(new, strict=False)
    print(f"  [warm start] {path}")
    print(f"    adapted head x{n_img}, tail x{n_tail}; "
          f"missing={len(getattr(missing,'missing_keys',[]))} "
          f"unexpected={len(getattr(missing,'unexpected_keys',[]))}")
    return model
