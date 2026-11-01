"""Evaluate T-insensitivity of trained DD_MADUN_TFree checkpoints.

The SAME weights are run on every T in --T_eval (including T never seen in
training). Each test clip is generated once at max(T_eval) frames and the first
T frames are used, so all T share the same scenes; noise is seeded per clip.

  python eval_tfree.py --ckpt output_tfree_mix/best_model.pth output_tfree_T8/best_model.pth \
                       --names mix T8only --T_eval 8,10,12,16,18,20,24,28,32,36,40

Optional: --v3_ckpt <path> also evaluates an original DD_MADUNv3 at its own T
(it cannot run other T). Architecture (nf, nb, K, K1) is inferred from the
state_dict; the encoding mask is taken from the checkpoint when saved there.
"""
import os, re, json, time, argparse
import numpy as np
import torch
from cup_simulator import CUPForwardModel
from baselines import compute_psnr, compute_ssim
from dataset import make_complex_scene
from dd_madun_v3 import DD_MADUNv3
from dd_madun_tfree import DD_MADUN_TFree, _build_exp_cdf, sample_grayscale_mask, make_mask_w


def load_ckpt(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        return ck["model_state_dict"], ck.get("mask"), ck.get("config")
    return ck, None, None


def infer_arch(sd):
    nf = sd["den.0.head.0.weight"].shape[0]
    nb = len({int(m.group(1)) for k in sd for m in [re.match(r"den\.0\.blocks\.(\d+)\.c1\.weight", k)] if m})
    K = len([k for k in sd if re.fullmatch(r"lrs\.\d+", k)])
    K2 = len([k for k in sd if re.fullmatch(r"lsg\.\d+", k)])
    return nf, nb, K, K - K2


def build_mask(saved_mask, cfg, args, H, W, dev):
    if saved_mask is not None:
        return saved_mask.to(dev).float(), "from checkpoint"
    gray = args.gray_mask or (cfg or {}).get("gray_mask", False)
    if gray:
        here = os.path.dirname(os.path.abspath(__file__))
        mp = os.path.join(here, "..", "实验数据", "7-28 10hz 3vpp", "encoding_mask_normalized.npy")
        cdf, cen, _ = _build_exp_cdf(mp)
        seed = (cfg or {}).get("gray_seed", args.gray_seed)
        return sample_grayscale_mask(H, W, dev, cdf, cen, (cfg or {}).get("gray_blur", 2.0), seed=seed), \
               f"gray, seed={seed}"
    g = torch.Generator().manual_seed(args.mask_seed)
    return torch.bernoulli(torch.full((H, W), 0.5), generator=g).to(dev), \
           f"random binary seed={args.mask_seed} (WARNING: not the training mask)"


def run_model(net, mask, T_list, test, args, dev, cfg):
    shear = (cfg or {}).get("shear_per_frame", 1)
    mdc = (cfg or {}).get("mask_dc", False) or args.mask_dc
    mw = make_mask_w(mask, (cfg or {}).get("mask_dc_thr", 0.05), (cfg or {}).get("mask_dc_soft", 0.05)) if mdc else None
    res = {}
    for T in T_list:
        cup = CUPForwardModel(H=mask.shape[0], W=mask.shape[1], T=T, shear_per_frame=shear).to(dev)
        cup.set_mask(mask)
        ps, ss, n_bad = [], [], 0
        t0 = time.time()
        with torch.no_grad():
            for i in range(test.shape[0]):
                gt = test[i:i + 1, :T].to(dev)
                E = cup.forward(gt)
                g = torch.Generator(device="cpu").manual_seed(args.noise_seed + i)
                E = E + args.noise_std * torch.randn(E.shape, generator=g).to(dev)
                Ip, _ = net(E, cup.forward, cup.adjoint, mw)
                if not torch.isfinite(Ip).all():
                    n_bad += 1; Ip = torch.nan_to_num(Ip)
                Ip = torch.clamp(Ip, 0, 1)
                ps.append(float(compute_psnr(Ip, gt))); ss.append(float(compute_ssim(Ip, gt)))
        if dev == "cuda": torch.cuda.synchronize()
        res[T] = {"psnr": float(np.mean(ps)), "psnr_std": float(np.std(ps)),
                  "ssim": float(np.mean(ss)), "nonfinite": n_bad,
                  "sec_per_clip": (time.time() - t0) / test.shape[0]}
        print(f"    T={T:2d}: PSNR={res[T]['psnr']:.2f}±{res[T]['psnr_std']:.2f}  "
              f"SSIM={res[T]['ssim']:.4f}  nonfinite={n_bad}  {res[T]['sec_per_clip']:.2f}s/clip")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True, help="DD_MADUN_TFree checkpoints")
    p.add_argument("--names", nargs="+", default=None)
    p.add_argument("--v3_ckpt", type=str, default=None, help="optional original DD_MADUNv3 checkpoint")
    p.add_argument("--T_eval", type=str, default="8,10,12,16,18,20,24,28,32,36,40")
    p.add_argument("--n_test", type=int, default=20)
    p.add_argument("--H", type=int, default=256)
    p.add_argument("--W", type=int, default=256)
    p.add_argument("--ng", type=int, default=6, help="GD steps (not stored in state_dict)")
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--noise_seed", type=int, default=1000)
    p.add_argument("--test_seed", type=int, default=777)
    p.add_argument("--mask_seed", type=int, default=0)
    p.add_argument("--gray_mask", action="store_true")
    p.add_argument("--gray_seed", type=int, default=12345)
    p.add_argument("--mask_dc", action="store_true")
    p.add_argument("--out", type=str, default="eval_tfree.json")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else
                              "  WARNING: CUDA not available, running on CPU (slow)"))
    T_eval = sorted(int(t) for t in args.T_eval.split(","))
    names = args.names or [os.path.basename(os.path.dirname(os.path.abspath(c))) for c in args.ckpt]
    assert len(names) == len(args.ckpt)

    print(f"Generating {args.n_test} test clips at T={max(T_eval)} (seed={args.test_seed}) ...")
    test = make_complex_scene(args.n_test, max(T_eval), args.H, args.W, seed=args.test_seed).float()

    all_res, masks = {}, {}
    for name, path in zip(names, args.ckpt):
        sd, smask, cfg = load_ckpt(path, dev)
        nf, nb, K, K1 = infer_arch(sd)
        ng = (cfg or {}).get("n_gd_steps", args.ng)
        net = DD_MADUN_TFree(H=args.H, W=args.W, K=K, K1=K1, ng=ng, nf=nf, nb=nb).to(dev)
        net.load_state_dict(sd); net.eval()
        mask, msrc = build_mask(smask, cfg, args, args.H, args.W, dev)
        masks[name] = mask
        trained = (cfg or {}).get("T_list")
        print(f"\n[{name}] {path}\n  nf={nf} nb={nb} K={K} K1={K1} ng={ng} | trained T={trained} | mask: {msrc}")
        all_res[name] = {"ckpt": path, "trained_T": trained, "mask": msrc,
                         "per_T": run_model(net, mask, T_eval, test, args, dev, cfg)}
        del net; torch.cuda.empty_cache() if dev == "cuda" else None

    if args.v3_ckpt:
        sd, smask, cfg = load_ckpt(args.v3_ckpt, dev)
        Tv3 = sd["dm"].shape[0]
        nf = sd["den.0.head.0.weight"].shape[0]
        nb = len({int(m.group(1)) for k in sd for m in [re.match(r"den\.0\.blocks\.(\d+)\.c1\.weight", k)] if m})
        K = len([k for k in sd if re.fullmatch(r"lrs\.\d+", k)])
        K1 = K - len([k for k in sd if re.fullmatch(r"lsg\.\d+", k)])
        net = DD_MADUNv3(T=Tv3, H=args.H, W=args.W, K=K, K1=K1, ng=args.ng, nf=nf, nb=nb).to(dev)
        net.load_state_dict(sd); net.eval()
        if smask is not None:
            mask, msrc = smask.to(dev).float(), "from checkpoint"
        else:
            mask, msrc = next(iter(masks.values())), "same as first tfree ckpt (v3 ckpt has no mask)"
        print(f"\n[v3] {args.v3_ckpt}\n  T={Tv3} nf={nf} nb={nb} | mask: {msrc}")
        all_res["v3"] = {"ckpt": args.v3_ckpt, "trained_T": [Tv3], "mask": msrc,
                         "per_T": run_model(net, mask, [Tv3], test, args, dev, cfg)}

    # ---- summary table ----
    cols = list(all_res.keys())
    print("\n" + "=" * (8 + 12 * len(cols)))
    print("PSNR (dB)   * = T seen in training")
    print("  T   " + "".join(f"{c:>12s}" for c in cols))
    for T in T_eval:
        row = f"  {T:<4d}"
        for c in cols:
            r = all_res[c]["per_T"].get(T)
            tr = all_res[c]["trained_T"] or []
            cell = "-" if r is None else "%.2f%s" % (r["psnr"], "*" if T in tr else " ")
            row += f"{cell:>12s}"
        print(row)
    print("-" * (8 + 12 * len(cols)))
    summary = {}
    for c in cols:
        pr = all_res[c]["per_T"]; tr = all_res[c]["trained_T"] or []
        seen = [pr[T]["psnr"] for T in pr if T in tr]
        unseen = [pr[T]["psnr"] for T in pr if T not in tr]
        allp = [v["psnr"] for v in pr.values()]
        summary[c] = {"mean_seen": float(np.mean(seen)) if seen else None,
                      "mean_unseen": float(np.mean(unseen)) if unseen else None,
                      "spread_max_minus_min": float(max(allp) - min(allp)),
                      "any_nonfinite": any(v["nonfinite"] > 0 for v in pr.values())}
        s = summary[c]
        print(f"  {c}: mean(seen)={s['mean_seen'] if s['mean_seen'] is None else round(s['mean_seen'],2)}  "
              f"mean(unseen)={s['mean_unseen'] if s['mean_unseen'] is None else round(s['mean_unseen'],2)}  "
              f"spread={s['spread_max_minus_min']:.2f}  nonfinite={s['any_nonfinite']}")

    out = {"T_eval": T_eval, "n_test": args.n_test, "test_seed": args.test_seed,
           "results": {c: {**v, "per_T": {str(k): x for k, x in v["per_T"].items()}} for c, v in all_res.items()},
           "summary": summary}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
