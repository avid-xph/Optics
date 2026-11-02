"""E2ECubeNet 训练（任务③ 第三类方法: End-to-end, 每个 T 一份权重）.

数据与 DD-MADUN T-free 训练一致: DAVIS 真实视频 (real_ratio) + 合成场景,
直接复用 dd_madun_tfree.create_mixed_dataset_tmax（T_max = 训练 T）。
每个 batch 重新采样随机块状 mask（--cell）, 让网络学到「任意同分布 mask」,
与消融评测时 mask 随 seed/cell 变化的设定对齐。

用法（远程 GPU, 每个 T 各跑一次）:
  python -m sci_ablation.train_e2e --T 8  --davis_root <DAVIS> --out_dir output_e2e_T8
  python -m sci_ablation.train_e2e --T 16 --davis_root <DAVIS> --out_dir output_e2e_T16
本地 CPU 冒烟（秒级, 只验证能跑通）:
  python -m sci_ablation.train_e2e --T 8 --H 32 --W 32 --n_train 8 --n_test 2 \
      --epochs 1 --davis_root none --device cpu --out_dir output_e2e_smoke

输出 <out_dir>/best_model.pth: {model_state_dict, config{T,nf,nb,...}, epoch, psnr, ssim}
评测: run_ablation --methods e2e --ckpt e2e=output_e2e_T{T}/best_model.pth
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines import compute_psnr, compute_ssim
from dd_madun_v3 import ssim_loss, edge_loss
from dd_madun_tfree import create_mixed_dataset_tmax, augment
from sci_ablation.e2e_model import E2ECubeNet, build_e2e_input
from sci_ablation.forward import ExperimentCUP, build_block_mask


def main():
    ap = argparse.ArgumentParser(description="E2ECubeNet training (one T per run)")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--cell", type=int, default=1, help="训练用编码单元大小")
    ap.add_argument("--noise_std", type=float, default=0.01)
    ap.add_argument("--nf", type=int, default=48)
    ap.add_argument("--nb", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=10)
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--shear", type=int, default=1)
    ap.add_argument("--real_ratio", type=float, default=0.7)
    ap.add_argument("--ssim_weight", type=float, default=0.2)
    ap.add_argument("--edge_weight", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--davis_root", default="DAVIS")
    ap.add_argument("--out_dir", default=None, help="默认 output_e2e_T<T>")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    T, dev = args.T, args.device
    out_dir = args.out_dir or f"output_e2e_T{T}"
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    config = dict(vars(args), out_dir=out_dir)

    print(f"[1/4] Dataset: T={T}, {args.n_train} train / {args.n_test} test, {args.H}x{args.W}")
    train_data, test_data = create_mixed_dataset_tmax(
        args.n_train, args.n_test, T, args.H, args.W, args.real_ratio,
        args.davis_root, seed=args.seed)

    # 训练前向: 每 batch 换 mask; 评测前向: 固定 seed 的 mask, 各 epoch 可比
    cup = ExperimentCUP(H=args.H, W=args.W, T=T, shear=args.shear, cell=args.cell,
                        noise_std=args.noise_std, device=dev, seed=args.seed)
    eval_cup = ExperimentCUP(H=args.H, W=args.W, T=T, shear=args.shear, cell=args.cell,
                             noise_std=args.noise_std, device=dev, seed=0)

    model = E2ECubeNet(T=T, nf=args.nf, nb=args.nb).to(dev)
    print(f"[2/4] E2ECubeNet T={T} nf={args.nf} nb={args.nb}: "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    def lr_lambda(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / args.warmup_epochs
        q = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * q))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def evaluate():
        model.eval()
        ps, ss = [], []
        g = torch.Generator().manual_seed(1000)
        with torch.no_grad():
            for i in range(args.n_test):
                gt = test_data[i:i + 1, :T].to(dev)
                E = eval_cup.forward(gt)     # 噪声用 CPU generator, 各 epoch 相同
                E = E + args.noise_std * torch.randn(E.shape, generator=g).to(dev)
                out = torch.clamp(model(build_e2e_input(eval_cup, E)), 0, 1)
                ps.append(float(compute_psnr(out, gt))); ss.append(float(compute_ssim(out, gt)))
        model.train()
        return float(np.mean(ps)), float(np.mean(ss))

    print(f"[3/4] Training on {dev}, {args.epochs} epochs")
    history, best, t0 = [], -1.0, time.time()
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(args.n_train)
        tot, nbat = 0.0, 0
        for s in range(0, args.n_train, args.batch_size):
            gt = augment(train_data[perm[s:s + args.batch_size]].float()).to(dev)
            cup.cup.set_mask(build_block_mask(args.H, args.W, args.cell, device=dev))
            E = cup.measure(gt)
            out = model(build_e2e_input(cup, E))
            loss = (F.l1_loss(out, gt) + args.ssim_weight * ssim_loss(out, gt)
                    + args.edge_weight * edge_loss(out, gt))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nbat += 1
        sched.step()

        if ep == 1 or ep % args.eval_every == 0 or ep == args.epochs:
            psnr, ssim = evaluate()
            history.append(dict(epoch=ep, loss=tot / max(1, nbat), psnr=psnr, ssim=ssim))
            print(f"  E{ep:3d}/{args.epochs} | L={tot / max(1, nbat):.4f} | "
                  f"PSNR={psnr:.2f} SSIM={ssim:.4f} | {time.time() - t0:.0f}s")
            if psnr > best:
                best = psnr
                torch.save(dict(epoch=ep, model_state_dict=model.state_dict(),
                                psnr=psnr, ssim=ssim, config=config),
                           os.path.join(out_dir, "best_model.pth"))

    print(f"[4/4] Done in {(time.time() - t0) / 3600:.2f}h, best test PSNR {best:.2f}")
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(dict(history=history, config=config, best_psnr=best), f, indent=1)


if __name__ == "__main__":
    main()
