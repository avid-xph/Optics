"""Resize every video folder of JPEG frames under SRC to height 480, written to DST/<prefix>_<video>/.

  python data_prep/resize_to_480p.py <src_root> <dst_480p_dir> <prefix>

A "video folder" is any directory that directly contains .jpg files (archive layouts differ).
Uses half the CPU cores so a concurrent training run keeps the rest.
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

H = 480
EXTS = {".jpg", ".jpeg", ".png"}   # output is always .jpg (video_dataset.py globs *.jpg)


def resize_video(args):
    src, dst = args
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS):
        im = Image.open(f).convert("RGB")
        w, h = im.size
        if h != H:
            im = im.resize((round(w * H / h), H), Image.BICUBIC)
        im.save(dst / (f.stem + ".jpg"), quality=95)
    return src.name


def main():
    src, dst, prefix = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    videos = sorted({p.parent for p in src.rglob("*") if p.suffix.lower() in EXTS})
    jobs = [(v, dst / f"{prefix}_{v.name}") for v in videos]
    print(f"{len(jobs)} videos under {src}")
    with Pool(max(1, os.cpu_count() // 2)) as pool:   # leave half the cores to training
        for i, _ in enumerate(pool.imap_unordered(resize_video, jobs), 1):
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
