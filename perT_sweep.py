"""Per-T specialist sweep: resumable training + 5x4 evaluation matrix.

Environment constraint: a background bash task is killed around ~2h. Therefore
EACH invocation trains AT MOST ONE model, and `next` is re-invoked from the
task notification (or by a periodic check). Every epoch writes checkpoint.pth,
so a killed run resumes losslessly via --resume.

Usage (cwd = optics_main):
  python perT_sweep.py status
  python perT_sweep.py next --dry-run     # what would run, without running it
  python perT_sweep.py next               # train the first unfinished model
  python perT_sweep.py supervise          # own the whole chain until eval finishes
  python perT_sweep.py train-one --model T8
  python perT_sweep.py eval               # after all weights exist
  python perT_sweep.py report             # reprint matrices from existing CSVs
"""
import argparse, csv, json, math, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SWEEP = ROOT / "perT_sweep"
PY = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe"
TS = ["8", "16", "24", "32"]
SCENES = ["aerial", "crash", "drop", "kobe", "runner", "traffic"]
LOCK = SWEEP / "train.lock"
SUP_LOCK = SWEEP / "supervisor.lock"

# name -> (T_list, n_train). 50 step/epoch x 30 epoch = 1500 steps for ALL.
MODELS = {
    "T8":     ("8",           200),
    "T16":    ("16",          100),
    "T24":    ("24",          50),
    "T32":    ("32",          50),
    "tfree":  ("8 16 24 32", 200),
}
# T16 is already running (launched manually, with DAVIS verified), so keep it first.
ORDER = ["T16", "T8", "T24", "T32", "tfree"]
COMMON = ["--epochs", "30", "--warm_start", "checkpoints/best_model.pth",
          "--legacy_T", "8", "--davis_root", "../DAVIS", "--real_ratio", "0.7",
          "--amp", "--ckpt", "--vgg_weight", "0.02", "--log_every", "10"]


def is_done(out):
    """history.json is written ONCE, at the very end of the training script
    (after the final per-T evaluation). Its per_T block is the real completion
    marker -- best_epoch is NOT, because EMA can peak before epoch 30."""
    h = out / "history.json"
    if not h.exists():
        return False
    try:
        j = json.loads(h.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(j.get("per_T"))


def status():
    for name in ORDER:
        out = SWEEP / name
        ck = (out / "checkpoint.pth").exists()
        ep = ""
        if ck:
            try:
                import torch
                c = torch.load(out / "checkpoint.pth", map_location="cpu",
                               weights_only=False)
                ep = f" last_epoch={c.get('epoch', '?')}"
            except Exception as e:
                ep = f" ckpt_unreadable={type(e).__name__}"
        print(f"{name:7s} complete={is_done(out)!s:5s} checkpoint={ck!s:5s} "
              f"history={(out / 'history.json').exists()!s:5s} "
              f"n_train={MODELS[name][1]}{ep}")


def live_trainers():
    """Windows: PIDs of python.exe processes running the training script.
    Returns None if detection failed (caller must refuse to start a job)."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
          "Where-Object { $_.CommandLine -like '*dd_madun_v3_train_tfree.py*' } | "
          "Select-Object -ExpandProperty ProcessId")
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            text=True, timeout=60)
        return [int(x) for x in out.split() if x.strip().isdigit()]
    except Exception:
        return None


def train_one(name, dry=False):
    if name not in MODELS:
        raise SystemExit(f"unknown model {name}; choose from {list(MODELS)}")
    T, n = MODELS[name]
    out = SWEEP / name
    out.mkdir(parents=True, exist_ok=True)
    if is_done(out):
        print(f"[skip] {name} already complete")
        return 0
    rel = f"perT_sweep/{name}"
    cmd = [PY, "-u", "dd_madun_v3_train_tfree.py",
           "--T_list", *T.split(), "--n_train", str(n),
           *COMMON, "--out_dir", rel]
    resumed = (out / "checkpoint.pth").exists()
    if resumed:
        cmd += ["--resume", f"{rel}/checkpoint.pth"]
    print("CMD:", " ".join(cmd), flush=True)
    if dry:
        return 0
    with open(out / "command.txt", "a", encoding="utf-8") as f:
        f.write(f"\n# driver {time.strftime('%F %T')} resume={resumed}\n"
                f"# {' '.join(cmd)}\n")
    t0 = time.time()
    with open(out / "train.log", "a", encoding="utf-8") as log:
        log.write(f"\n===== driver run {time.strftime('%F %T')} "
                  f"resume={resumed} =====\n")
        log.flush()
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log,
                             stderr=subprocess.STDOUT)
        try:
            LOCK.write_text(str(p.pid), encoding="utf-8")
        except Exception:
            pass
        rc = p.wait()
        LOCK.unlink(missing_ok=True)
    rec = dict(model=name, cmd=" ".join(cmd), exit=rc, resumed=resumed,
               started=t0, ended=time.time(), complete=is_done(out))
    mf = SWEEP / "manifest.json"
    arr = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else []
    arr.append(rec)
    mf.write_text(json.dumps(arr, indent=2), encoding="utf-8")
    print(f"[done] {name} rc={rc} complete={rec['complete']} "
          f"elapsed={(rec['ended'] - t0) / 3600:.2f}h", flush=True)
    return rc


def pid_alive(pid):
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], text=True, timeout=30)
        return str(pid) in out
    except Exception:
        return False


def supervisor_alive():
    """PID of a live supervisor, or None. A stale lock is ignored."""
    if not SUP_LOCK.exists():
        return None
    try:
        pid = int(SUP_LOCK.read_text(encoding="utf-8").strip())
    except Exception:
        return None
    return pid if pid_alive(pid) else None


def supervise(poll=60, budget_h=24):
    """One process owns the whole chain: wait for the current trainer, start the
    next unfinished model, repeat, then evaluate. Terminates when the matrix is
    produced (or after budget_h). Safe to re-invoke: the lock prevents two
    supervisors, and `live_trainers()` prevents two GPU jobs."""
    other = supervisor_alive()
    if other:
        print(f"[abort] supervisor pid={other} already owns the chain")
        return 2
    SWEEP.mkdir(parents=True, exist_ok=True)
    SUP_LOCK.write_text(str(os.getpid()), encoding="utf-8")
    deadline = time.time() + budget_h * 3600
    slog = SWEEP / "supervisor.log"
    try:
        while time.time() < deadline:
            live = live_trainers()
            if live is None:
                print("[abort] cannot detect trainers; refusing to risk two jobs",
                      flush=True)
                return 2
            if live:
                with open(slog, "a", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%F %T')} waiting: trainer {live}\n")
                time.sleep(poll)
                continue
            todo = [m for m in ORDER if not is_done(SWEEP / m)]
            with open(slog, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%F %T')} remaining={todo}\n")
            if not todo:
                print("[all-done] every model complete; running eval", flush=True)
                evaluate()
                return 0
            print(f"[supervise] starting {todo[0]}", flush=True)
            rc = train_one(todo[0])
            if rc:
                print(f"[warn] {todo[0]} exited rc={rc}; re-invoke to resume",
                      flush=True)
                return rc
        print("[supervise] time budget reached; re-invoke to continue")
        return 0
    finally:
        SUP_LOCK.unlink(missing_ok=True)


def train_next(dry=False):
    """Pick the first unfinished model, but NEVER start a second GPU job."""
    sup = supervisor_alive()
    if sup:
        print(f"[busy] supervisor pid={sup} owns the chain; nothing to start")
        return 0
    live = live_trainers()
    if live is None:
        print("[abort] could not determine whether a trainer is alive; "
              "refusing to risk two jobs on one GPU")
        return 2
    if live:
        print(f"[busy] trainer alive pids={live}; nothing to start")
        return 0
    if LOCK.exists():
        LOCK.unlink(missing_ok=True)  # stale, no live trainer
    todo = [m for m in ORDER if not is_done(SWEEP / m)]
    if not todo:
        print("[all-done] every model is complete; run `eval` next")
        return 0
    print(f"[next] -> {todo[0]}   ({len(todo)} models remaining: {todo})",
          flush=True)
    if dry:
        print("CMD:", " ".join(
            [PY, "-u", "dd_madun_v3_train_tfree.py", "--T_list",
             *MODELS[todo[0]][0].split(), "--n_train", str(MODELS[todo[0]][1]),
             *COMMON, "--out_dir", f"perT_sweep/{todo[0]}"]))
        return 0
    return train_one(todo[0])


def collect_weights():
    w = []
    for name in ORDER:
        for tag, fn in [("nonEMA", "checkpoint.pth"), ("EMA", "best_model.pth")]:
            p = SWEEP / name / fn
            if p.exists():
                w.append((f"{name}_{tag}", f"perT_sweep/{name}/{fn}"))
    for tag, fn in [("nonEMA", "checkpoint.pth"), ("EMA", "best_model.pth")]:
        p = ROOT / "output_tfree" / fn
        if p.exists():
            w.append((f"existing_tfree_{tag}", f"output_tfree/{fn}"))
    return w


def evaluate():
    weights = collect_weights()
    if not weights:
        raise SystemExit("no weights found")
    print(f"evaluating {len(weights)} weight variants x {len(TS)} T x "
          f"{len(SCENES)} real scenes", flush=True)
    for label, rel in weights:
        out_csv = ROOT / f"perT_sweep/eval_{label}.csv"
        if out_csv.exists() and sum(1 for _ in open(out_csv, encoding="utf-8")) > 1:
            print(f"[skip] {label}: CSV already exists", flush=True)
            continue
        cmd = [PY, "-m", "sci_ablation.run_ablation", "--exp", "T",
               "--Ts", *TS, "--methods", "tfree",
               "--ckpt", f"tfree={rel}", "--device", "cuda",
               "--out", f"perT_sweep/eval_{label}.csv"]
        print(f"[{label}] tfree={rel}", flush=True)
        log = ROOT / f"perT_sweep/eval_{label}.log"
        with open(log, "w", encoding="utf-8") as lf:
            subprocess.call(cmd, cwd=str(ROOT), stdout=lf,
                            stderr=subprocess.STDOUT)
    report()


def _means(path):
    if not path.exists():
        return {}, {}
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    mp, ms = {}, {}
    for T in TS:
        ok = [r for r in rows if r["T"] == T and not r.get("err", "").strip()]
        mp[T] = sum(float(r["psnr"]) for r in ok) / len(ok) if ok else math.nan
        ms[T] = sum(float(r["ssim"]) for r in ok) / len(ok) if ok else math.nan
    return mp, ms


def report():
    weights = collect_weights()
    tab = {}
    for label, _ in weights:
        tab[label] = _means(ROOT / f"perT_sweep/eval_{label}.csv")
    have_data = any(not math.isnan(v)
                    for (mp, ms) in tab.values() for v in list(mp.values()) + list(ms.values()))
    if not have_data:
        print("[report] no evaluation CSVs yet -- refusing to write all-NaN "
              "matrix files (nothing has been measured)")
        return
    for key, fname, fmt in [("psnr", "matrix_psnr.csv", ".2f"),
                            ("ssim", "matrix_ssim.csv", ".4f")]:
        out = SWEEP / fname
        with open(out, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["weight", *[f"T={T}" for T in TS]])
            for label, _ in weights:
                m = tab[label][0 if key == "psnr" else 1]
                wcsv.writerow([label, *[format(m.get(T, math.nan), fmt)
                                        for T in TS]])
        spec = "10.2f" if key == "psnr" else "10.4f"
        print(f"\n=== 6-scene mean {key.upper()} ===  saved -> {out}")
        print(f"{'weight':26s}" + "".join(f"{'T=' + T:>10s}" for T in TS))
        for label, _ in weights:
            m = tab[label][0 if key == "psnr" else 1]
            vals = "".join(format(m.get(T, math.nan), spec) for T in TS)
            print(f"{label:26s}{vals}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("train-one")
    p1.add_argument("--model", required=True)
    p1.add_argument("--dry-run", action="store_true")
    p2 = sub.add_parser("next")
    p2.add_argument("--dry-run", action="store_true")
    p3 = sub.add_parser("supervise")
    p3.add_argument("--poll", type=int, default=60)
    p3.add_argument("--budget-hours", type=float, default=24)
    sub.add_parser("eval")
    sub.add_parser("report")
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "train-one":
        sys.exit(train_one(a.model, a.dry_run) or 0)
    elif a.cmd == "next":
        sys.exit(train_next(a.dry_run) or 0)
    elif a.cmd == "supervise":
        sys.exit(supervise(a.poll, a.budget_hours) or 0)
    elif a.cmd == "eval":
        evaluate()
    elif a.cmd == "report":
        report()
    else:
        status()
