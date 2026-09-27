"""Environment check + honest compute budget estimate for THIS machine.

Run this first. It measures actual forward-pass throughput on your hardware and
converts it into wall-clock estimates for each experiment, so the GPU-vs-CPU
question gets answered with numbers from your laptop rather than a rule of
thumb.

    python scripts/00_setup_check.py
    python scripts/00_setup_check.py --backbone clip_vitb16 --benchmark
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from xgf.config import PATHS, RunConfig  # noqa: E402


def check_env() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_vram_gb"] = round(props.total_memory / 1024 ** 3, 2)
    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)
    for k, v in info.items():
        print("  {:<18} {}".format(k, v))

    if not torch.cuda.is_available():
        print("\n  !! torch cannot see a GPU.")
        print("     If nvidia-smi works, you installed the CPU-only wheel.")
        print("     Fix: update the NVIDIA driver, then")
        print("     pip uninstall -y torch torchvision")
        print("     pip install torch torchvision --index-url "
              "https://download.pytorch.org/whl/cu121")
    return info


def check_deps() -> None:
    print("\n" + "=" * 70)
    print("DEPENDENCIES")
    print("=" * 70)
    mods = ["numpy", "scipy", "pandas", "sklearn", "skimage", "PIL", "cv2",
            "timm", "open_clip", "matplotlib", "tqdm", "yaml"]
    missing = []
    for m in mods:
        try:
            __import__(m)
            print("  [ok]      {}".format(m))
        except ImportError:
            print("  [MISSING] {}".format(m))
            missing.append(m)
    if missing:
        print("\n  Install with: pip install -r requirements.txt")


def benchmark(backbone: str, image_size: int = 224, n_warmup: int = 3,
              n_iter: int = 12) -> dict:
    """Measure real forward throughput, per device available."""
    from xgf.models.detector import Detector

    print("\n" + "=" * 70)
    print("THROUGHPUT BENCHMARK  ({})".format(backbone))
    print("=" * 70)
    results = {}
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    for dev in devices:
        try:
            det = Detector(backbone).to(dev).eval()
        except Exception as exc:
            print("  {}: could not build backbone ({})".format(dev, exc))
            continue
        bs = 32 if dev == "cuda" else 16
        x = torch.rand(bs, 3, image_size, image_size, device=dev)
        amp = dev == "cuda"
        with torch.no_grad():
            for _ in range(n_warmup):
                if amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        det(x)
                else:
                    det(x)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iter):
                if amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        det(x)
                else:
                    det(x)
            if dev == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        ips = bs * n_iter / dt
        results[dev] = ips
        print("  {:<5} {:>8.1f} images/sec   (batch {}, {})".format(
            dev, ips, bs, "fp16" if amp else "fp32"))
        if dev == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            print("        peak VRAM {:.2f} GB".format(peak))
        del det
        if dev == "cuda":
            torch.cuda.empty_cache()
    return results


def estimate(ips: dict, cfg: RunConfig, n_detectors: int = 3,
             n_families: int = 4, n_operators: int = 6) -> None:
    """Convert throughput into per-experiment wall-clock estimates."""
    print("\n" + "=" * 70)
    print("WALL-CLOCK ESTIMATE FOR THIS MACHINE")
    print("=" * 70)

    n_img = cfg.n_eval_images
    steps = cfg.n_mask_steps

    # Forward-pass accounting, per detector.
    cheap_expl = 4                       # gradcam, gradcam++, inputgrad, random
    fwd_saliency_cheap = n_img * cheap_expl * 2
    fwd_rise = n_img * cfg.rise_n_masks
    fwd_lime = n_img * cfg.lime_n_samples
    fwd_curves = n_img * (cheap_expl + 2) * steps * 2 * n_operators
    fwd_leak = n_img * 21 * cfg.n_random_controls * n_operators

    budgets = {
        "E0 leakage null test (no explainer)": fwd_leak * n_detectors,
        "E1 feature extraction + probes": 60000 * n_detectors,
        "E2 cheap-saliency faithfulness grid": (
            (fwd_saliency_cheap + fwd_curves) * n_detectors * n_families
        ),
        "E3 RISE + LIME arm": (fwd_rise + fwd_lime) * n_detectors * n_families,
    }

    for dev, rate in ips.items():
        print("\n  --- on {} ({:.0f} img/s) ---".format(dev.upper(), rate))
        total = 0.0
        for name, fwds in budgets.items():
            hrs = fwds / rate / 3600.0
            total += hrs
            print("    {:<40} {:>10,.0f} fwd  {:>7.1f} h".format(name, fwds, hrs))
        print("    {:<40} {:>10}  {:>7.1f} h  ({:.1f} days)".format(
            "TOTAL", "", total, total / 24))

    print("\n  Knobs, in order of effect:")
    print("    rise_n_masks  {} -> 500     cuts E3 by ~4x".format(cfg.rise_n_masks))
    print("    n_eval_images {} -> 200     cuts everything by {:.1f}x".format(
        n_img, n_img / 200))
    print("    drop RISE+LIME entirely     removes E3")
    print("\n  Probe TRAINING itself is not in this table because it is")
    print("  logistic regression on cached features: seconds, CPU, no GPU needed.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50",
                    help="backbone to benchmark (resnet50 is fastest to download)")
    ap.add_argument("--benchmark", action="store_true",
                    help="run the throughput benchmark (downloads weights)")
    ap.add_argument("--eval-images", type=int, default=None)
    args = ap.parse_args()

    PATHS.ensure()
    check_env()
    check_deps()

    cfg = RunConfig()
    if args.eval_images:
        cfg.n_eval_images = args.eval_images

    if args.benchmark:
        ips = benchmark(args.backbone)
    else:
        # Conservative priors so the estimate table is still useful without a
        # weight download; clearly flagged as such.
        ips = {"cpu": 15.0}
        if torch.cuda.is_available():
            ips["cuda"] = 150.0
        print("\n  (using PRIOR throughput values; pass --benchmark to measure)")
    estimate(ips, cfg)

    print("\n" + "=" * 70)
    print("NEXT: python scripts/01_build_manifest.py --selftest")
    print("=" * 70)


if __name__ == "__main__":
    main()
