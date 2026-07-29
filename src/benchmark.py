"""Days 9-10: measure how reliable the pick-and-place actually is.

Runs many trials with the cube randomized each time, logs every trial to CSV, and
reports a success rate with a breakdown of *how* things failed.

    python src/benchmark.py                        # 50 trials, full workspace
    python src/benchmark.py --trials 50 --zone safe
    python src/benchmark.py --retries 0            # measure without recovery
    python src/benchmark.py --noise-sweep          # perception-error tolerance

Why two workspace zones
-----------------------
Sampling only the middle of the workspace gives 100% and measures nothing. The
"full" zone deliberately includes the edges found by probing in pick_place.py --
near the base where top-down IK stops converging, and near the bin where the arm
fouls the walls -- so the number means something and the failure handling is
actually exercised. "safe" is the interior, for a clean regression check.
"""
import argparse
import csv
import io
import contextlib
import time
from pathlib import Path

import numpy as np

from pick_place import (DROPPED, GRASP_FAILED, IK_FAILED, MISSED_BIN,
                        NOT_DETECTED, SUCCESS, run_trial)
from robot import load_pick_scene

ZONES = {
    # x range, y range. Cube always rests on the floor, so z = half-extent.
    "safe": ((0.38, 0.65), (-0.20, 0.25)),
    "full": ((0.30, 0.80), (-0.28, 0.45)),
}
OUTCOMES = [SUCCESS, GRASP_FAILED, DROPPED, MISSED_BIN, IK_FAILED, NOT_DETECTED]
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def sample_cube(rng, zone):
    (x0, x1), (y0, y1) = ZONES[zone]
    return [rng.uniform(x0, x1), rng.uniform(y0, y1), 0.025]


def run_campaign(model, data, n, zone, retries, noise_mm, rng, quiet=True,
                 label=""):
    """Run n randomized trials; return the list of TrialResult."""
    results = []
    t_wall = time.perf_counter()
    for i in range(n):
        xyz = sample_cube(rng, zone)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = run_trial(model, data, xyz, max_retries=retries,
                          vision_noise_mm=noise_mm, rng=rng, verbose=True)
        results.append(r)
        if not quiet:
            flag = "ok " if r.success else "FAIL"
            print(f"  {label}trial {i+1:3d}/{n}  ({xyz[0]:.3f},{xyz[1]:+.3f})  "
                  f"{flag} {r.outcome:<13s} attempts={r.attempts}")
    elapsed = time.perf_counter() - t_wall
    return results, elapsed


def summarize(results, elapsed, title):
    n = len(results)
    wins = [r for r in results if r.success]
    print(f"\n{'='*62}")
    print(f"{title}   n={n}   wall time {elapsed:.0f}s")
    print(f"{'='*62}")
    print(f"SUCCESS RATE : {len(wins)}/{n} = {100*len(wins)/n:.1f}%")

    print("\noutcome breakdown")
    for o in OUTCOMES:
        c = sum(1 for r in results if r.outcome == o)
        if c:
            bar = "#" * int(round(40 * c / n))
            print(f"  {o:<14s} {c:3d}  {100*c/n:5.1f}%  {bar}")

    # Did the retry logic earn its keep?
    recovered = [r for r in wins if r.attempts > 1]
    print(f"\nfirst-attempt successes : {len(wins)-len(recovered)}/{n}")
    print(f"recovered by retry      : {len(recovered)}/{n}"
          + (f"  (+{100*len(recovered)/n:.1f} pts)" if recovered else ""))

    errs = [r.vision_err_mm for r in results if r.vision_err_mm is not None]
    if errs:
        print(f"\nvision error  mean {np.mean(errs):.1f} mm | "
              f"max {np.max(errs):.1f} mm")
    dists = [r.bin_dist_mm for r in wins]
    if dists:
        print(f"placement     mean {np.mean(dists):.1f} mm from bin centre | "
              f"max {np.max(dists):.1f} mm")

    # Where in the workspace did failures cluster?
    fails = [r for r in results if not r.success]
    if fails:
        print("\nfailures by cube position")
        for r in fails:
            ph = f" during {r.abort_phase}" if r.abort_phase else ""
            print(f"  ({r.cube_start[0]:.3f},{r.cube_start[1]:+.3f})  "
                  f"{r.outcome}{ph}")
    return len(wins) / n


def write_csv(results, path, zone, retries, noise_mm):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial", "zone", "retries_allowed", "vision_noise_mm",
                    "cube_x", "cube_y", "outcome", "success", "attempts",
                    "vision_err_mm", "bin_dist_mm", "sim_time_s", "abort_phase"])
        for i, r in enumerate(results, 1):
            w.writerow([i, zone, retries, noise_mm,
                        f"{r.cube_start[0]:.4f}", f"{r.cube_start[1]:.4f}",
                        r.outcome, int(r.success), r.attempts,
                        "" if r.vision_err_mm is None else f"{r.vision_err_mm:.2f}",
                        f"{r.bin_dist_mm:.2f}", f"{r.sim_time:.2f}",
                        r.abort_phase or ""])
    print(f"\nlogged {len(results)} trials -> {path}")


def noise_sweep(model, data, per_level, zone, retries, seed):
    """How much perception error can the grasp absorb before it fails?

    The cliff lands where geometry says it should: the gripper opens to 40 mm per
    side and the cube half-width is 25 mm, leaving 15 mm of finger clearance. Past
    roughly that much lateral error a finger strikes the cube instead of passing
    around it, and the grasp fails.
    """
    levels = [0, 5, 10, 15, 20, 25, 30, 40]
    print(f"perception-error tolerance ({per_level} trials per level, "
          f"zone={zone}, retries={retries})\n")
    print(f"{'noise mm':>9} {'success':>9} {'rate':>7}   ")
    rows = []
    for mm in levels:
        rng = np.random.default_rng(seed)      # same placements at every level
        res, _ = run_campaign(model, data, per_level, zone, retries, mm, rng)
        wins = sum(r.success for r in res)
        rate = 100 * wins / per_level
        bar = "#" * int(round(rate / 4))
        print(f"{mm:>9} {wins:>4}/{per_level:<4} {rate:>6.0f}%  {bar}")
        rows.append((mm, wins, per_level, rate))

    path = RESULTS_DIR / f"noise_sweep_{zone}_r{retries}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["vision_noise_mm", "successes", "trials", "success_rate_pct",
                    "zone", "retries_allowed"])
        for mm, wins, tot, rate in rows:
            w.writerow([mm, wins, tot, f"{rate:.1f}", zone, retries])
    print(f"\nlogged sweep -> {path}")
    return rows


def main():
    p = argparse.ArgumentParser(description="Pick-and-place reliability benchmark")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--zone", choices=list(ZONES), default="full")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--noise", type=float, default=0.0,
                   help="Synthetic perception error, mm std dev.")
    p.add_argument("--noise-sweep", action="store_true")
    p.add_argument("--per-level", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    model, data = load_pick_scene()

    if args.noise_sweep:
        noise_sweep(model, data, args.per_level, args.zone, args.retries, args.seed)
        return

    rng = np.random.default_rng(args.seed)
    print(f"running {args.trials} trials | zone={args.zone} "
          f"x{ZONES[args.zone][0]} y{ZONES[args.zone][1]} | "
          f"retries={args.retries} | noise={args.noise} mm\n")
    results, elapsed = run_campaign(model, data, args.trials, args.zone,
                                    args.retries, args.noise, rng,
                                    quiet=args.quiet)
    rate = summarize(results, elapsed,
                     f"zone={args.zone}  retries={args.retries}  "
                     f"noise={args.noise}mm")
    name = args.csv or f"benchmark_{args.zone}_r{args.retries}_n{args.noise:g}.csv"
    write_csv(results, RESULTS_DIR / name, args.zone, args.retries, args.noise)
    print("\nRESULT:", "PASS" if rate >= 0.9 else "NEEDS WORK",
          f"(threshold 90%, got {100*rate:.1f}%)")


if __name__ == "__main__":
    main()
