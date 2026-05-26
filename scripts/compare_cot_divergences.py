#!/usr/bin/env python3
"""
Compare per-episode eval results of two CoT models (e.g. base_cot vs grad_loss_cot)
and isolate the *divergent* episodes -- those where one model succeeds and the other
fails on the same task.

It reads the per-episode result files written by run_libero_plus_eval.py
(``{suite}_{model}_{start}_{end}_episodes.json``), compares the two models on the
set of task_ids they both ran, prints a summary broken down by perturbation
category, and writes a details file listing every divergent episode together with
the exact rollout video path(s) so they are easy to pull up and watch.

Example:
    python scripts/compare_cot_divergences.py \
        --suite libero_10 --model-a base_cot --model-b grad_loss_cot
"""

import argparse
import glob
import json
import os
import shutil
from collections import defaultdict


def load_episodes(results_dir, suite, model):
    """Merge all ``{suite}_{model}_*_episodes.json`` files into {task_id: record}."""
    pattern = os.path.join(results_dir, f"{suite}_{model}_*_episodes.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No episode files matched: {pattern}")

    episodes = {}
    for path in files:
        with open(path) as f:
            for rec in json.load(f):
                tid = rec["task_id"]
                prev = episodes.get(tid)
                if prev is not None and prev["success"] != rec["success"]:
                    print(f"  [warn] task_id {tid} appears twice for '{model}' with "
                          f"conflicting outcomes ({prev['success']} vs {rec['success']}); "
                          f"keeping the later one from {os.path.basename(path)}")
                episodes[tid] = rec
    print(f"Loaded {len(episodes)} unique episodes for '{model}' from {len(files)} file(s)")
    return episodes


def find_videos(rollouts_dir, model, task_id):
    """Return rollout video paths for (model, task_id). The ``--ep=<id>--`` delimiter
    makes the match exact (e.g. ep=12 will not match ep=121)."""
    pattern = os.path.join(rollouts_dir, f"rollouts_{model}", "**", f"*--ep={task_id}--*.mp4")
    return sorted(os.path.relpath(p, rollouts_dir) for p in glob.glob(pattern, recursive=True))


def copy_divergence_videos(divergences, rollouts_dir, out_dir, a, b):
    """Create one folder per divergence direction and copy both models' rollouts for
    each case into it (renamed so winner/loser is obvious). Returns the two dirs."""
    dir_a = os.path.join(out_dir, f"{a}_succeeds_{b}_fails")   # base wins
    dir_b = os.path.join(out_dir, f"{b}_succeeds_{a}_fails")   # grad wins
    os.makedirs(dir_a, exist_ok=True)
    os.makedirs(dir_b, exist_ok=True)

    copied, missing = 0, 0
    for d in divergences:
        dest = dir_a if d["winner"] == a else dir_b
        for model in (a, b):
            outcome = "SUCCESS" if d[f"{model}_success"] else "FAIL"
            vids = d[f"{model}_videos"]
            if not vids:
                missing += 1
                continue
            for i, rel in enumerate(vids):
                suffix = f"_{i}" if len(vids) > 1 else ""
                name = f"ep{d['task_id']:04d}__{model}__{outcome}{suffix}.mp4"
                shutil.copy2(os.path.join(rollouts_dir, rel), os.path.join(dest, name))
                copied += 1

    print(f"\nCopied {copied} videos into:")
    print(f"  {dir_a}/   ({a} ✓ / {b} ✗)")
    print(f"  {dir_b}/   ({b} ✓ / {a} ✗)")
    if missing:
        print(f"  [note] {missing} expected video(s) were not found on disk and could not be copied.")
    return dir_a, dir_b


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=".", help="Directory holding the *_episodes.json files")
    parser.add_argument("--rollouts-dir", default=None, help="Directory holding rollouts_<model>/ (default: results-dir)")
    parser.add_argument("--suite", default="libero_10", help="Task suite prefix in the filenames")
    parser.add_argument("--model-a", default="base_cot", help="First model / checkpoint name")
    parser.add_argument("--model-b", default="grad_loss_cot", help="Second model / checkpoint name")
    parser.add_argument("--out", default=None, help="Output details JSON (default: <suite>_<a>_vs_<b>_divergences.json)")
    parser.add_argument("--video-out-dir", default=None,
                        help="Where to copy divergence videos (default: <suite>_<a>_vs_<b>_divergence_videos)")
    parser.add_argument("--no-copy", action="store_true", help="Skip copying videos; only print summary and write the JSON")
    args = parser.parse_args()

    rollouts_dir = args.rollouts_dir or args.results_dir
    a, b = args.model_a, args.model_b
    out_path = args.out or f"{args.suite}_{a}_vs_{b}_divergences.json"
    video_out_dir = args.video_out_dir or f"{args.suite}_{a}_vs_{b}_divergence_videos"

    eps_a = load_episodes(args.results_dir, args.suite, a)
    eps_b = load_episodes(args.results_dir, args.suite, b)

    common = sorted(set(eps_a) & set(eps_b))
    print(f"\nTask ids run by both models: {len(common)} "
          f"(only-{a}: {len(set(eps_a) - set(eps_b))}, only-{b}: {len(set(eps_b) - set(eps_a))})\n")

    # Tally agreement / divergence, broken down by perturbation category.
    # direction "a_only"  -> a succeeds, b fails ; "b_only" -> b succeeds, a fails
    counts = defaultdict(lambda: {"both_success": 0, "both_fail": 0, "a_only": 0, "b_only": 0})
    divergences = []
    for tid in common:
        ra, rb = eps_a[tid], eps_b[tid]
        sa, sb = bool(ra["success"]), bool(rb["success"])
        category = ra.get("category") or rb.get("category")
        bucket = counts[category]
        if sa and sb:
            bucket["both_success"] += 1
        elif not sa and not sb:
            bucket["both_fail"] += 1
        else:
            direction = "a_only" if sa else "b_only"
            bucket[direction] += 1
            divergences.append({
                "task_id": tid,
                "task_name": ra.get("task_name") or rb.get("task_name"),
                "category": category,
                "winner": a if sa else b,
                "loser": b if sa else a,
                f"{a}_success": sa,
                f"{b}_success": sb,
                f"{a}_videos": find_videos(rollouts_dir, a, tid),
                f"{b}_videos": find_videos(rollouts_dir, b, tid),
            })

    # ---- Summary ----
    totals = {"both_success": 0, "both_fail": 0, "a_only": 0, "b_only": 0}
    for c in counts.values():
        for k in totals:
            totals[k] += c[k]

    # Column widths: category fixed, "<model> only" columns sized to their labels.
    a_label, b_label = f"{a} only", f"{b} only"
    cat_w, num_w = 22, 9
    aw, bw = max(len(a_label) + 2, 10), max(len(b_label) + 2, 10)
    line_w = cat_w + num_w * 3 + aw + bw

    def row(cat, both_s, both_f, a_only, b_only, diverge):
        return (f"{str(cat):<{cat_w}}{both_s:>{num_w}}{both_f:>{num_w}}"
                f"{a_only:>{aw}}{b_only:>{bw}}{diverge:>{num_w}}")

    total_div = totals["a_only"] + totals["b_only"]
    print("=" * line_w)
    print(f"DIVERGENCE SUMMARY  ({a} vs {b})  |  '{a} only' = {a} ✓ / {b} ✗   "
          f"'{b} only' = {b} ✓ / {a} ✗")
    print("=" * line_w)
    print(row("category", "both ✓", "both ✗", a_label, b_label, "diverge"))
    print("-" * line_w)
    for category in sorted(counts):
        c = counts[category]
        print(row(category, c["both_success"], c["both_fail"],
                  c["a_only"], c["b_only"], c["a_only"] + c["b_only"]))
    print("-" * line_w)
    print(row("TOTAL", totals["both_success"], totals["both_fail"],
              totals["a_only"], totals["b_only"], total_div))
    print("=" * line_w)
    print(f"\n{total_div} divergent episodes out of {len(common)} compared "
          f"({a} wins {totals['a_only']}, {b} wins {totals['b_only']}).")

    # ---- Details file ----
    divergences.sort(key=lambda d: (str(d["category"]), d["winner"], d["task_id"]))
    payload = {
        "suite": args.suite,
        "model_a": a,
        "model_b": b,
        "num_common_episodes": len(common),
        "totals": totals,
        "num_divergences": total_div,
        "divergences": divergences,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"\nWrote per-episode divergence details (with video paths) to: {out_path}")

    missing = [d for d in divergences if not d[f"{a}_videos"] or not d[f"{b}_videos"]]
    if missing:
        print(f"[note] {len(missing)} divergent episode(s) are missing a saved video for one "
              f"of the models (e.g. video_save_freq skipped them).")

    # ---- Copy videos into per-direction folders ----
    if not args.no_copy:
        copy_divergence_videos(divergences, rollouts_dir, video_out_dir, a, b)


if __name__ == "__main__":
    main()
