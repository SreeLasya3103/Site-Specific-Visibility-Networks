"""Diagnose problem sites before generating references."""
import os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_references import extract_site_id, extract_visibility

SPLITS_PATH = r"D:\Research - Lasya\NewWebcams"
MIN_SAMPLES = 20  # must match your trainpersite.py setting

all_samples = []
for split in ["train.csv", "validation.csv", "test.csv"]:
    fp = os.path.join(SPLITS_PATH, split)
    if os.path.exists(fp):
        with open(fp) as f:
            all_samples.extend([l.rstrip() for l in f if l.strip()])

sites = defaultdict(list)
for s in all_samples:
    sites[extract_site_id(s)].append(s)

# Compute best_vis and sample count per site
site_info = {}
for site_id, samples in sites.items():
    best = -1
    for s in samples:
        v = extract_visibility(s)
        if v > best:
            best = v
    site_info[site_id] = (best, len(samples))

# ----- Issue A: sites with best_vis=-1 -----
print("=" * 70)
print("ISSUE A: Sites where parser failed on ALL samples (best_vis=-1)")
print("=" * 70)
failed = [(sid, n) for sid, (v, n) in site_info.items() if v < 0]
for sid, n in failed:
    print(f"\n{sid} ({n} samples):")
    for s in sites[sid][:5]:
        print(f"    {s}")

# ----- Issue B: low-vis references -----
print("\n" + "=" * 70)
print(f"ISSUE B: Sites with best_vis <= 6 (bad clear-day references)")
print("=" * 70)
low = sorted(
    [(sid, v, n) for sid, (v, n) in site_info.items() if 0 <= v <= 6],
    key=lambda x: x[1]
)
print(f"\n{'SITE':<12} {'best_vis':>10} {'samples':>10} {'>=min?':>10}")
for sid, v, n in low:
    eligible = "YES" if n >= MIN_SAMPLES else "no"
    print(f"{sid:<12} {v:>10.2f} {n:>10d} {eligible:>10}")

# ----- Issue C: intersection of problems with min_samples filter -----
print("\n" + "=" * 70)
print(f"ISSUE C: How many problem sites survive the min_samples={MIN_SAMPLES} filter?")
print("=" * 70)
eligible = {sid for sid, (_, n) in site_info.items() if n >= MIN_SAMPLES}
print(f"Total sites: {len(site_info)}")
print(f"Eligible sites (>={MIN_SAMPLES} samples): {len(eligible)}")

problem_eligible_parsefail = [sid for sid, n in failed if sid in eligible]
problem_eligible_lowvis = [sid for sid, v, n in low if sid in eligible]
print(f"\nOf the {len(failed)} parser-failure sites, {len(problem_eligible_parsefail)} are eligible:")
for sid in problem_eligible_parsefail:
    print(f"  {sid}")
print(f"\nOf the {len(low)} low-vis sites (<=6mi), {len(problem_eligible_lowvis)} are eligible:")
for sid in problem_eligible_lowvis:
    v, n = site_info[sid]
    print(f"  {sid}: best_vis={v}, n_samples={n}")

# ----- Summary -----
good_eligible = sum(1 for sid in eligible if site_info[sid][0] >= 7)
print("\n" + "=" * 70)
print("SUMMARY for 4-stream experiment")
print("=" * 70)
print(f"  Eligible sites with good reference (best_vis >= 7mi): {good_eligible}")
print(f"  Eligible sites with problematic reference:           {len(problem_eligible_lowvis) + len(problem_eligible_parsefail)}")
