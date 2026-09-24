# subset_stats.py  -- per-class counts of the BALANCED subset actually used.
# Run:  python subset_stats.py            (defaults to D:\Research - Lasya\NewWebcams)
#   or: python subset_stats.py --splits_dir "D:\Research - Lasya\NewWebcams"
import argparse, os
from collections import Counter
from statistics import median

# ---- verbatim from dsets/webcams/cls_10full.py ----
CLASS_NAMES  = ['1.0','2.0','3.0','4.0','5.0','6.0','7.0','8.0','9.0','10.0']
CLASS_GROUPS = [{0.0,1.0,1.25,1.5,1.75},{2.0,2.25,2.5},{3.0},{4.0},{5.0},
                {6.0},{7.0},{8.0},{9.0},{10.0}]
# ---- verbatim from dsets/webcams/common.py ----
def get_str_label(n): return n.split('_')[2].split('.')[0].split('S')[1].split('m')[0].replace('-', '.')
def get_float_label(s): return 10.0 if s == '10+' else min(float(s), 10.0)
def get_class_label(f):
    for i, g in enumerate(CLASS_GROUPS):
        if f in g: return i
    return -1
def path_from_line(line):
    line = line.strip().strip('"')
    if not line: return None
    for tok in line.split(','):
        if tok.lower().rstrip('"').endswith(('.png','.jpg','.jpeg')): return tok.strip().strip('"')
    return line
def site_of(p): return os.path.basename(p).split('_')[0]
def classify(p):
    try: return get_class_label(get_float_label(get_str_label(os.path.basename(p))))
    except Exception: return -2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default=r"D:\Research - Lasya\NewWebcams")
    ap.add_argument("--files", nargs="+", default=["train.csv","validation.csv","test.csv"])
    a = ap.parse_args()
    combined = []
    for fn in a.files:
        fp = os.path.join(a.splits_dir, fn)
        if not os.path.exists(fp): print(f"  [skip] {fp} not found"); continue
        paths = [p for p in (path_from_line(l) for l in open(fp, encoding="utf-8")) if p]
        combined += paths; print(f"  {fn:16s}: {len(paths):6d} paths")
    if not combined: print("No split files found. Pass --splits_dir."); return
    cls = [classify(p) for p in combined]
    counts = Counter(c for c in cls if c >= 0); total = sum(counts.values())
    print(f"\n  combined paths {len(combined)} | dropped {sum(1 for c in cls if c<0)} | classified {total}\n")
    names = ["$\\leq$1~mi","2~mi","3~mi","4~mi","5~mi","6~mi","7~mi","8~mi","9~mi","$\\geq$10~mi"]
    print("LaTeX rows for Table 2 (balanced subset):"); print("-"*46)
    for i in range(10):
        c = counts.get(i,0)
        print(f"{names[i]:14s}& {c:>6} & {100*c/total:4.1f}\\% \\\\".replace(str(c), f"{c:,}".replace(",","{,}"),1))
    print("\\midrule"); print(f"\\textbf{{Total}} & \\textbf{{{total:,}}}".replace(",","{,}") + " & \\textbf{100.0\\%} \\\\")
    by_site = Counter(site_of(p) for p,c in zip(combined,cls) if c>=0)
    elig = [n for n in by_site.values() if n>=20]
    print("-"*46)
    print(f"\n  sites total {len(by_site)} | sites>=20 {len(elig)} (expect 255) | median/site(>=20) {median(sorted(elig)):.0f} (expect 41)")

if __name__ == "__main__": main()