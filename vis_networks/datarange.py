# save as date_range.py in the NewWebcams folder, run: python date_range.py
import os, re
from datetime import datetime

ROOT = r"D:\Research - Lasya\NewWebcams"
pat = re.compile(r'(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})')
dates = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    for name in dirnames + filenames:
        m = pat.search(name)
        if m:
            try:
                dates.append(datetime(*map(int, m.groups())))
            except ValueError:
                pass

if dates:
    print(f"Total timestamped items : {len(dates):,}")
    print(f"Oldest : {min(dates)}")
    print(f"Newest : {max(dates)}")
    print(f"Span   : {(max(dates)-min(dates)).days} days "
          f"(~{(max(dates)-min(dates)).days/365.25:.1f} years)")
else:
    print("No timestamp pattern matched — check folder naming.")