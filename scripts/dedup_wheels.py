#!/usr/bin/env python3
"""Remove duplicate older wheels, keeping only the newest version of each package."""
import os
import re
from collections import defaultdict

WHEEL_DIR = "/wheels"
files = os.listdir(WHEEL_DIR)
pkgs = defaultdict(list)

for f in files:
    m = re.match(r'^([a-zA-Z0-9_.]+)-([0-9][a-zA-Z0-9.]*)', f)
    if m:
        pkgs[m.group(1)].append((m.group(2), f))

for name, versions in pkgs.items():
    if len(versions) > 1:
        versions.sort()
        for _, old_filename in versions[:-1]:
            path = os.path.join(WHEEL_DIR, old_filename)
            os.remove(path)
            print("Removed duplicate:", old_filename)

print("Done. Unique packages:", len(pkgs))
