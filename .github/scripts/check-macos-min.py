#!/usr/bin/env python3
"""Checks the macOS a Mach-O binary, object or archive is built for.

Every Mach-O file names the oldest macOS it runs on (LC_BUILD_VERSION's
'minos', or LC_VERSION_MIN_MACOSX's 'version' in older ones). SafeC's release
binaries and prebuilt std objects must run on macOS 11.0 (release.yml), so
this fails if any of them -- each member of an archive too -- names a newer
macOS than --max, or names none.

Usage: check-macos-min.py [--max 11.0] <file>...
"""
import argparse
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--max", default="11.0")
ap.add_argument("files", nargs="+")
args = ap.parse_args()


def ver(s):
    return tuple(int(x) for x in s.split("."))


limit = ver(args.max)
failed = False
for f in args.files:
    r = subprocess.run(["otool", "-l", f], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{f}: otool -l failed: {r.stderr.strip()}")
        failed = True
        continue
    found = {}      # member -> minos
    member = f
    cmd = None
    for line in r.stdout.splitlines():
        s = line.strip()
        if line.endswith(":") and not line.startswith(" ") and "Load command" not in line:
            member = line[:-1]
            found.setdefault(member, None)
            continue
        if s.startswith("cmd "):
            cmd = s.split()[1]
        elif cmd == "LC_BUILD_VERSION" and s.startswith("minos "):
            found[member] = s.split()[1]
        elif cmd == "LC_VERSION_MIN_MACOSX" and s.startswith("version "):
            found[member] = s.split()[1]
    bad = {m: v for m, v in found.items() if v is None or ver(v) > limit}
    vs = sorted({v for v in found.values() if v}, key=ver)
    if bad:
        failed = True
        print(f"{f}: built for a newer macOS than {args.max} (or none named):")
        for m, v in sorted(bad.items()):
            print(f"  {m}: {v or 'no minimum macOS'}")
    else:
        print(f"{f}: ok ({len(found)} Mach-O file(s), minimum macOS {', '.join(vs)})")
sys.exit(1 if failed else 0)
