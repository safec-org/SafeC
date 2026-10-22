#!/usr/bin/env python3
"""Checks that safeguard's std module sets are complete.

safeguard compiles and links only the std modules a build includes (see
safeguard/include/StdModules.h): a std header brings in the std sources
implementing it, found by file name or named by '// safeguard:' directives.
This fails if any std header's functions would end up without their
implementation in that subset.

For every std header H:
  - the functions H declares are the 'declare's in the IR of a program that
    includes H (safec declares every function a header names, used or not);
  - the subset is what 'safeguard std-modules std/H' prints;
  - each declared function, and each symbol the subset's own objects leave
    undefined, that some std object defines must be defined in the subset.
Std objects come from building a project with [build] std = "all" (every
std source compiled). It runs twice: without features, where a symbol only
a feature-gated source ('// safeguard: feature') defines is fine, and with
every such feature enabled. It also fails on 'problem' lines (a std source
implementing no header, a malformed directive).

Usage: check.py [--target T] [--keep DIR]
Env: SAFEC (or SAFEC_BIN), SAFEGUARD_BIN (default: safeguard), SAFEC_HOME
(the SafeC checkout), LLVM_NM (default: llvm-nm beside $CLANG_BIN or on PATH).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="")
ap.add_argument("--keep", default="")
args = ap.parse_args()

home = os.environ.get("SAFEC_HOME") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ["SAFEC_HOME"] = home
std = os.path.join(home, "std")
safec = os.environ.get("SAFEC") or os.environ.get("SAFEC_BIN") or "safec"
os.environ["SAFEC"] = safec
safeguard = os.environ.get("SAFEGUARD_BIN", "safeguard")


def find_nm():
    if os.environ.get("LLVM_NM"):
        return os.environ["LLVM_NM"]
    clang = shutil.which(os.environ.get("CLANG_BIN", "clang") or "clang")
    for d in ([os.path.dirname(clang)] if clang else []) + ["/opt/homebrew/opt/llvm/bin", "/usr/local/opt/llvm/bin"]:
        for n in ("llvm-nm", "llvm-nm.exe"):
            if os.path.exists(os.path.join(d, n)):
                return os.path.join(d, n)
    return shutil.which("llvm-nm") or "llvm-nm"


nm = find_nm()
target = ["--target", args.target] if args.target else []

# Feature-gated std sources, and every feature std knows.
gated, features = set(), set()
for d, _, fs in os.walk(std):
    for f in fs:
        if not f.endswith(".sc"):
            continue
        rel = os.path.relpath(os.path.join(d, f), std).replace(os.sep, "/")
        for line in open(os.path.join(d, f), errors="ignore"):
            m = re.match(r"\s*//\s*safeguard:\s*feature\s+([^(]*)", line)
            if m:
                gated.add(rel)
                features.update(m.group(1).split())
allfeat = ["--features", ",".join(sorted(features))] if features else []

work = args.keep or tempfile.mkdtemp(prefix="std_modules_")
proj = os.path.join(work, "all")
os.makedirs(os.path.join(proj, "src"), exist_ok=True)
with open(os.path.join(proj, "Package.toml"), "w") as f:
    f.write('[package]\nname = "std_all"\nversion = "0.1.0"\n\n[build]\nedition = "2026"\nstd = "all"\n')
with open(os.path.join(proj, "src", "main.sc"), "w") as f:
    f.write("int main() { return 0; }\n")
r = subprocess.run([safeguard, "build"] + target + allfeat, cwd=proj, capture_output=True, text=True)
objdir = os.path.join(proj, "build", "deps", *( [args.target] if args.target else [] ), "std")
if not os.path.isdir(objdir):
    print(r.stdout + r.stderr)
    sys.exit("check.py: no std objects in " + objdir)


def stem(unit):  # std/x/y.sc -> x_y_sc (Builder::ensureStdLib's object names)
    return re.sub(r"[/.]", "_", unit[4:] if unit.startswith("std/") else unit)


def nm_syms(path):
    out = subprocess.run([nm, "-P", "--extern-only", path], capture_output=True, text=True).stdout
    defs, undef = set(), set()
    for line in out.splitlines():
        p = line.split()
        if len(p) < 2 or p[0].endswith(":"):
            continue
        name = p[0]
        if sys.platform == "darwin" and not args.target and name.startswith("_"):
            name = name[1:]
        (undef if p[1] in ("U", "w", "v") else defs).add(name)
    return defs, undef


objs = {}      # stem -> (defs, undefs)
definers = {}  # symbol -> stems
for f in os.listdir(objdir):
    if f.endswith(".o"):
        s = f[:-2]
        objs[s] = nm_syms(os.path.join(objdir, f))
        for d in objs[s][0]:
            definers.setdefault(d, set()).add(s)
gated_stems = {stem(u) for u in gated}

headers = []
for d, _, fs in os.walk(std):
    for f in fs:
        if f.endswith(".h"):
            rel = os.path.relpath(os.path.join(d, f), std).replace(os.sep, "/")
            if not rel.startswith("wasm/") or "wasm" in args.target:
                headers.append(rel)
headers.sort()

errors, skipped = [], []
tu = os.path.join(work, "tu.sc")
ll = os.path.join(work, "tu.ll")


def modules(h, extra):
    r = subprocess.run([safeguard, "std-modules"] + target + extra + ["std/" + h], cwd=proj,
                       capture_output=True, text=True)
    if r.returncode != 0:
        errors.append(f"std/{h}: safeguard std-modules failed: {r.stderr.strip()}")
        return set(), []
    units, problems = set(), []
    for line in r.stdout.splitlines():
        if line.startswith("unit "):
            units.add(stem(line[5:]))
        elif line.startswith("problem "):
            problems.append(line[8:])
    return units, problems


problems_seen = set()
for h in headers:
    with open(tu, "w") as f:
        f.write(f"#include <std/{h}>\nint main() {{ return 0; }}\n")
    r = subprocess.run([safec, "-I", std, "-I", home, tu, "--emit-llvm", "--compat-preprocessor", "-o", ll]
                       + (["--target=" + args.target] if args.target else []), capture_output=True, text=True)
    if r.returncode != 0:
        skipped.append(h)
        continue
    declared = set()
    for line in open(ll, errors="ignore"):
        m = re.match(r"declare [^@]*@(\"[^\"]+\"|[\w.$]+)\(", line)
        if m and not m.group(1).startswith("llvm."):
            declared.add(m.group(1).strip('"'))
    for passname, extra in (("no features", []), ("all features", allfeat)):
        units, problems = modules(h, extra)
        for p in problems:
            if p not in problems_seen:
                problems_seen.add(p)
                errors.append("problem: " + p)
        need = set(declared)
        for u in units:
            if u in objs:
                need |= objs[u][1]
        for s in sorted(need):
            ds = definers.get(s)
            if not ds or ds & units:
                continue
            if passname == "no features" and ds <= gated_stems:
                continue
            errors.append(f"std/{h} ({passname}): {s} is defined by {', '.join(sorted(ds))}, "
                          f"not in its modules")

print(f"std module check: {len(headers)} headers, {len(objs)} std objects, "
      f"{len(skipped)} headers that don't compile alone for this target skipped")
if skipped:
    print("  skipped: " + " ".join(skipped))
for e in errors:
    print("  " + e)
if not args.keep:
    shutil.rmtree(work, ignore_errors=True)
sys.exit(1 if errors else 0)
