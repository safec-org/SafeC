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

With --prebuilt DIR (a release install's lib/<triple>/std, see
.github/scripts/package.sh), it also checks the prebuilt std's sections: each
<section>.units lists exactly the std sources of that section no feature
gates (std/<section>/..., 'core' for std/*.sc), each <section>.a holds
exactly their objects, no symbol is defined in two sections, and for every
header the sections holding its modules define what it needs -- what a build
linking only those sections gets.

Usage: check.py [--target T] [--keep DIR] [--prebuilt DIR]
Env: SAFEC (or SAFEC_BIN), SAFEGUARD_BIN (default: safeguard), SAFEC_HOME
(the SafeC checkout), LLVM_NM (default: llvm-nm beside $CLANG_BIN or its
realpath, llvm-nm-<clang major>, or llvm-nm on PATH).
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
ap.add_argument("--prebuilt", default="")
args = ap.parse_args()

home = os.environ.get("SAFEC_HOME") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ["SAFEC_HOME"] = home
std = os.path.join(home, "std")
safec = os.environ.get("SAFEC") or os.environ.get("SAFEC_BIN") or "safec"
os.environ["SAFEC"] = safec
safeguard = os.environ.get("SAFEGUARD_BIN", "safeguard")


# COFF COMDAT constants and string literals (MSVC's '__real@<hex>',
# '__xmm@<hex>', '??_C@...'): every object that needs one defines it and the
# linker keeps one copy, so they repeat across sections by design.
MERGEABLE = re.compile(r"^(__real@|__xmm@|__ymm@|__zmm@|\?\?_C@)")

def find_llvm_tool(tool, env):
    """An LLVM tool: $<env>; beside realpath($CLANG_BIN) (apt.llvm.org's
    /usr/bin/clang-21 links to /usr/lib/llvm-21/bin, where the unversioned
    tools are); <tool>-<major> with clang's major version; <tool> on PATH."""
    if os.environ.get(env):
        return os.environ[env]
    exe = ".exe" if os.name == "nt" else ""
    clang_name = os.environ.get("CLANG_BIN") or "clang"
    clang = shutil.which(clang_name)
    dirs = []
    if clang:
        dirs += [os.path.dirname(clang), os.path.dirname(os.path.realpath(clang))]
    for d in dirs:
        if os.path.exists(os.path.join(d, tool + exe)):
            return os.path.join(d, tool + exe)
    major = None
    m = re.search(r"-(\d+)(?:\.exe)?$", os.path.basename(clang_name))
    if m:
        major = m.group(1)
    elif clang:
        try:
            v = subprocess.run([clang, "--version"], capture_output=True, text=True).stdout
            m = re.search(r"clang version (\d+)", v)
            major = m.group(1) if m else None
        except OSError:
            pass
    if major and shutil.which(f"{tool}-{major}"):
        return shutil.which(f"{tool}-{major}")
    if shutil.which(tool):
        return shutil.which(tool)
    for d in ("/opt/homebrew/opt/llvm/bin", "/usr/local/opt/llvm/bin"):   # Homebrew's keg-only LLVM
        if os.path.exists(os.path.join(d, tool)):
            return os.path.join(d, tool)
    sys.exit(f"check.py: cannot find {tool}: set {env}, or CLANG_BIN to a clang with {tool} beside it "
             f"(looked beside {clang or clang_name}, for {tool}-{major or '<major>'} and {tool} on PATH)")


nm = find_llvm_tool("llvm-nm", "LLVM_NM")
try:
    objdump = find_llvm_tool("llvm-objdump", "LLVM_OBJDUMP")
except SystemExit:
    objdump = None   # then COFF COMDAT definitions count as strong
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


def nm_members(path):
    """{member: (defs, undefs, weak defs)} of an object (one member, its own
    name) or an archive."""
    out = subprocess.run([nm, "-P", "--extern-only", path], capture_output=True, text=True).stdout
    members = {}
    cur = os.path.basename(path)
    for line in out.splitlines():
        p = line.split()
        if len(p) == 1 and p[0].endswith(":"):
            cur = p[0][:-1]
            m = re.match(r".*\[(.*)\]$", cur)  # 'lib.a[member.o]'
            if m:
                cur = m.group(1)
            members.setdefault(cur, (set(), set(), set()))
            continue
        if len(p) < 2:
            continue
        name = p[0]
        if sys.platform == "darwin" and not args.target and name.startswith("_"):
            name = name[1:]
        d, u, w = members.setdefault(cur, (set(), set(), set()))
        if p[1] in ("U", "w", "v"):
            u.add(name)
        else:
            d.add(name)
            if p[1] in ("W", "V"):
                w.add(name)
    return members


def linker_merged(path):
    """{member: names} of definitions the linker keeps one copy of, though
    'nm -P' prints them as plain 'T'/'D': Mach-O weak definitions ('weak
    external' in 'nm -m') and COFF definitions in a pick-any COMDAT section
    (both what a linkonce_odr generic instance becomes; ELF's show as 'W')."""
    merged = {}
    cur = os.path.basename(path)
    out = subprocess.run([nm, "-m", "--extern-only", path], capture_output=True, text=True).stdout
    for line in out.splitlines():
        m = re.match(r"^(?:.*\((.*)\)|(\S+)):$", line)   # 'lib.a(member.o):' or 'file.o:'
        if m:
            cur = m.group(1) or os.path.basename(m.group(2))
            continue
        if "weak external" in line:
            name = line.split()[-1]
            if sys.platform == "darwin" and not args.target and name.startswith("_"):
                name = name[1:]
            elif args.target and "apple" in args.target and name.startswith("_"):
                name = name[1:]
            merged.setdefault(cur, set()).add(name)
    if objdump:
        out = subprocess.run([objdump, "-t", path], capture_output=True, text=True).stdout
        cur, comdat_secs, last_sec = os.path.basename(path), set(), None
        for line in out.splitlines():
            m = re.match(r"^(?:.*\((.*)\)|(\S+?)):\s+file format coff", line)
            if m:
                cur, comdat_secs, last_sec = m.group(1) or os.path.basename(m.group(2)), set(), None
                continue
            m = re.match(r"^\[\s*\d+\]\(sec\s+(-?\d+)\).*\(scl\s+(\d+)\).*\s(\S+)$", line)
            if m:
                sec, scl, name = int(m.group(1)), int(m.group(2)), m.group(3)
                last_sec = sec if scl == 3 else None          # a section-definition symbol
                if scl == 2 and sec in comdat_secs:           # external, in a COMDAT section
                    merged.setdefault(cur, set()).add(name)
                continue
            m = re.match(r"^AUX .*\bcomdat\s+(\d+)", line)
            if m and last_sec is not None and m.group(1) == "2":   # IMAGE_COMDAT_SELECT_ANY
                comdat_secs.add(last_sec)
    return merged


def nm_syms(path):
    defs, undef = set(), set()
    for d, u, _ in nm_members(path).values():
        defs |= d
        undef |= u
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


def section(st):  # x_y_sc -> x (by the unit's path, kept below)
    return unit_section.get(st, "core")


# The prebuilt std's sections.
unit_section = {}           # stem -> section, for every std source
for d, _, fs in os.walk(std):
    for f in fs:
        if f.endswith(".sc"):
            rel = os.path.relpath(os.path.join(d, f), std).replace(os.sep, "/")
            unit_section[stem(rel)] = rel.split("/")[0] if "/" in rel else "core"
sec_defs = {}               # section -> symbols its archive defines
sec_strong = {}
if args.prebuilt:
    pre_units, pre_skips = {}, set()
    for f in sorted(os.listdir(args.prebuilt)):
        if not f.endswith(".units"):
            continue
        sec = f[:-6]
        listed = set()
        for line in open(os.path.join(args.prebuilt, f)):
            w = line.split()
            if len(w) >= 2 and w[0] in ("unit", "skip"):
                st = stem(w[1])
                if unit_section.get(st) != sec:
                    errors.append(f"prebuilt {f}: lists std/{w[1]}, which is in section "
                                  f"'{unit_section.get(st, '?')}'")
                if st in pre_units:
                    errors.append(f"prebuilt {f}: std/{w[1]} is also in {pre_units[st]}.units")
                pre_units[st] = sec
                if w[0] == "skip":
                    pre_skips.add(st)
                else:
                    listed.add(st)
        archive = os.path.join(args.prebuilt, sec + ".a")
        members = nm_members(archive) if os.path.exists(archive) else {}
        if listed and not members:
            errors.append(f"prebuilt section '{sec}': {sec}.a is missing or empty")
        got = {m[:-2] for m in members if m.endswith(".o")}
        for st in sorted(listed - got):
            errors.append(f"prebuilt {sec}.a: no member {st}.o")
        for st in sorted(got - listed):
            errors.append(f"prebuilt {sec}.a: member {st}.o isn't in {sec}.units")
        # sec_defs: everything the section defines, weak definitions included
        # (what a build linking it gets); sec_strong: the definitions that may
        # appear only once across sections.
        sec_defs[sec] = {}
        sec_strong[sec] = {}
        merged = linker_merged(archive) if os.path.exists(archive) else {}
        for m, (d, _, w) in members.items():
            w = w | merged.get(m, set())
            for sym in d:
                sec_defs[sec].setdefault(sym, m)
            for sym in d - w:
                if not MERGEABLE.match(sym):
                    sec_strong[sec][sym] = m
    # Every std unit no feature gates is in exactly one section.
    want = {st for st in objs if st not in gated_stems}
    for st in sorted(want - set(pre_units)):
        errors.append(f"prebuilt: {st} (section '{section(st)}') is in no section's .units")
    for st in sorted(set(pre_units) - want - pre_skips):
        if st not in gated_stems:
            errors.append(f"prebuilt: {pre_units[st]}.units lists {st}, which isn't a std unit here")
    seen = {}
    for sec, ds in sorted(sec_strong.items()):
        for sym, m in ds.items():
            if sym in seen:
                errors.append(f"prebuilt: {sym} is defined in {seen[sym]} and {sec}.a({m})")
            else:
                seen[sym] = f"{sec}.a({m})"

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
        if args.prebuilt and passname == "no features":
            # A build linking only the sections holding its modules' units.
            secs = {section(u) for u in units if u not in gated_stems}
            have = set()
            for sec in secs:
                have |= set(sec_defs.get(sec, {}))
            for s in sorted(need):
                ds = definers.get(s)
                if not ds or s in have or ds <= gated_stems:
                    continue
                errors.append(f"std/{h} (prebuilt): {s} is defined by {', '.join(sorted(ds))}, "
                              f"not in the sections it links ({', '.join(sorted(secs))})")

print(f"std module check: {len(headers)} headers, {len(objs)} std objects, "
      f"{len(skipped)} headers that don't compile alone for this target skipped"
      + (f", {len(sec_defs)} prebuilt sections" if args.prebuilt else ""))
if skipped:
    print("  skipped: " + " ".join(skipped))
for e in errors:
    print("  " + e)
if not args.keep:
    shutil.rmtree(work, ignore_errors=True)
sys.exit(1 if errors else 0)
