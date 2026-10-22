#!/usr/bin/env python3
"""Checks that Linux binaries and objects run on (and link with) an old glibc.

SafeC's Linux release binaries must run on glibc 2.25 or newer, and the
prebuilt std objects shipped with them must link there.

Binaries (safec, safeguard, sc-lsp): every versioned symbol they import is
read with 'objdump -T'; a GLIBC_x.y newer than --max (default 2.25) fails, and
so does a GLIBCXX_/CXXABI_ newer than gcc 6's (3.4.22 / 1.3.10), which a
dynamically linked libstdc++ would bring.

Objects (--objects; .o and .a, e.g. lib/<triple>/std/*.a): they are not
linked yet, so their undefined symbols carry no version. Each is looked up
in the glibc libraries given with --libc (default: this system's libc.so.6,
libm, libpthread, libdl, librt, libresolv, libutil): a symbol whose oldest
version there is newer than --max isn't in that glibc at all, and a link
against it would fail. Weak references are listed, not failed: they resolve
to null where the symbol is missing.

Usage:
  check-glibc.py [--max 2.25] <binary>...
  check-glibc.py [--max 2.25] --objects [--libc <libc.so.6>...] <file.o|file.a>...
Env: OBJDUMP, NM (default: objdump/nm, else llvm-objdump/llvm-nm).
Exits 1 listing the offenders, 0 when there are none.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--max", default="2.25", help="newest GLIBC version allowed")
ap.add_argument("--max-glibcxx", default="3.4.22")
ap.add_argument("--max-cxxabi", default="1.3.10")
ap.add_argument("--objects", action="store_true", help="check unlinked objects/archives")
ap.add_argument("--libc", action="append", default=[], help="glibc shared libraries to look symbols up in")
ap.add_argument("files", nargs="+")
args = ap.parse_args()


def tool(env, names):
    if os.environ.get(env):
        return os.environ[env]
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    sys.exit(f"check-glibc.py: none of {', '.join(names)} found (set {env})")


def ver(s):
    return tuple(int(x) for x in s.split("."))


limits = {"GLIBC": ver(args.max), "GLIBCXX": ver(args.max_glibcxx), "CXXABI": ver(args.max_cxxabi)}
VER = re.compile(r"\(?(GLIBC|GLIBCXX|CXXABI)_([0-9][0-9.]*)\)?$")


def dynsyms(path):
    """[(name, family, version, undefined)] from objdump -T."""
    objdump = tool("OBJDUMP", ["objdump", "llvm-objdump"])
    r = subprocess.run([objdump, "-T", path], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"check-glibc.py: {objdump} -T {path} failed:\n{r.stderr}")
    out = []
    for line in r.stdout.splitlines():
        w = line.split()
        if len(w) < 4:
            continue
        for i, t in enumerate(w):
            m = VER.match(t)
            if m:
                name = w[-1]
                if name == t or i == len(w) - 1:
                    break
                out.append((name, m.group(1), ver(m.group(2)), "*UND*" in w))
                break
    return out


failed = False

if not args.objects:
    for f in args.files:
        bad = {}
        newest = {}
        for name, fam, v, und in dynsyms(f):
            if not und:
                continue
            newest[fam] = max(newest.get(fam, ()), v)
            if v > limits[fam]:
                bad.setdefault(f"{fam}_{'.'.join(map(str, v))}", []).append(name)
        summary = ", ".join(f"{k}_{'.'.join(map(str, v))}" for k, v in sorted(newest.items())) or "no versioned imports"
        if bad:
            failed = True
            print(f"{f}: needs a newer glibc than {args.max} (newest: {summary})")
            for k in sorted(bad, key=lambda k: ver(k.split("_")[-1])):
                print(f"  {k}: {' '.join(sorted(set(bad[k])))}")
        else:
            print(f"{f}: ok (newest: {summary})")
    sys.exit(1 if failed else 0)

# --objects: each undefined symbol's oldest version in the given glibc.
libs = args.libc
if not libs:
    for d in ("/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu", "/lib/aarch64-linux-gnu",
              "/usr/lib/aarch64-linux-gnu", "/lib64", "/usr/lib64"):
        for n in ("libc.so.6", "libm.so.6", "libpthread.so.0", "libdl.so.2", "librt.so.1",
                  "libresolv.so.2", "libutil.so.1"):
            p = os.path.join(d, n)
            if os.path.exists(p) and not any(os.path.basename(x) == n for x in libs):
                libs.append(p)
if not any(os.path.basename(l).startswith("libc.so") for l in libs):
    sys.exit("check-glibc.py: no libc.so.6 to look symbols up in (give it with --libc)")
oldest = {}   # symbol -> oldest GLIBC version defining it
for l in libs:
    for name, fam, v, und in dynsyms(l):
        if fam == "GLIBC" and not und and (name not in oldest or v < oldest[name]):
            oldest[name] = v

nm = tool("NM", ["nm", "llvm-nm"])
defined, undef, weak = set(), {}, {}
for f in args.files:
    for path in glob.glob(f) or [f]:
        r = subprocess.run([nm, "-P", path], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"check-glibc.py: {nm} -P {path} failed:\n{r.stderr}")
        member = os.path.basename(path)
        for line in r.stdout.splitlines():
            w = line.split()
            if len(w) == 1 and w[0].endswith(":"):
                member = w[0][:-1]
                continue
            if len(w) < 2:
                continue
            name, kind = w[0], w[1]
            where = f"{os.path.basename(path)}({member})" if path.endswith(".a") else os.path.basename(path)
            if kind == "U":
                undef.setdefault(name, set()).add(where)
            elif kind in ("w", "v"):
                weak.setdefault(name, set()).add(where)
            else:
                defined.add(name)

bad, weak_new = [], []
for name, users in sorted(undef.items()):
    v = oldest.get(name)
    if name not in defined and v and v > limits["GLIBC"]:
        bad.append((name, v, users))
for name, users in sorted(weak.items()):
    v = oldest.get(name)
    if name not in defined and v and v > limits["GLIBC"]:
        weak_new.append((name, v, users))
used = [oldest[n] for n in undef if n in oldest and n not in defined]
newest = ".".join(map(str, max(used))) if used else "none"
print(f"{len(args.files)} input(s): {len(undef)} undefined symbols, {len(used)} from glibc "
      f"(looked up in {', '.join(libs)}); the newest a strong reference needs: GLIBC_{newest}")
for name, v, users in weak_new:
    print(f"  weak (null where missing): {name} GLIBC_{'.'.join(map(str, v))}: {' '.join(sorted(users))}")
for name, v, users in bad:
    print(f"  NEEDS GLIBC_{'.'.join(map(str, v))}: {name}: {' '.join(sorted(users))}")
if bad:
    print(f"{len(bad)} symbol(s) not in glibc {args.max}")
sys.exit(1 if bad else 0)
