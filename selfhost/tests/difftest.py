#!/usr/bin/env python3
"""Differential tests: every program in tests/{docs,corpus,neg} is compiled by
the C++ safec (and run natively) and by scc (and the wasm run under Node);
accept/reject decisions, stdout and exit codes are compared.

   usage: difftest.py [--scc build/scc] [--scc-wasm build/scc.wasm] [-v] [files...]

A '// expect-error: <text>' line in a test (tests/neg/) additionally
requires scc's diagnostics to contain <text>.  '// stdin: <text>' feeds stdin.
--scc-wasm compiles with the wasm build of scc under Node (the playground
path) instead of the native one."""
import argparse, os, re, subprocess, sys, tempfile, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--scc", default=os.path.join(ROOT, "build", "scc"))
ap.add_argument("--scc-wasm", default=None)
ap.add_argument("--safec", default=os.environ.get("SAFEC", "safec"))
ap.add_argument("--node", default=os.environ.get("NODE", "node"))
ap.add_argument("-v", action="store_true")
ap.add_argument("--json", default=None, help="write per-test results here")
ap.add_argument("files", nargs="*")
args = ap.parse_args()

files = args.files or sorted(
    os.path.join(HERE, d, f) for d in ("docs", "corpus", "neg")
    for f in os.listdir(os.path.join(HERE, d)) if f.endswith(".sc"))
tmp = tempfile.mkdtemp(prefix="scc-diff-")
runner = os.path.join(ROOT, "web", "run-node.mjs")
cli = os.path.join(ROOT, "web", "scc-cli.mjs")

def run(cmd, stdin=None, timeout=20, cwd=None):
    try:
        p = subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout, cwd=cwd)
        return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return -999, "", "timeout"

def errors(text):
    return [l for l in text.splitlines() if ": error: " in l]

def first_error_msg(text):
    e = errors(text)
    return e[0].split(": error: ", 1)[1] if e else ""

stdlib = os.path.join(ROOT, "build", "native-std", "libsafec_std.a")
if os.environ.get("SAFEC_HOME") and not os.path.exists(stdlib):
    subprocess.run(["bash", os.path.join(HERE, "native_std.sh")])
results = []
counts = {"pass": 0, "fail": 0}
cats = {}
for f in files:
    name = os.path.relpath(f, HERE)
    src = open(f, encoding="utf-8").read()
    stdin = None
    m = re.search(r"^// stdin: (.*)$", src, re.M)
    if m: stdin = m.group(1).encode().decode("unicode_escape").encode()
    expect = re.findall(r"^// expect-error: (.*)$", src, re.M)
    base = os.path.join(tmp, os.path.basename(f)[:-3])
    # native
    pdir = base + ".d"
    os.makedirs(pdir, exist_ok=True)
    std = os.environ.get("SAFEC_HOME", "")
    cmd = [args.safec, f, "--compat-preprocessor", "--emit-bin", "-o", os.path.join(pdir, "prog")]
    if std: cmd += ["-I", std]
    rc, out, err = run(cmd + ["-lm"])
    if rc != 0 and "undefined reference to `std_" in err and os.path.exists(stdlib):
        # link the std library (as safeguard would)
        rc, out, err = run(cmd + ["-L", os.path.dirname(stdlib), "-lsafec_std", "-lm", "-lpthread"])
    n_ok = rc == 0 and os.path.exists(os.path.join(pdir, "prog"))
    n_diag = err
    n_rc = n_out = None
    if n_ok:
        n_rc, n_out, _ = run(["stdbuf", "-o0", "./prog"], stdin, cwd=pdir)
    # scc
    if args.scc_wasm:
        rc, out, err = run([args.node, cli, "--scc", args.scc_wasm, f, "-o", base + ".wasm"])
    else:
        rc, out, err = run([args.scc, f, "-o", base + ".wasm"])
    s_ok = rc == 0
    s_diag = err
    s_rc = s_out = None
    if s_ok:
        s_rc, s_out, s_err = run([args.node, runner, base + ".wasm"], stdin)
    # verdict
    why = ""
    if n_ok != s_ok:
        why = "accept/reject differs (safec %s, scc %s)" % ("ok" if n_ok else "rejects", "ok" if s_ok else "rejects")
    elif n_ok:
        if n_out != s_out: why = "stdout differs"
        elif n_rc != s_rc: why = "exit code differs (%s vs %s)" % (n_rc, s_rc)
    for x in expect:
        if x not in s_diag: why = why or ("missing expected error: " + x)
    same_first = (not n_ok and not s_ok and first_error_msg(n_diag) == first_error_msg(s_diag))
    ok = why == ""
    counts["pass" if ok else "fail"] += 1
    cat = name.split(os.sep)[0]
    c = cats.setdefault(cat, [0, 0, 0, 0])
    c[0] += 1; c[1] += ok
    if not n_ok and not s_ok: c[2] += 1; c[3] += same_first
    results.append({"test": name, "ok": ok, "why": why, "same_first_error": same_first})
    if not ok or args.v:
        print("%-4s %s %s" % ("ok" if ok else "FAIL", name, why))
        if not ok and args.v:
            if n_ok != s_ok:
                print("   safec: " + " | ".join(errors(n_diag)[:3]))
                print("   scc:   " + " | ".join(errors(s_diag)[:3]) + (" " + s_diag[:300] if not errors(s_diag) and not s_ok else ""))
            elif n_ok:
                print("   safec out: %r rc=%s" % (n_out[:300], n_rc))
                print("   scc   out: %r rc=%s" % (s_out[:300], s_rc))
for cat, (n, p, rej, same) in sorted(cats.items()):
    extra = " (both rejected %d, same first error %d)" % (rej, same) if rej else ""
    print("%-8s %d/%d pass%s" % (cat, p, n, extra))
print("TOTAL    %d/%d pass" % (counts["pass"], counts["pass"] + counts["fail"]))
if args.json:
    json.dump(results, open(args.json, "w"), indent=1)
sys.exit(0 if counts["fail"] == 0 else 1)
