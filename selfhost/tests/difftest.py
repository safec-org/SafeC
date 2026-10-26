#!/usr/bin/env python3
"""Differential tests: every program in tests/{docs,corpus,neg} and in
compiler/examples is compiled by
the C++ safec (and run natively) and by scc (and the wasm run under Node);
accept/reject decisions, stdout and exit codes are compared.

   usage: difftest.py [--scc build/scc] [--scc-wasm build/scc.wasm] [-v] [files...]

A '// expect-error: <text>' line in a test (tests/neg/) additionally
requires scc's diagnostics to contain <text>.  '// stdin: <text>' feeds stdin.
'// output: unordered' compares stdout as a multiset of lines (threads).
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
ap.add_argument("--no-ctests", action="store_true", help="skip compiler/tests")
ap.add_argument("files", nargs="*")
args = ap.parse_args()
CTESTS = os.path.join(os.path.dirname(ROOT), "compiler", "tests")

EXAMPLES = os.path.join(os.path.dirname(ROOT), "compiler", "examples")
files = args.files or (sorted(
    os.path.join(HERE, d, f) for d in ("docs", "corpus", "neg")
    for f in os.listdir(os.path.join(HERE, d)) if f.endswith(".sc")) +
    sorted(os.path.join(EXAMPLES, f) for f in os.listdir(EXAMPLES) if f.endswith(".sc")))
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
UNORDERED = {"examples/concurrency.sc"}
ctest_files = [] if (args.files or args.no_ctests) else sorted(
    os.path.join(d, f) for d, _, fs in os.walk(CTESTS) for f in fs if f.endswith(".sc"))
for f in files:
    name = os.path.relpath(f, HERE)
    if f.startswith(EXAMPLES + os.sep): name = os.path.join("examples", os.path.relpath(f, EXAMPLES))
    src = open(f, encoding="utf-8").read()
    stdin = None
    m = re.search(r"^// stdin: (.*)$", src, re.M)
    if m: stdin = m.group(1).encode().decode("unicode_escape").encode()
    expect = re.findall(r"^// expect-error: (.*)$", src, re.M)
    # threads print in any order natively (scc runs a spawned function to
    # completion right away): compare such output as a multiset of lines
    unordered = re.search(r"^// output: unordered", src, re.M) is not None or name in UNORDERED
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
        if unordered:
            if sorted(n_out.splitlines()) != sorted(s_out.splitlines()): why = "stdout differs (as a set of lines)"
        elif n_out != s_out: why = "stdout differs"
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
# compiler/tests: the C++ compiler's regression tests, compile-only.  Both
# compilers must agree on accept/reject; for '// EXPECT-ERROR:' tests scc's
# diagnostics are also checked against the directives (filecheck.py rules).
sys.path.insert(0, CTESTS)
if ctest_files:
    import io, contextlib, filecheck
for f in ctest_files:
    name = os.path.join("ctests", os.path.relpath(f, CTESTS))
    src = open(f, encoding="utf-8").read()
    neg = "EXPECT-ERROR:" in src
    m = re.search(r"^// SAFEC-ARGS: *(.*)$", src, re.M)
    # '%root' is the repository root (compiler/tests/run.sh expands it too)
    extra = m.group(1).replace("%root", os.path.dirname(ROOT)).split() if m else []
    rc, _, n_diag = run([args.safec, f, "--emit-llvm", "-o", os.path.join(tmp, "ct.ll")] + extra)
    n_ok = rc == 0
    if args.scc_wasm:
        rc, _, s_diag = run([args.node, cli, "--scc", args.scc_wasm, "--check", f])
    else:
        rc, _, s_diag = run([args.scc, "--check", f])
    s_ok = rc == 0
    why = ""
    exact = False
    if n_ok != s_ok:
        why = "accept/reject differs (safec %s, scc %s)" % ("ok" if n_ok else "rejects", "ok" if s_ok else "rejects")
    elif neg:
        ef = os.path.join(tmp, "ct.err")
        open(ef, "w").write(s_diag)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bad = filecheck.check_errors(f, ef)
        exact = not bad
    ok = why == ""
    counts["pass" if ok else "fail"] += 1
    c = cats.setdefault("ctests", [0, 0, 0, 0])
    c[0] += 1; c[1] += ok
    if neg and not n_ok and not s_ok: c[2] += 1; c[3] += exact
    results.append({"test": name, "ok": ok, "why": why, "expected_errors_match": exact if neg else None})
    if not ok or args.v:
        print("%-4s %s %s" % ("ok" if ok else "FAIL", name, why))
        if not ok and args.v:
            print("   safec: " + " | ".join(errors(n_diag)[:3]))
            print("   scc:   " + " | ".join(errors(s_diag)[:3]))
for cat, (n, p, rej, same) in sorted(cats.items()):
    if cat == "ctests":
        extra = " (EXPECT-ERROR tests rejected by both %d, scc matches every directive %d)" % (rej, same) if rej else ""
    else:
        extra = " (both rejected %d, same first error %d)" % (rej, same) if rej else ""
    print("%-8s %d/%d pass%s" % (cat, p, n, extra))
print("TOTAL    %d/%d pass" % (counts["pass"], counts["pass"] + counts["fail"]))
if args.json:
    json.dump(results, open(args.json, "w"), indent=1)
# known-failures.txt: tests that are known to differ, each with the reason.
# They still run and count as failures above; the exit status fails only on
# a failure not listed there, or on a listed test that now passes (so the
# list can only shrink).
known = {}
kf = os.path.join(HERE, "known-failures.txt")
if os.path.exists(kf):
    for line in open(kf, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if line: known[line.split()[0]] = True
ran = {r["test"]: r["ok"] for r in results}
new_fail = [t for t, ok in ran.items() if not ok and t not in known]
fixed = [t for t in known if ran.get(t) is True]
for t in new_fail: print("NEW FAILURE (not in known-failures.txt): " + t)
for t in fixed: print("NOW PASSES (remove it from known-failures.txt): " + t)
if known and not args.files:
    print("known failures: %d listed, %d still failing" % (len(known), sum(1 for t in known if ran.get(t) is False)))
sys.exit(1 if new_fail or fixed else 0)
