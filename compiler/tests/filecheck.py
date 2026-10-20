#!/usr/bin/env python3
"""Minimal FileCheck subset for SafeC's compiler tests (LLVM's FileCheck
isn't shipped in the release toolchains we build against).

Usage: filecheck.py <test.sc> <output.ll>

Directives, read from '// ' comments in the test source:
  CHECK-LABEL: pat   starts a new block at the next line matching pat; the
                     block ends where the next CHECK-LABEL matches.
  CHECK: pat         matches in order within the current block.
  CHECK-NOT: pat     must not match between the previous match and the next
                     CHECK (or the end of the block).
'pat' is literal text, except '{{re}}' spans which are regular expressions.

  EXPECT-ERROR: msg  the test must fail to compile with an error containing
                     msg: on that line when written as a trailing comment,
                     on any line when it's a comment line of its own (repeat
                     it once per expected occurrence). Checked with --errors
                     <stderr-file>; every reported error must be expected.
"""
import re
import sys


def compile_pattern(pat):
    out, pos = [], 0
    for m in re.finditer(r"\{\{(.*?)\}\}", pat):
        out.append(re.escape(pat[pos:m.start()]))
        out.append("(?:" + m.group(1) + ")")
        pos = m.end()
    out.append(re.escape(pat[pos:]))
    return re.compile("".join(out))


def check_errors(src, errfile):
    # (line or None, msg): a trailing directive is tied to its own line; a
    # whole-line one ('// EXPECT-ERROR: msg' alone) may match any line.
    expected = []
    for n, line in enumerate(open(src), 1):
        m = re.search(r"//\s*EXPECT-ERROR:\s?(.*)$", line)
        if m:
            anchored = line[:m.start()].strip() != ""
            expected.append((n if anchored else None, m.group(1).strip(), n))
    reported = []
    for line in open(errfile):
        m = re.search(r":(\d+):\d+: error: (.*)$", line)
        if m:
            reported.append((int(m.group(1)), m.group(2)))
    # Each reported error satisfies at most one directive: line-tied ones
    # first, so a free-floating directive can't steal their match.
    used = [False] * len(reported)
    failures = 0
    for want_line, msg, n in sorted(expected, key=lambda e: e[0] is None):
        hit = next((i for i, (rn, rmsg) in enumerate(reported)
                    if not used[i] and msg in rmsg and want_line in (None, rn)), None)
        if hit is None:
            print(f"{src}:{n}: expected error not reported: {msg}")
            failures += 1
        else:
            used[hit] = True
    for i, (rn, rmsg) in enumerate(reported):
        if not used[i]:
            print(f"{src}:{rn}: unexpected error: {rmsg}")
            failures += 1
    return 1 if failures else 0


def main():
    if sys.argv[1] == "--errors":
        return check_errors(sys.argv[2], sys.argv[3])
    src, ll = sys.argv[1], sys.argv[2]
    directives = []
    for n, line in enumerate(open(src), 1):
        m = re.match(r"\s*//\s*(CHECK(?:-LABEL|-NOT)?):\s?(.*)$", line)
        if m:
            directives.append((m.group(1), m.group(2).rstrip(), n))
    lines = open(ll).read().splitlines()

    # Split directives into label groups.
    groups, cur = [], None
    for d in directives:
        if d[0] == "CHECK-LABEL":
            cur = [d, []]
            groups.append(cur)
        elif cur is None:
            cur = [None, [d]]
            groups.append(cur)
        else:
            cur[1].append(d)

    # Locate each label in order.
    starts, pos = [], 0
    for label, _ in groups:
        if label is None:
            starts.append(pos)
            continue
        rx = compile_pattern(label[1])
        while pos < len(lines) and not rx.search(lines[pos]):
            pos += 1
        if pos == len(lines):
            print(f"{src}:{label[2]}: CHECK-LABEL not found: {label[1]}")
            return 1
        starts.append(pos)
        pos += 1

    failures = 0
    for gi, (label, checks) in enumerate(groups):
        begin = starts[gi] + (1 if label else 0)
        end = starts[gi + 1] if gi + 1 < len(groups) else len(lines)
        pos, pending_not = begin, []
        for kind, pat, n in checks:
            rx = compile_pattern(pat)
            if kind == "CHECK-NOT":
                pending_not.append((rx, pat, n))
                continue
            hit = next((i for i in range(pos, end) if rx.search(lines[i])), None)
            if hit is None:
                print(f"{src}:{n}: CHECK not found: {pat}")
                failures += 1
                break
            for nrx, npat, nn in pending_not:
                for i in range(pos, hit):
                    if nrx.search(lines[i]):
                        print(f"{src}:{nn}: CHECK-NOT matched: {npat}\n    {lines[i]}")
                        failures += 1
            pending_not, pos = [], hit + 1
        for nrx, npat, nn in pending_not:
            for i in range(pos, end):
                if nrx.search(lines[i]):
                    print(f"{src}:{nn}: CHECK-NOT matched: {npat}\n    {lines[i]}")
                    failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
