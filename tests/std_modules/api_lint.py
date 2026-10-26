#!/usr/bin/env python3
"""Lints the public std API surface for C-style raw pointers.

SafeC's std is written in SafeC, not C: a public declaration takes and
returns references (&T, ?&T), slices ([]T), fn types, optionals and Results,
never 'void*' (or '?&void') or a raw 'T*'. Type erasure that a generic wrapper genuinely
needs stays behind a std-private name. This walks every std header
(std/**/*.h), collects the declarations at namespace and struct scope
(function signatures, struct fields, typedefs, globals -- not the bodies of
functions defined in the header), and fails on any raw pointer in one,
except:

  * 'extern' declarations of real C/OS/JS functions (FFI);
  * std-private names: functions, methods and typedefs whose name ends in
    '_raw_' (the type-erased half of a generic wrapper), 'static'
    header-local helpers whose name starts with '__', any function whose
    name ends in '_' (std's cross-file internals, e.g. bn_add_), and the
    members of structs so named;
  * internal headers: a header whose leading comment says it is internal
    ("Internal:", "std-private", "Not a public header", "do not edit"),
    and the FFI binding headers listed in INTERNAL below.

Usage: api_lint.py [--list] [SAFEC_HOME]
  --list  print every finding, also in allowed places (the audit view)
Exit status 1 when a public declaration has a raw pointer.
"""
import os
import re
import sys

LIST = "--list" in sys.argv
argv = [a for a in sys.argv[1:] if not a.startswith("--")]
home = argv[0] if argv else os.environ.get("SAFEC_HOME") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
std = os.path.join(home, "std")

# Headers that are bindings of a foreign C/JS API or freestanding libc
# replacements rather than SafeC API: their declarations mirror the C
# headers they stand in for.
INTERNAL = {
    "std/ml/webgpu_api.h",      # webgpu.h, declared for SafeC (FFI)
    "std/wasm/wasm_rt.h",       # the freestanding wasm32 libc (malloc, memcpy, ...)
    "std/stddef.h",
    "std/stdint.h",
    "std/inttypes.h",
    "std/errno_compat.h",
    "std/stderr_compat.h",
}
INTERNAL_MARK = re.compile(r"^//\s*(Internal: included|.*\bstd-private\b|.*\bNot a public header\b|.*\bdo not edit\b)", re.I | re.M)

POINTER = re.compile(r"(\bvoid|\w|>|\))\s*\*+(?!\s*\*)")
# '&void' / '?&void' is void* by another spelling.
VOIDREF = re.compile(r"&\s*(const\s+|volatile\s+)?void\b")


def strip(src):
    """Comments, strings and preprocessor lines blanked, newlines kept."""
    out = []
    i, n = 0, len(src)
    line_start = True
    while i < n:
        c = src[i]
        if line_start and c in " \t":
            out.append(c); i += 1; continue
        if line_start and c == "#":
            j = i
            while j < n and src[j] != "\n":
                if src[j] == "\\" and j + 1 < n and src[j + 1] == "\n":
                    out.append("\n"); j += 2; continue
                j += 1
            i = j
            continue
        line_start = False
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("\n" * src.count("\n", i, j))
            i = j
            continue
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append(c + c)
            i = j + 1
            continue
        if c == "\n":
            line_start = True
        out.append(c)
        i += 1
    return "".join(out)


def declarations(text):
    """(line, decl text) for every declaration at namespace/struct scope."""
    decls = []
    stack = []          # 'ns' | 'agg' | 'skip'
    cur = []
    cur_line = 1
    line = 1
    depth_paren = 0
    for c in text:
        if c == "\n":
            line += 1
        skipping = bool(stack) and stack[-1] == "skip"
        if skipping:
            if c == "{":
                stack.append("skip")
            elif c == "}":
                stack.pop()
            continue
        if not "".join(cur).strip():
            cur_line = line
        if c == "(":
            depth_paren += 1
        elif c == ")":
            depth_paren -= 1
        if c == "{" and depth_paren == 0:
            head = "".join(cur)
            h = head.strip()
            if re.search(r"\bnamespace\b|^extern\s*\"\"", h) or (h.startswith("extern") and h.endswith('""')):
                stack.append("ns")
            elif re.search(r"\b(struct|union|enum|trait)\b", h) and not re.search(r"\)\s*(const)?\s*(__attribute__.*)?$", h) and "=" not in h:
                m = re.search(r"\b(?:struct|union|enum|trait)\s+(\w+)", h)
                # A std-private aggregate ('__TensorEwJob', 'AudioHost_'):
                # its members aren't API.
                private = bool(m) and (m.group(1).startswith("__") or m.group(1).endswith("_"))
                stack.append("skip" if private else "agg")
                decls.append((cur_line, h + " {"))
            elif "=" in h:
                stack.append("skip")      # an initializer
                decls.append((cur_line, h))
            else:
                stack.append("skip")      # a function body
                decls.append((cur_line, h))
            cur = []
            continue
        if c == "}" and depth_paren == 0:
            if stack:
                stack.pop()
            cur = []
            continue
        if c == ";" and depth_paren == 0:
            h = "".join(cur).strip()
            if h:
                decls.append((cur_line, h))
            cur = []
            continue
        cur.append(c)
    return decls


def strip_fn_types(d):
    """'fn R(args)' types replaced by FN, so the declared name is findable."""
    while True:
        m = re.search(r"\bfn\b", d)
        if not m:
            return d
        j = d.find("(", m.end())
        if j < 0:
            return d
        depth = 0
        k = j
        while k < len(d):
            if d[k] == "(":
                depth += 1
            elif d[k] == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        d = d[:m.start()] + " FN " + d[k + 1:]


def decl_name(d):
    s = strip_fn_types(d)
    s = re.sub(r"__attribute__\s*\(\(.*?\)\)", "", s)
    s = re.sub(r"<[^<>]*>", "", re.sub(r"<[^<>]*>", "", s))
    m = re.search(r"([A-Za-z_]\w*(?:::[A-Za-z_~]\w*|::operator\S+)?)\s*\(", s)
    if m and not s.lstrip().startswith("typedef"):
        return m.group(1).split("::")[-1], "fn"
    s = re.sub(r"\[.*?\]", "", s.split("=")[0])
    ids = re.findall(r"[A-Za-z_]\w*", s)
    return (ids[-1] if ids else ""), ("typedef" if s.lstrip().startswith("typedef") else "field")


def classify(rel, d):
    """None if a raw pointer here is fine, else the reason it is not."""
    d0 = re.sub(r"\boperator\s*\*", "operator_mul", d)
    d0 = re.sub(r"\[[^\]]*\]", "[]", d0)          # array sizes like [A * B]
    if not POINTER.search(strip_fn_types(d0)) and not re.search(r"\bvoid\s*\*", d0) \
            and not VOIDREF.search(d0):
        return None
    name, kind = decl_name(d)
    if re.match(r"\s*extern\b", d):
        return "ffi-extern"
    if name.endswith("_raw_"):
        return "raw-wrapper"
    if re.match(r"\s*(generic\s*<[^>]*>\s*)?(static|inline|static\s+inline)\b", d) and (name.startswith("__") or name.endswith("_")):
        return "static-helper"
    if name.endswith("_"):
        # std's naming convention for cross-file internals (bn_add_, asn1_view_):
        # a trailing underscore is never public API.
        return "internal-helper"
    return "PUBLIC"


def main():
    bad = []
    allowed = 0
    for dirpath, _, files in os.walk(std):
        for f in sorted(files):
            if not f.endswith(".h"):
                continue
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, home).replace(os.sep, "/")
            src = open(path, encoding="utf-8", errors="replace").read()
            internal = rel in INTERNAL or bool(INTERNAL_MARK.search(src[:2500]))
            for line, d in declarations(strip(src)):
                why = classify(rel, d)
                if why is None:
                    continue
                if internal:
                    why = "internal-header" if why == "PUBLIC" else why
                one = " ".join(d.split())
                if why == "PUBLIC":
                    bad.append(f"{rel}:{line}: {one}")
                else:
                    allowed += 1
                    if LIST:
                        print(f"[{why}] {rel}:{line}: {one}")
    for b in bad:
        print(b)
    if bad:
        print(f"api_lint: {len(bad)} public std declaration(s) expose a raw pointer "
              f"(use &T / ?&T / []T / fn types; keep erasure in *_raw_ functions)", file=sys.stderr)
        return 1
    print(f"api_lint: no raw pointers in public std declarations ({allowed} allowed: FFI externs, *_raw_, internal headers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
