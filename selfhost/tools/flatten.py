#!/usr/bin/env python3
"""Inlines #include "..." (honoring #pragma once) to make one self-contained
source, e.g. the whole compiler for the bootstrap check:
    flatten.py src/scc_unity.sc > scc_flat.sc"""
import os, re, sys
seen = set()
def emit(path, out):
    path = os.path.normpath(path)
    text = open(path, encoding="utf-8").read()
    if "#pragma once" in text:
        if path in seen: return
        seen.add(path)
    for line in text.splitlines():
        m = re.match(r'\s*#\s*include\s+"([^"]+)"', line)
        if m:
            emit(os.path.join(os.path.dirname(path), m.group(1)), out)
        elif re.match(r"\s*#\s*pragma\s+once", line):
            continue
        else:
            out.append(line)
out = []
emit(sys.argv[1], out)
sys.stdout.write("\n".join(out) + "\n")
