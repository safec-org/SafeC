#!/usr/bin/env python3
"""Extracts complete programs (```c blocks defining main) from the SafeC docs
(book/ and reference/ pages) into tests/docs/<page>_<n>.sc.
   usage: extract_docs.py <safec-docs checkout>"""
import os, re, sys
docs = sys.argv[1]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
os.makedirs(out, exist_ok=True)
for f in os.listdir(out):
    if f.endswith(".sc"): os.remove(os.path.join(out, f))
n = 0
for sub in ("book", "reference"):
    d = os.path.join(docs, sub)
    for page in sorted(os.listdir(d)):
        if not page.endswith(".md"): continue
        text = open(os.path.join(d, page), encoding="utf-8").read()
        blocks = re.findall(r"```c\n(.*?)```", text, re.S)
        k = 0
        for b in blocks:
            if not re.search(r"\bint\s+main\s*\(", b): continue
            k += 1
            name = "%s_%s_%d.sc" % (sub, page[:-3].replace("-", "_"), k)
            with open(os.path.join(out, name), "w") as fh:
                fh.write("// extracted from safec-docs %s/%s (block %d)\n" % (sub, page, k))
                fh.write(b)
            n += 1
print("extracted %d programs" % n)
