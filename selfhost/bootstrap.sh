#!/usr/bin/env bash
# Bootstrap check: scc compiling itself.
#   stage1  build/scc            native scc, built by the C++ safec (build.sh)
#   stage2  build/scc2.wasm      scc's own source compiled by stage1
#   stage3  build/scc3.wasm      the same source compiled by stage2 under Node
# The check passes when stage2 and stage3 are byte-identical (a fixpoint),
# and, with --test, when stage2 passes the differential tests like stage1.
#   SAFEC=path/to/safec  NODE=node
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NODE="${NODE:-node}"
B="$HERE/build"
[ -x "$B/scc" ] || "$HERE/build.sh"
python3 "$HERE/tools/flatten.py" "$HERE/src/scc_unity.sc" > "$B/scc_flat.sc"
echo "scc source: $(wc -l < "$B/scc_flat.sc") lines (flattened)"
t0=$(date +%s.%N)
"$B/scc" "$B/scc_flat.sc" -o "$B/scc2.wasm"
t1=$(date +%s.%N)
"$NODE" "$HERE/web/scc-cli.mjs" --scc "$B/scc2.wasm" "$B/scc_flat.sc" -o "$B/scc3.wasm"
t2=$(date +%s.%N)
printf 'stage1 -> stage2: %.3fs, %d bytes\n' "$(echo "$t1 - $t0" | bc)" "$(stat -c %s "$B/scc2.wasm")"
printf 'stage2 -> stage3: %.3fs (Node), %d bytes\n' "$(echo "$t2 - $t1" | bc)" "$(stat -c %s "$B/scc3.wasm")"
if cmp -s "$B/scc2.wasm" "$B/scc3.wasm"; then
    echo "bootstrap fixpoint: stage2 == stage3"
else
    echo "bootstrap FAILED: stage2 and stage3 differ"; exit 1
fi
if [ "${1:-}" = "--test" ]; then
    python3 "$HERE/tests/difftest.py" --scc-wasm "$B/scc2.wasm"
fi
