#!/usr/bin/env bash
# Builds and runs the MPS (Metal) backend checks (needs an Apple GPU and
# Homebrew LLVM's clang, which safec's IR targets): attention, nn_mps.h's
# training against nn.h's, and train.h's mps feature against the CPU build.
#   usage: run.sh
#   Env: SAFEC (default compiler/build/safec), SAFEGUARD (default: on PATH),
#        OUT (build dir), STD_ARCHIVE (a prebuilt release libsafec_std.a).
# Links the release std archive 'safeguard build --release' makes for this
# checkout (rebuilt incrementally on every run, so it follows std edits)
# rather than listing std sources by hand: the checks reach std/ml, whose
# source set changes as the library grows.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SAFEC="${SAFEC:-$ROOT/compiler/build/safec}"
SAFEGUARD="${SAFEGUARD:-safeguard}"
OUT="${OUT:-$HERE/build}"
mkdir -p "$OUT"
CLANG=clang
for c in /opt/homebrew/opt/llvm/bin/clang /usr/local/opt/llvm/bin/clang; do
    if [[ -x "$c" ]]; then CLANG="$c"; break; fi
done

STD="${STD_ARCHIVE:-}"
if [[ -z "$STD" ]]; then
    P="$OUT/_std"
    mkdir -p "$P/src"
    printf '[package]\nname = "mps_checks_std"\nversion = "0.1.0"\n\n[build]\nedition = "2026"\n' > "$P/Package.toml"
    echo 'int main() { return 0; }' > "$P/src/main.sc"
    (cd "$P" && SAFEC="$SAFEC" SAFEC_HOME="$ROOT" "$SAFEGUARD" build --release >/dev/null)
    STD="$P/build/deps/libsafec_std.a"
fi

WRAP="$OUT/clang_link.sh"
printf '#!/bin/sh\nexec "%s" -w "$@" "%s" -framework Metal -framework Foundation -framework Accelerate -lobjc -lm -lpthread\n' \
    "$CLANG" "$STD" > "$WRAP"
chmod +x "$WRAP"
for t in mps_attention_check nn_mps_check; do
    SAFEC_CLANG="$WRAP" "$SAFEC" "$HERE/$t.sc" -I "$ROOT" --emit-bin -O2 -o "$OUT/$t"
    "$OUT/$t"
done

# One training program, two backends (as rocm/run.sh does for rocm):
# rocm/train_backend_check.sc built for the CPU and with the mps feature
# (SAFEC_FEATURE_MPS: train.h trains MlpTrainer on the GPU, the other
# trainers on the CPU), and the two runs' RESULT lines must agree.
TB="$ROOT/tests/stdlib/rocm/train_backend_check.sc"
SAFEC_CLANG="$WRAP" "$SAFEC" "$TB" -I "$ROOT" --emit-bin -O2 -o "$OUT/train_backend_cpu"
SAFEC_CLANG="$WRAP" "$SAFEC" "$TB" -I "$ROOT" -DSAFEC_FEATURE_MPS=1 --emit-bin -O2 -o "$OUT/train_backend_mps"
"$OUT/train_backend_cpu" > "$OUT/train_backend_cpu.txt"
"$OUT/train_backend_mps" | tee "$OUT/train_backend_mps.txt"
# value b (mps) against a (cpu): |a - b| <= tol * (1 + |a|), every name present in both
awk '$1 == "RESULT" { if (FILENAME == ARGV[1]) { a[$2] = $3; tol[$2] = $4 } else b[$2] = $3 }
     END {
         bad = 0; n = 0
         for (k in a) {
             n++
             if (!(k in b)) { printf "  FAIL %s: missing from the mps run\n", k; bad++; continue }
             d = a[k] - b[k]; if (d < 0) d = -d
             m = a[k] < 0 ? -a[k] : a[k]
             if (d > tol[k] * (1 + m)) { printf "  FAIL %s: cpu %s mps %s (tolerance %s)\n", k, a[k], b[k], tol[k]; bad++ }
         }
         printf "train_backend_check: cpu vs mps, %d values, %d differ\n", n, bad
         exit bad > 0 || n == 0
     }' "$OUT/train_backend_cpu.txt" "$OUT/train_backend_mps.txt"
