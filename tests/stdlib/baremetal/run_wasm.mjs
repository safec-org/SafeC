// WebAssembly counterpart of run.py: instantiates the test image (no
// imports — bm_rt.sc supplies the runtime), calls main(), and reads the
// result globals bm_main.sc leaves in linear memory. Each exported data
// symbol is a wasm global holding that symbol's address.
//   usage: node run_wasm.mjs <alias> <image.wasm>
import { readFileSync } from 'node:fs';

const [alias, path] = process.argv.slice(2);
const pad = (s, n) => s.padEnd(n);
let err = null;
let inst;
try {
    ({ instance: inst } = await WebAssembly.instantiate(readFileSync(path), {}));
    // wasm-ld wraps a no-argument main in main(argc, argv); on wasm64 argv
    // is an i64, which JS must pass as a BigInt.
    const main = inst.exports.main;
    main(...(main.length === 2 ? [0, alias === 'wasm64' ? 0n : 0] : []));
} catch (e) {
    err = e;
}

const read32 = (name) => {
    const g = inst?.exports[name];
    if (!g) return 0;
    const addr = typeof g.value === 'bigint' ? Number(g.value) : g.value;
    return new DataView(inst.exports.memory.buffer).getInt32(addr, true);
};
const [done, fails, checks, line, aborted] =
    ['g_done', 'g_fails', 'g_checks', 'g_first_fail_line', 'g_aborted'].map(read32);
const ok = err === null && done === 1 && fails === 0;
let detail = `checks=${checks} fails=${fails}`;
if (fails) detail += ` first-fail=bm_main.sc:${line}`;
if (aborted) detail += ' (abort() called)';
if (err !== null) detail += ` wasm: ${err.message ?? err}`;
else if (done !== 1) detail += ' (main did not finish)';
console.log(`${pad(alias, 8)} ${pad(ok ? 'ok' : 'FAILED', 6)} ${detail}`);
process.exit(ok ? 0 : 1);
