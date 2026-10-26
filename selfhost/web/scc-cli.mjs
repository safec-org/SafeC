#!/usr/bin/env node
// scc-cli.mjs — the playground path from the command line: compiles with
// scc.wasm (through scc.js) and optionally runs the result.
//   node scc-cli.mjs [--scc path/to/scc.wasm] file.sc [-o out.wasm] [--run [args...]]
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { init, compile, run } from './scc.js';

const argv = process.argv.slice(2);
let sccPath = new URL('./scc.wasm', import.meta.url);
let input = null, out = null, doRun = false, runArgs = [], check = false;
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === '--scc') sccPath = pathToFileURL(argv[++i]);
  else if (argv[i] === '-o') out = argv[++i];
  else if (argv[i] === '--check') check = true;
  else if (argv[i] === '--run') { doRun = true; runArgs = argv.slice(i + 1); break; }
  else input = argv[i];
}
if (!input) { process.stderr.write('usage: scc-cli.mjs [--scc scc.wasm] file.sc [-o out.wasm] [--run args...]\n'); process.exit(2); }
await init(sccPath);
const src = readFileSync(input, 'utf8');
const t0 = performance.now();
const r = await compile(src, { check });
const ms = performance.now() - t0;
process.stderr.write(r.text.replaceAll('main.sc:', input + ':'));
if (process.env.SCC_TIME) process.stderr.write(`compile: ${ms.toFixed(1)} ms\n`);
if (!r.ok) process.exit(1);
if (out && r.wasm) writeFileSync(out, r.wasm);
if (doRun) {
  const code = await run(r.wasm, {
    stdout: (s) => process.stdout.write(s),
    stderr: (s) => process.stderr.write(s),
    args: runArgs,
    stdin: process.stdin.isTTY ? '' : readFileSync(0, 'utf8'),
  });
  process.exit(code);
}
