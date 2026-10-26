#!/usr/bin/env node
// Runs a wasm module produced by scc under Node: node run-node.mjs prog.wasm [args...]
// Exit status: main's return value, the code passed to exit(), 134 for a
// trap/abort (bounds check, unreachable), 136 for an arithmetic trap.
import { readFileSync, writeSync, readSync } from 'node:fs';

class Exit extends Error { constructor(code) { super('exit'); this.code = code; } }

const bytes = readFileSync(process.argv[2]);
let memory = null;
const imports = {
  env: {
    __host_write(fd, ptr, len) {
      const b = new Uint8Array(memory.buffer, ptr, len);
      writeSync(fd === 2 ? 2 : 1, b);
      return len;
    },
    __host_read(fd, ptr, len) {
      try {
        const b = Buffer.alloc(len);
        const n = readSync(0, b, 0, len, null);
        new Uint8Array(memory.buffer, ptr, n).set(b.subarray(0, n));
        return n;
      } catch { return 0; }
    },
    __host_exit(code) { throw new Exit(code); },
  },
};
let code = 0;
try {
  const { instance } = await WebAssembly.instantiate(bytes, imports);
  memory = instance.exports.memory;
  const main = instance.exports.main;
  if (main.length >= 2) {
    // argv: pointer table followed by the strings, in the runtime's buffer
    const args = [process.env.SCC_ARGV0 || './prog', ...process.argv.slice(3)];
    const base = instance.exports.__rt_argv ? instance.exports.__rt_argv() : 0;
    const view = new DataView(memory.buffer);
    let at = base + 4 * (args.length + 1);
    args.forEach((a, i) => {
      const b = new TextEncoder().encode(a + '\0');
      new Uint8Array(memory.buffer, at, b.length).set(b);
      view.setInt32(base + 4 * i, at, true);
      at += b.length;
    });
    view.setInt32(base + 4 * args.length, 0, true);
    code = main(args.length, base);
  } else {
    code = main();
  }
} catch (e) {
  if (e instanceof Exit) code = e.code;
  else if (e instanceof WebAssembly.RuntimeError) {
    const m = String(e.message);
    code = /divide by zero|integer overflow/.test(m) ? 136 : (/call stack/.test(m) ? 139 : 134);
    process.stderr.write(`trap: ${m}\n`);
  } else { process.stderr.write(String(e && e.stack || e) + '\n'); code = 1; }
}
process.exit(code & 255);
