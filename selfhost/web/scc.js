// scc.js — the SafeC playground API (browser and Node 18+).
//
//   import { init, compile, run } from './scc.js';
//   await init();                                   // loads ./scc.wasm
//   const r = await compile(source);                // { ok, diagnostics, wasm }
//   if (r.ok) {
//     const code = await run(r.wasm, { stdout: s => out += s });
//   }
//
// compile() runs the SafeC compiler (scc, itself written in SafeC) compiled
// to WebAssembly; the program it produces is a WebAssembly module that
// run() executes in a Web Worker (worker_threads in Node) with a time and
// memory limit. See README.md for the full contract.

const isNode = typeof process !== 'undefined' && !!(process.versions && process.versions.node);
let sccModule = null;     // WebAssembly.Module of scc.wasm

async function readBytes(src) {
  if (src instanceof WebAssembly.Module) return src;
  if (src instanceof ArrayBuffer || ArrayBuffer.isView(src)) return src;
  const url = src instanceof URL ? src : new URL(String(src), import.meta.url);
  if (isNode && url.protocol === 'file:') {
    const { readFile } = await import('node:fs/promises');
    return readFile(url);
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`scc: cannot load ${url}: ${res.status}`);
  return res.arrayBuffer();
}

/** Loads the compiler. `wasm` is a URL/path, bytes, or a compiled Module
 *  (default: scc.wasm next to this file). Safe to call more than once. */
export async function init(wasm) {
  if (sccModule && wasm === undefined) return;
  const b = await readBytes(wasm ?? new URL('./scc.wasm', import.meta.url));
  sccModule = b instanceof WebAssembly.Module ? b : await WebAssembly.compile(b);
}

/** Parses scc's diagnostics text into objects. */
export function parseDiagnostics(text) {
  const out = [];
  for (const line of text.split('\n')) {
    if (!line) continue;
    const m = /^(.*?):(\d+):(\d+): (error|warning): (.*)$/.exec(line);
    if (m) out.push({ file: m[1], line: +m[2], col: +m[3], severity: m[4], message: m[5] });
    else {
      const g = /^(error|warning): (.*)$/.exec(line);
      out.push({ file: '', line: 0, col: 0, severity: g ? g[1] : 'error', message: g ? g[2] : line });
    }
  }
  return out;
}

/** Compiles one SafeC source file (named main.sc). Resolves to
 *  { ok: boolean, diagnostics: [{file,line,col,severity,message}],
 *    wasm: Uint8Array | null, text: string (raw diagnostics) }. */
export async function compile(source, options = {}) {
  if (!sccModule) await init();
  let memory = null;
  const imports = { env: { abort() { throw new Error('scc: internal error (abort)'); } } };
  const inst = await WebAssembly.instantiate(sccModule, imports);
  const ex = inst.exports;
  memory = ex.memory;
  const bytes = new TextEncoder().encode(String(source));
  const at = ex.scc_begin();
  if (bytes.length > ex.scc_capacity()) throw new Error('scc: source too large');
  new Uint8Array(memory.buffer, at, bytes.length).set(bytes);
  let ok = false;
  try {
    ok = ex.scc_compile(bytes.length) === 1;
  } catch (e) {
    const text = `error: internal compiler error: ${e && e.message || e}\n`;
    return { ok: false, diagnostics: parseDiagnostics(text), wasm: null, text };
  }
  const dlen = ex.scc_diagnostics_len();
  const text = new TextDecoder().decode(new Uint8Array(memory.buffer, ex.scc_diagnostics(), dlen));
  let wasm = null;
  if (ok) {
    const olen = ex.scc_output_len();
    wasm = new Uint8Array(memory.buffer, ex.scc_output(), olen).slice();
  }
  return { ok, diagnostics: parseDiagnostics(text), wasm, text };
}

/** Runs a compiled program in a worker. Options:
 *    stdout(text), stderr(text)  — output callbacks (called as output arrives)
 *    stdin: string               — the program's whole standard input
 *    args: string[]              — argv[1..] (argv[0] is "main")
 *    timeoutMs (default 10000), maxMemoryMb (default 256)
 *  Resolves to the exit code: main's return value or exit()'s argument;
 *  134 for a trap/abort (bounds check, assert, unreachable), 136 for an
 *  arithmetic trap (division by zero), 139 for stack overflow, 124 when
 *  the time limit is hit, 137 when the memory limit is hit. */
export function run(wasm, options = {}) {
  const timeoutMs = options.timeoutMs ?? 10000;
  const workerUrl = new URL('./worker.js', import.meta.url);
  const msg = {
    wasm: wasm instanceof Uint8Array ? wasm : new Uint8Array(wasm),
    stdin: options.stdin ?? '',
    args: ['main', ...(options.args ?? [])],
    maxMemoryMb: options.maxMemoryMb ?? 256,
  };
  const onOut = options.stdout ?? (() => {});
  const onErr = options.stderr ?? (() => {});
  return new Promise(async (resolve, reject) => {
    let worker;
    let done = false;
    const finish = (code) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      try { worker.terminate(); } catch {}
      resolve(code);
    };
    const onMessage = (m) => {
      if (m.type === 'stdout') onOut(m.data);
      else if (m.type === 'stderr') onErr(m.data);
      else if (m.type === 'exit') finish(m.code);
    };
    try {
      if (isNode) {
        const { Worker } = await import('node:worker_threads');
        worker = new Worker(workerUrl);
        worker.on('message', onMessage);
        worker.on('error', (e) => { onErr(String(e) + '\n'); finish(1); });
      } else {
        worker = new Worker(workerUrl, { type: 'module' });
        worker.onmessage = (e) => onMessage(e.data);
        worker.onerror = (e) => { onErr(String(e.message || e) + '\n'); finish(1); };
      }
    } catch (e) { reject(e); return; }
    var timer = setTimeout(() => { onErr('\n[time limit exceeded]\n'); finish(124); }, timeoutMs);
    worker.postMessage(msg);
  });
}
