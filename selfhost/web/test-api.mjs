// test-api.mjs — checks the playground API contract (README.md) end to end:
//   node web/test-api.mjs [dir with scc.js, worker.js, exec.js, scc.wasm]
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const dir = path.resolve(process.argv[2] || new URL('.', import.meta.url).pathname);
const { init, compile, run } = await import(pathToFileURL(path.join(dir, 'scc.js')).href);
await init();

let failed = 0;
function check(name, cond, detail) {
  console.log(`${cond ? 'ok  ' : 'FAIL'} ${name}${cond ? '' : ': ' + detail}`);
  if (!cond) failed++;
}
async function runSrc(src, opts = {}) {
  const r = await compile(src);
  if (!r.ok) return { r, code: null, out: '', err: '' };
  let out = '', err = '';
  const code = await run(r.wasm, { stdout: s => out += s, stderr: s => err += s, ...opts });
  return { r, code, out, err };
}

{
  const t0 = performance.now();
  const { r, code, out } = await runSrc('#include <stdio.h>\nint main() { printf("hello %d\\n", 42); return 3; }\n');
  const ms = performance.now() - t0;
  check('compile ok', r.ok && r.wasm instanceof Uint8Array && r.diagnostics.length === 0, r.text);
  check('stdout', out === 'hello 42\n', JSON.stringify(out));
  check('exit code', code === 3, code);
  console.log(`     (compile + run: ${ms.toFixed(1)} ms)`);
}
{
  const r = await compile('int main() {\n    int x = "s";\n    return 0;\n}\n');
  const d = r.diagnostics[0] || {};
  check('error diagnostics', !r.ok && r.wasm === null && d.file === 'main.sc' && d.line === 2 && d.col > 0 &&
        d.severity === 'error' && /cannot/.test(d.message), JSON.stringify(r.diagnostics));
}
{
  const { code, out } = await runSrc('#include <stdio.h>\nint main() { char b[32]; if (fgets(b, 32, stdin) == null) return 1; printf("[%s]", b); return 0; }\n',
                                     { stdin: 'line\n' });
  check('stdin', code === 0 && out === '[line\n]', JSON.stringify(out));
}
{
  const { code } = await runSrc('int main() { int a[4]; int i = 7; a[i] = 1; return a[0]; }\n');
  check('bounds trap -> 134', code === 134, code);
}
{
  const { code, err } = await runSrc('int main() { while (1) { } return 0; }\n', { timeoutMs: 300 });
  check('time limit -> 124', code === 124, code + ' ' + err);
}
{
  const { code } = await runSrc('#include <stdlib.h>\nint main() { while (1) { if (malloc(1048576UL) == null) return 9; } return 0; }\n',
                                { maxMemoryMb: 32 });
  check('memory limit (malloc fails) -> 9', code === 9, code);
}
{
  const { code, out } = await runSrc('#include <stdio.h>\nint main(int argc, char** argv) { unsafe { printf("%d %s", argc, argv[1]); } return 0; }\n',
                                     { args: ['x1'] });
  check('args', code === 0 && out === '2 x1', JSON.stringify(out));
}
process.exit(failed ? 1 : 0);
