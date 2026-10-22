// Instantiates the wasm_rt test module with an empty import object (any
// import left unresolved fails here) and runs its checks.
import { readFileSync } from 'node:fs';
const bytes = readFileSync(process.argv[2]);
const mod = new WebAssembly.Module(bytes);
const imports = WebAssembly.Module.imports(mod);
if (imports.length) {
  console.error('FAIL unexpected imports:', imports.map(i => `${i.module}.${i.name}`).join(', '));
  process.exit(1);
}
const { exports } = new WebAssembly.Instance(mod, {});
const pagesBefore = exports.memory.buffer.byteLength / 65536;
const failed = exports.run_tests();
const pagesAfter = exports.memory.buffer.byteLength / 65536;
if (failed !== 0) {
  console.error(`FAIL check #${failed}`);
  process.exit(1);
}
console.log(`wasm_rt: all checks passed (memory ${pagesBefore} -> ${pagesAfter} pages)`);
