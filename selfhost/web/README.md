# scc playground API

`scc.js` compiles and runs SafeC in a browser (or Node 18+) with no server.
The compiler is **scc**, SafeC's self-hosted compiler (written in SafeC,
see `../README.md`) built to WebAssembly; the programs it produces are
WebAssembly modules too, run in a Web Worker with a time and memory limit.

This page is the contract the docs site builds on. Anything not written here
may change.

## Files

`web/` is the directory to deploy. `build-web.sh` (one level up) builds
`scc.wasm` into it, next to the scripts that are checked in. The docs site
runs `selfhost/build-web.sh` and copies all of `selfhost/web/` to
`assets/scc/`. `build-web.sh` finds `safec`, `clang` and `wasm-ld` on its
own; the environment variables in its header override what it finds.

| file        | what it is                                                    |
|-------------|---------------------------------------------------------------|
| `scc.js`    | the API (an ES module)                                        |
| `worker.js`, `exec.js` | what `run()` uses; they must sit next to `scc.js`  |
| `scc.wasm`  | the compiler with its bundled headers and std sources (about 630 KB, 165 KB gzipped), built |

The other files in `web/` (`run-node.mjs`, `scc-cli.mjs`, `test-api.mjs`,
this README) are Node tools, and serving them does no harm.

Serve `scc.wasm` as `application/wasm`. Nothing else is needed: no
cross-origin isolation, no SharedArrayBuffer.

## API

```js
import { init, compile, run, parseDiagnostics } from './scc.js';

await init();                        // optional; compile() calls it
const r = await compile(source);
for (const d of r.diagnostics) console.log(`${d.line}:${d.col} ${d.severity}: ${d.message}`);
if (r.ok) {
  const code = await run(r.wasm, { stdout: s => out.append(s), stdin: '' });
}
```

### `init(wasm?) → Promise<void>`

Loads the compiler. `wasm` is a URL or path string, a `URL`, bytes
(`ArrayBuffer` or typed array) or a `WebAssembly.Module`. The default is
`scc.wasm` next to `scc.js`. Calling it again with no argument does nothing;
calling it with an argument replaces the compiler.

### `compile(source, options?) → Promise<{ ok, diagnostics, wasm, text }>`

Compiles one SafeC translation unit. The source file is named `main.sc`
(that is the `file` of its diagnostics).

`options.filename` names the source in diagnostics and `__FILE__`
(default `main.sc`, at most 255 bytes).

With `options.check` set to `true`, the source is only checked: there is
no code generation, `main` is not required, and `wasm` is `null`. An editor
can use this for live diagnostics.

- `ok`: `true` when there were no errors. Warnings do not make it `false`.
- `diagnostics`: an array of
  `{ file, line, col, severity: 'error' | 'warning', message }`. `line` and
  `col` are 1-based. A diagnostic that has no location (`no 'main'
  function`, say) has `file: ''`, `line: 0`, `col: 0`. Errors in a bundled
  header name that header (`stdio.h`, `std/mem.h`). Messages use the C++
  `safec` compiler's wording.
- `wasm`: the program as a `Uint8Array` when `ok`, otherwise `null`.
- `text`: the raw diagnostics, one per line, formatted
  `file:line:col: severity: message`.

Each call runs on a fresh compiler instance, so calls are independent. A
compile takes 5 to 50 ms for a typical example program once `init()` has
run. A source larger than 1 MB throws an `Error`. A compiler crash does not
throw: it comes back as an `ok: false` result whose one diagnostic starts
with `internal compiler error`.

Programs can `#include` the headers bundled into the compiler:

- C library subsets: `stdio.h`, `stdlib.h`, `string.h`, `ctype.h`,
  `math.h`, `stdbool.h`, `stdint.h`, `stddef.h`, `limits.h`, `time.h` and
  `assert.h`.
- The SafeC std library: `std/io.h`, `std/str.h`, `std/mem.h`,
  `std/panic.h`, `std/math.h`, `std/fmt.h`, `std/convert.h`,
  `std/collections/{slice,vec,string,map}.h`, `std/stdint.h`,
  `std/limits.h`, `std/float.h`, `std/stdbool.h`, `std/inttypes.h`,
  `std/ctype.h`, `std/assert.h`, `std/stdckdint.h`, `std/cmp.h`,
  `std/traits.h`, `std/bit.h`, `std/result.h`, `std/atomic.h` and
  `std/sys.h`, plus what those include. These are the real std sources,
  compiled into the program.

Any other include is an error. `../SUBSET.md` lists the language subset.

### `run(wasm, options?) → Promise<exitCode>`

Runs a compiled program in a new worker and resolves to its exit code when
it finishes. It never rejects because of something the program did.

| option        | default | meaning                                            |
|---------------|---------|----------------------------------------------------|
| `stdout(text)`| ignore  | called with output as it is written                |
| `stderr(text)`| ignore  | the same for standard error                        |
| `stdin`       | `''`    | the program's whole standard input                 |
| `args`        | `[]`    | `argv[1..]` (`argv[0]` is `"main"`)                |
| `timeoutMs`   | `10000` | wall-clock limit                                   |
| `maxMemoryMb` | `256`   | the most linear memory the program can grow to     |

Output is unbuffered: the callbacks get text when the program writes it.

`run()` starts its own worker, nested if the caller is itself a worker (as
the docs site's playground is). Where a worker can't start another one,
`run()` runs the program in the caller's thread instead. There it can't
enforce `timeoutMs`, so the caller enforces a limit by terminating its own
worker.

The exit code is `main`'s return value or `exit()`'s argument (0 to 255),
or one of these:

| code  | cause                                                                  |
|-------|------------------------------------------------------------------------|
| `124` | the time limit was hit; the worker is terminated                      |
| `134` | a trap: a failed bounds check, `assert`, `abort()`, a `panic`, an unreachable point |
| `136` | an arithmetic trap: integer division by zero, or an overflowing `INT_MIN / -1` |
| `137` | the module could not be instantiated within the memory limit          |
| `139` | stack overflow: the 1 MB shadow stack ran out                          |

When the program asks for more memory than `maxMemoryMb` while it runs,
`malloc` returns `null`. The program sees that and decides what to do, so
the run ends with whatever exit code the program then returns.

A failed runtime check prints a message to `stderr` first (`runtime
error: index out of bounds`, `runtime error: arena capacity exceeded`,
`runtime error: stack overflow`, `Assertion failed: ...`, `panic at ...`). A
WebAssembly trap prints `[trap: <engine message>]`.

### `parseDiagnostics(text) → Diagnostic[]`

The parser `compile()` uses for `text`, exported for tools that run `scc`
in other ways.

## The compiled program's interface

This is for running a module without `run()`. It is a stable interface, but
most pages need only `run()`.

- Exports: `memory`, `main`, and `__rt_argv() → ptr`. `main` takes no
  parameters or `(argc, argv)`, as the source declares it (check
  `main.length`). For the second form, write into the 16 KB buffer at
  `__rt_argv()` a table of `argc + 1` little-endian i32 string addresses
  (the last one 0), followed by the NUL-terminated strings, then call
  `main(argc, buffer)`.
- Imports, all from module `env`: `__host_write(fd, ptr, len) → i32`,
  `__host_read(fd, ptr, len) → i32` (0 at end of input) and
  `__host_exit(code)`, which must not return (throw from it).

## The compiler module's interface

This is what `compile()` uses. Use `compile()` itself where you can.

Imports: `env.abort()`. A self-compiled `scc.wasm` (see `../bootstrap.sh`)
also imports `__host_write`, `__host_read` and `__host_exit`, and stubs for
those are enough.

| export                             | meaning                                         |
|------------------------------------|-------------------------------------------------|
| `scc_begin() → ptr`                | where to write the UTF-8 source                  |
| `scc_capacity() → n`               | the most bytes that fit there                    |
| `scc_compile(len) → 1 / 0`         | compiles the `len` bytes; 1 when ok              |
| `scc_set_check(1)`                 | makes the next `scc_compile` only check (call it before) |
| `scc_name_buf()`, `scc_set_name(len)` | the source's file name: write it to the buffer, then set its length |
| `scc_output()`, `scc_output_len()` | the produced wasm module                         |
| `scc_diagnostics()`, `scc_diagnostics_len()` | the diagnostics text                   |
| `scc_error_count()`                | the number of errors                             |

Use a fresh instance for each compile.

## Testing

`node web/test-api.mjs` checks this contract end to end. CI runs it
(`../ci.sh`). `node web/scc-cli.mjs file.sc --run` compiles and runs a file
through `scc.js`.
