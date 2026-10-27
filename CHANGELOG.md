# Changelog

All notable changes to SafeC — the compiler, the standard library,
safeguard and the install scripts. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are the
release tags.

## [Unreleased]

### Fixed

- **Compile time:** the memory-flow check grew exponentially with a run of
  `insert()` calls on one `BST<K, struct S, C>` (7 calls took 46 s,
  `tests/stdlib/tests/test_collections.sc` 23.6 s; now 0.2 s). Applying a
  callee summary nested a node inside a node's `val` once per call and
  multiplied the caller cells per dereference; cell paths are now bounded
  and aliased cells merged. `compiler/tests` can bound a test's compile
  time (`// MAX-SECONDS:`).
- A second `#include` of a C header imported through clang (`<math.h>`
  twice, or once more from another header) is a no-op instead of pasting
  the raw `/usr/include` header (16 errors).
- Diagnostics from inside an `#include`d file carry that file's name
  (stored diagnostics, e.g. the language server's, pointed at a freed
  string).
- `safeguard` links libm on Linux and the BSDs: a program using the C
  library's `<math.h>` (`sqrt`, `cbrt`, ...) without `std/math.h` failed to
  link.
- A use of a dropped or moved value through a method call (`b.len()`) is
  reported at the variable, not at the `(`; function references print as
  `fn R(Args)` / `?fn R(Args)` in messages (and doc JSON, and scc) instead
  of `&static const R(Args)`.
- `__VA_ARGS__` expands to every variadic argument (`F(1, 2, 3)` with
  `#define F(...) g(__VA_ARGS__)` gave `g(1)`).
- `#line N ["file"]` sets `__LINE__`, `__FILE__` and the locations of the
  code after it (it was ignored; compiler and selfhost).
- `atomic_cas` / `atomic_compare_exchange_strong` with `"release"` or
  `"acq_rel"` generate valid IR: the failure ordering is the strongest one
  allowed (relaxed / acquire), as in C11.
- `must_use` / `warn_unused_result` on one declaration of a function
  applies to it when another declaration (the definition) doesn't repeat
  it.
- `safec --emit-bin` no longer prints clang's "overriding the module target
  triple" warning: host builds name their triple in the IR.
- `safeguard --help` lists every option (`doc --home`, `--source-ref`,
  `--verbose`, `publish --no-verify`, `--dry-run`, `--remote`, ...); the
  smoke test checks it against the parser. `safeguard new`'s template calls
  `printf` without an unneeded `unsafe { }`, and the install scripts' help
  no longer names a stale release tag.
- Compiler build warnings (unused variables, missing field initializers,
  member initialization order).
- **Warnings on incremental-cache hits:** a cache hit skips lexing, parsing,
  sema and codegen, so a rebuild of an unchanged file printed none of their
  warnings (e.g. a non-exhaustive `switch`), and `safeguard build` showed them
  only on the first build. safec now stores the diagnostics a compile printed
  after preprocessing in a `.bc.diag` file beside the cached bitcode and
  prints them again on a hit. A cache entry without one (from an older safec)
  is a miss, and `--clear-cache` removes both files. The smoke test compiles a
  file twice into a fresh cache and compares the warnings.

## [1.0.0-beta-7] - 2026-10-27

### Fixed

- A local declaration whose type is a typedef name followed by two or more
  `*` (`Foo_t** x = pp;`, `Foo_t*** w;`) parses as a declaration instead of
  failing with "undeclared identifier"; nested prefix operators in
  expressions (`**pp`, `a * **pp`, `-*p`, `- -x`) parse too (compiler and
  selfhost).
- **Containers of `Drop` elements:** `HashMap<K, String>`, `StrMap<String>`,
  `List<String>`, `Stack<String>`, `Queue<String>`, `BST<K, String, C>`,
  `BTree<String>`, `StaticVec<String>` and `StaticMap<String>` failed the
  ownership check inside std (the value was not dropped on the OOM / full
  path, or was moved out of a node outside `unsafe`). Every container now
  owns its elements the way `Vec` does: overwriting a key drops the old
  value, a value that can't be stored is dropped, `remove()` / `pop()` move
  it out, `List::remove` drops the element it unlinks, and `clear()` /
  `drop()` drop every element still held (before, `StrMap::clear()` /
  `drop()` and the others released the storage but leaked the values).
  `BTree`, `StaticVec` and `StaticMap` live in place / over the caller's
  storage and have no `drop()`: `clear()` them. Don't free a container's
  values yourself before dropping it (`m.foreach(free_value); m.drop()`):
  drop the container alone.
- **`std::Reactor` is a `Drop` type:** it had no `drop()`, and `close_()`
  left the `sigfds` / `watched` tables allocated (and, on Windows, called
  `WSACleanup` even if `init()` had failed). `Reactor::drop()` closes it;
  `close_()` now also frees both tables, is idempotent (`kq = -1`
  afterwards) and leaves a reactor `init()` can reopen. The HTTP server's
  reactor threads drop their reactor (on the `init()` failure path too).
- **`WsConn` never touches a descriptor it doesn't own:** a zero-initialised
  `WsConn` (e.g. after a failed `accept()`) sent a close frame to fd 0 and
  closed it when dropped. New `WsConn::closed()` (fd = -1) is the starting
  value for `accept()` and std's own placeholders; `accept()` leaves the
  connection closed when it fails; `drop()` / `close()` and the send
  functions do nothing on a closed connection (fd < 0).
- `safeguard --help` lists the `arm-81m` target alias (Armv8.1-M,
  Cortex-M55/M85) with the others; the smoke test now checks the help
  against `safec --list-targets`.
- **`#include <math.h>` is the C library's header in a safeguard build:**
  safeguard put std/ itself on SafeC's include path (`-I <std>`), so a bare
  `<math.h>` (or `<time.h>`, `<stdint.h>`, `<ctype.h>`, ...) found std's
  header instead and `sqrt` & co. were undeclared. std is now reached only
  as `<std/...>` (its parent directory is on the path) — in builds, tests,
  `check`, `lint`, `doc` and dependency builds, and for std's own units.
  **Breaking** for code that included std headers without the prefix
  (`#include <io.h>`, `"prelude.h"`): write `<std/io.h>`. The `safeguard
  new` template says `<std/prelude.h>`; the smoke test builds a project
  using both `<math.h>` and `<std/math.h>`.
- **std/time links in the playground:** scc's runtime had no `gmtime_r`, so
  a program using std/time failed with "undefined function". The runtime
  now provides `gmtime_r`, `localtime_r` (UTC: the playground has no time
  zone), `mktime`, `strftime`, `strptime`, `clock_gettime` (the epoch: no
  clock) and `nanosleep`; a difftest corpus case checks them against glibc.
  `tools/genvfs.py` now embeds each bundled file in chunks of at most
  32 KiB: scc decodes a string literal into a 64 KiB buffer, so the grown
  runtime came out truncated in the self-compiled scc (bootstrap stage 2).

### Changed — no `void*` anywhere in std

- **The raw memory layer is bytes:** `alloc_raw_`, `dealloc_raw_`,
  `realloc_buf_raw_`, `safe_memcpy_raw_` and the other `std/mem.h` `_raw_`
  forms take and return `unsigned char*` (symbol names and ABI unchanged:
  `new T` still calls `std_alloc_raw_`); the generic `alloc` / `dealloc` /
  `safe_memcpy` wrappers cast inside `unsafe`.
- **Breaking:** the allocator traits (`Allocate` / `Deallocate`) and every
  std allocator (`BumpAllocator`, `PoolAllocator`, `SlabAllocator`,
  `TlsfAllocator`, `SecureAllocator`, `CountingAllocator`) hand out
  `?&heap unsigned char` instead of `?&heap void`, and `heap_alloc` /
  `heap_free` / `heap_realloc` use `&heap unsigned char`. A custom
  allocator changes its `alloc` / `dealloc` signatures the same way;
  `alloc_t` / `alloc_array_t` / `dealloc_t` are unchanged.
- `T x = malloc_defer(n);` now expands to `std::alloc(n)` plus a deferred
  `std::dealloc(x)`, taking `T` from the declaration (`auto` and `void*`
  keep `alloc_raw_` / `dealloc_raw_`, so `auto` is now `unsigned char*`).
- libc / OS / C-library declarations use typed handles: C's `FILE` is the
  opaque `CFile_`; pthread and Win32 synchronisation objects, `HANDLE`s,
  `DIR`, find handles, `timespec`, `struct tm`, `sigset_t`, `pollfd`,
  `epoll_event`, `kevent`, `fd_set`, `addrinfo` and `WSADATA` are passed as
  their (opaque or mirrored) structs; thread start routines, `signal()`
  handlers and comparators are `fn` types. The TLS backends declare
  OpenSSL's dlopen'd API (`SSL`, `SSL_CTX`, `BIO`, `X509`, ..., `dlsym`
  returning a function), CoreFoundation/Security and SChannel/crypt32 with
  opaque handle structs and one `fn` type per signature. The channel
  runtime (`__safec_chan_*`) takes a `SafecChan_*` and its 8-byte slot, and
  `__safec_thread_create` the task `fn` and its `?&Erased` argument.
- `tests/std_modules/api_lint.py` fails on any `void*` / `&void` (any
  region) in std's headers and sources, FFI externs included, with no
  exemptions (std/ml, std/gui and the CUDA/ROCm FFTs included, below).

### Changed — no `void*` left in std/gui

- **std/gui** has no `void*` at all, internal FFI included (public API
  unchanged): the X11, Wayland, Win32 and Cocoa backends and the EGL/GLES,
  D3D11 and Metal presenters declare their system APIs with opaque handle
  structs (`XDisplay_`, `WlDisplay_` / `WlProxy_`, `HWND_`, `ObjcId_` /
  `ObjcSel_`, `EGLDisplay_`, ...), typed `struct wl_interface` tables,
  Wayland dispatchers taking `?&Erased`, COM objects through a typed method
  table, one typed `objc_msgSend` cast per call shape, typed dlopen()/dlsym()
  lookups and byte pointers for pixel and property data; the portable core
  uses `safe_memset_obj` / `safe_memcpy` and byte-pointer identity compares.

### Changed — no `void*` left in std/ml

- **std/ml** (and std/dsp's `fft_cuda` / `fft_rocm`) has no `void*` at all:
  a tensor's `gradFn` is a `fn void(&DenseTensor<float>)` taking the node
  through its float32 layout (shared by every element type), the coherence
  internals take that `FloatTensor*`, and the GPU bindings (CUDA, NVRTC,
  cuBLAS(Lt), cuDNN, cuFFT, HIP, hiprtc, rocBLAS, MIOpen, hipFFT, Metal
  through the Objective-C runtime, Vulkan, WebGPU) declare each handle kind
  as its own opaque struct (`CuEvent_`, `CudnnTensorDesc_`, `HipModule_`,
  `ObjcSel_`, `VkDevice_`, `WGPUBuffer_`, ...; the shared declarations in the
  internal `cuda_ffi.h` / `rocm_ffi.h` / `mps_ffi.h`), device memory as
  `CuDevPtr_` / `HipDevPtr_`, kernel arguments as `CuKernelArg_` /
  `HipKernelArg_`, pNext / nextInChain chains as `VkNext_` /
  `WGPUChainedStruct`, and host buffers as `unsigned char*`; cuDNN's SDPA
  entry points are looked up as typed functions.

### Changed — no `void*` left in std/audio; fewer in std's internals

- **std/audio** has no `void*` at all, internal headers and FFI included:
  the audio/MIDI backend tables are `?&static AudioBackendVt_` /
  `MidiBackendVt_`, each backend's state is an opaque typed handle
  (`?&heap AudioStreamImpl_` / `MidiPortImpl_`), the stream buffers are
  `[]unsigned char` slices, the stream thread's body and the MIDI callback
  slot are `?fn`, dlopen()/dlsym() return `?&AudioLib_` / a function, and
  libasound, libjack, Core Audio, Core MIDI, WASAPI (COM) and WinMM are
  declared with opaque handle structs, typed callbacks taking `?&Erased`
  and byte pointers for their buffers.
- **Breaking:** `Audio::open_stream<T>` and `MidiIn::set_callback<T>` require
  `T: Send + Sync`: the callback runs on the audio/MIDI thread while the
  context stays reachable from its owner's (plain data and atomics pass; raw
  pointers, stack references and `!Send` handles are compile errors).
- `Reactor::poll_events_raw_`, `parallel_for_raw_`'s job, `sys_qsort` /
  `sys_bsearch` / `sys_atexit`, the HTTP server's handler context and the
  fs drivers use typed `fn` / `?&Erased` instead of `void*`;
  `dealloc_raw_((void*)p)` is the typed `dealloc(p)` throughout std.

### Changed — std's public API has no `void*` and no pointer + length pairs

std's public declarations no longer take or return `void*`, raw `T*`
buffers, `(pointer, length)` pairs or arrays through pointers.

- **Buffers are slices.** SafeC has no `&T[]` type; the equivalent is the
  built-in slice: `[]T` for a buffer the function writes, `[]const T` for one
  it only reads. A slice carries its length, indexing it is bounds-checked,
  and a fixed-size argument (a key, a nonce, a digest, a tag) is checked on
  entry (a short one panics instead of being read or written past).
  `std::str_bytes("text")` (`std/str.h`) gives a string's bytes as a
  `[]const char`, accepted wherever `[]const unsigned char` is expected.
- **Type erasure is typed.** Where std must store callbacks with different
  context types side by side (executors, wakers, timers, widgets, the VFS,
  audio/MIDI callbacks, HTTP handlers, DOM listeners, thread and task entry
  points), the slot is a `?&Erased` (`std/erased.h`) with a typed `fn`, filled
  only by generic methods that take a matching `fn(?&T)` / `?&T`. Opaque
  handles are typed too (`?&MtlObject`, `?&CudaNativeStream`,
  `?&struct GuiWindowState`, ...). The `void*` halves that remain are private
  (`*_raw_`, `*_n_`, std-internal headers) or FFI externs.
- `tests/std_modules/api_lint.py`, run by the smoke test, fails when a raw
  pointer appears in a public std declaration.

### Added — read-only, optional and checked slices; typed erasure

- Compiler: `[]const T` read-only slices (`[]T` converts to it implicitly;
  writing through one is an error), `?[]T` optional slices, slice-returning
  functions and slice-typed top-level declarations, `?fn R(Args)` nullable
  function references, range-checked re-slicing (`a[i..j]` traps when
  `i > j` or `j > len`, outside `unsafe`), `.len` / indexing / re-slicing on
  a slice-valued call, byte slices interconverting across `char`,
  `signed char` and `unsigned char`, `&s[i]` on a slice, slices of
  typedef/newtype elements (`[]Fixed`, `[]S24`), struct fields naming a
  struct defined later.
- `std/erased.h` (`Erased`), `std/audio/s24.h` (`S24`, a packed 24-bit
  sample), `std::str_bytes`, `struct TimeFields` (`std/time.h`),
  `RpcResult::bytes()`, `safe_memset_obj` / `mem_zero_secure_obj`,
  `Vec::from_slice` / `as_slice`, `Slice::from` / `as_slice`,
  `String::from_bytes` / `push_bytes` / `as_bytes`, `StaticVec::over` /
  `StaticMap::over` / `RingBuffer::over` / `LFQueue::over` on slices.

### Breaking changes — slices and typed erasure in std

Every change below is a signature change the compiler reports at each call
site (wrong argument count or type); none changes behaviour silently except
where noted.

- **I/O.** `Reader::read([]unsigned char)` / `Writer::write([]const unsigned char)`
  replace `read(buf, n)` / `write(buf, n)`; `read_exact(&r, buf)`,
  `write_all(&w, buf)`. `File`, `Pipe`, `BufReader`, `BufWriter`, `TcpStream`,
  `TlsStream`, `HttpChunkedWriter` follow. `File::gets([]char)`,
  `File::write_bytes(path, []const unsigned char)`, `file_read` /
  `file_write` on slices.
- **Formatting and strings.** `fmt_*` / `io_fmt_*` / `io_read_line` /
  `io_read_token` / `wasm_itoa` / `dom_get_prop_buf` / `time_format` write
  into a `[]char`. `str_copy([]char, src)`, `str_cat([]char, src)` and
  `str_ncat([]char, src, n)` are bounded by the slice, always NUL-terminate
  and return 1 when everything fit, 0 when truncated (`str_copy` no longer
  has `strncpy`'s missing-NUL case). `mem_chr([]const unsigned char, c)`.
  `String::from_n` / `push_n` / `string_from_n` are gone from the public API
  (use `from_bytes` / `push_bytes`); `String::join(sep, []String)`,
  `split*(…, []String)`.
- **Memory.** `safe_memcpy` / `safe_memmove([]T dst, []const T src)` copy
  `min(dst.len, src.len)` **elements** (not bytes) and return the count;
  `safe_memset([]T, val)`, `safe_memcmp([]const T, []const T)` (a shorter
  equal prefix compares less), `mem_zero_secure([]T)`. Callers that passed a
  byte count must pass a slice of elements now.
- **Time.** `time_gmtime` / `time_localtime` return a `struct TimeFields`
  instead of filling a 9-int array; `time_format([]char, fmt, &tm)`,
  `time_parse(s, fmt, &tm)`, `time_mktime(&tm)`.
- **Sorting.** `sys_qsort([]T, cmp)`; `sys_bsearch(&key, []const T, cmp)`
  returns `?unsigned long` (the index) instead of `?&T`.
- **Collections.** `Vec::from_arr` → `Vec::from_slice`; `Vec::extend([]const T)`;
  `push_all` / `enqueue_all` / `arr_*` on slices; `StaticVec::over([]T)`,
  `StaticMap::over([]StaticMapSlot<V>)`, `RingBuffer::over/write/read/peek`
  on slices; `LFQueue::over([]T)`.
- **Allocators and memory-mapped I/O.** `BumpAllocator` / `PoolAllocator` /
  `TlsfAllocator::over([]unsigned char)`,
  `SlabAllocator::over([]unsigned char, obj_size)`,
  `DmaBuffer::wrap([]unsigned char, phys)`.
- **Async, sync, threads.** `Waker`, `Context`, `TaskHeader`, `Task`,
  `thread_create`, `parallel_for`, channels: typed `?&Erased` contexts
  (`Waker::empty_()`); async net writes take `[]const unsigned char`.
- **Audio / MIDI.** `AudioIo`'s buffers are byte slices with typed slice
  accessors (`output_f32()` → `[]float`, `input_f32()` → `[]const float`,
  `output_i24()` → `[]S24`, …); `audio_convert.h` and the MIDI API
  (`MidiMsg::of`, `midi_build_sysex`, `MidiParser::new`, `encode`,
  `get_message`, `send_message`, `MidiEvent.bytes`) on slices.
- **DSP.** FFT / DFT / DCT / windows / convolution / resampling / imaging /
  STFT / CQT / CWT / minimum-phase / z-transform take slices (lengths come
  from the slices; DFT / DCT / convolution / imaging / STFT / CWT /
  minimum-phase now return `int`, 0 when an output is too short);
  `FirFilter` / `IirFilter` / comb / Karplus-Strong constructors take their
  coefficient and history buffers as slices (a comb's delay is
  `history.len`); `process_block([]const T in, []T out)`, `pluck([]const T)`,
  `filter_impulse_response` / `filter_step_response` / `generate_block` on
  slices; `AudioBuffer` frames, `cuda_` / `rocm_fft*` on slices.
- **Crypto.** `Hasher::update([]const unsigned char)` / `finish([]unsigned char)`,
  `sha1` / `sha256` / `sha384` / `sha512(data, digest)`, `Hmac::new(fresh, key)`,
  `hmac_sha256(key, data, out)`, `hkdf_extract(fresh, salt, ikm, prk)`,
  `hkdf_expand(fresh, prk, info, out)`, `digest`, `mgf1_xor`, AES (ECB / CBC
  / GCM), ChaCha20 / Poly1305 and the AEAD, `ct_eq` / `ct_is_zero` /
  `ct_copy_if`, X25519, EC (keygen, ECDH, ECDSA), RSA, `BnMont`, X.509,
  `RngCtx::fill` and the TLS record layer take slices. A user-defined
  `Hasher` must implement the slice signatures.
- **Networking, HTTP, RPC.** `NetDriver::transmit([]const unsigned char)`,
  `TcpConn::send` / `read`, `ip_checksum`, `tcp_checksum`; `http_request`,
  `http_request_timeout`, `http_post`, `http_fetch`, `http_post_url`,
  `http_build_response`, `HttpResponse::bytes` / `set_body_bytes`, the
  chunked coding, `MultipartBuilder::add_file`, `SseParser::feed`,
  `WsConn::send_binary` take `[]const unsigned char` bodies (empty = none);
  `RpcMethodHandler` is `fn RpcResult([]const unsigned char)` and
  `rpc_call(…, []const unsigned char)`. Handler/router state and connection
  readers are typed (`?&Erased`).
- **Filesystems.** `BlockDriver` / `BlockDevice` read and write slices;
  `VfsOps` is a table of typed `?fn` entries over `&Erased`;
  `VfsNode::read` / `write` and the drivers' `read` / `write` on slices;
  `VfsNode::readdir` is generic over the user data.
- **Encoding and serialization.** `base64_encode`, deflate / gzip / zlib,
  `ascii_is_valid_bytes`, `QueryParams::parse_bytes`, `PbReader::new([]const
  unsigned char)`, `PbReader::read_length_delimited` returns
  `?[]const unsigned char`, `pb_write_bytes_field(&out, field, bytes)`.
- **SIMD and ML.** `vm_exp_f32` / `vm_tanh_f32` / `vm_sigmoid_f32([]const float,
  []float)`; `sgemm` / `sgemm_serial` take slices and return 0 when one is
  too short for its dimensions. The host-array GPU ops (`cuda_` / `rocm_` /
  `mps_` / `spirv_` / `webgpu_` `*_f32`, including `*_matmul_f32`), buffer `upload` / `read` / `write`,
  the float16 bulk conversions, `DenseTensor::from_1d([]const float, grad)`,
  `from_2d([]const float, rows, cols, grad)`, `SparseTensor::new_csr`,
  `HalfTensor` / `BF16Tensor::gemv` / `gemv_batch` take slices; tensor
  `grad_fn`s are typed; the CUDA `*_persistent` raw-handle API is removed
  (use `CudaBuf` / `CudaHalfBuf`); Metal / CUDA / ROCm native handles are
  typed opaque structs.
- **GUI (signatures only).** Widget and window callbacks are typed `?fn`
  with `?&Erased` data; `GuiWindow.platform` is a `?&struct GuiWindowState`;
  `GuiCanvas::set_dash([]const float, offset)`, `fill_text_n([]const char, x, y)`,
  `GuiPath::polygon([]const float)`, `GuiImage::load_png([]const unsigned char)`,
  `GuiTypeface::load_memory(_index)` / `measure_n`, `GuiTextLayout::build`,
  `gui_utf8_next([]const char, &i)`; `gui_window_presenter_name` returns
  `const &char`.
- **Low level.** I2C (`I2cChip`, `I2cBus::write/read/write_read`) and SPI
  (`SpiDevice::write(tx, ?rx)`, `read(rx)`) on slices;
  `Mailbox::send(pid, type, []const unsigned char)`; syscall and ISR tables
  hold `?fn`; `signal_handle` returns `int` (0 / -1) instead of an opaque
  previous handler; `FuzzTarget::run([]const unsigned char)` and fuzz
  targets are `fn void([]const unsigned char)`.
- Compiler: casting a slice to a pointer (`(T*)s`) is now an error (it used
  to reach LLVM and fail verification); use `s.ptr`.
- scc (`selfhost/`) follows: it compiles the slice forms above (`[]T`
  returns and fields, `[]const T`, `?[]T`, `?fn`, byte-slice conversions,
  the slice-to-pointer cast error), and its bundled std (`src/vfs.sc`) is
  regenerated from the slice-based headers.

### Added — self-hosted compiler (selfhost/)

- **scc**, a SafeC compiler written in SafeC, is in the new top-level
  `selfhost/` directory. It compiles SafeC straight to a WebAssembly binary,
  with no LLVM and no linker. It has its own lexer, preprocessor, parser,
  semantic analysis with safec's safety checks and diagnostics, and wasm
  backend, plus a small runtime written in SafeC. `selfhost/SUBSET.md`
  lists the supported language subset.
- scc builds with the C++ `safec`, both natively (`selfhost/build.sh`) and
  as WebAssembly (`selfhost/build-web.sh`). It also compiles itself:
  `selfhost/bootstrap.sh` checks that the self-compiled compiler reproduces
  itself byte for byte.
- There is a playground API for running SafeC in the browser:
  `selfhost/web/scc.js` provides `compile(source)` and
  `run(wasm, { stdout, stdin })`, with time and memory limits in a Web
  Worker. The contract is in `selfhost/web/README.md`.
- A differential test harness, `selfhost/tests/difftest.py`, compiles the
  book and reference programs, a corpus and negative tests with both
  compilers and compares the results. It also runs `compiler/tests`
  compile-only. CI runs all of it on Linux, in the new `selfhost` job.

### Added — `safeguard doc` and `safeguard publish`

- `safeguard doc` generates a package's API reference into `build/doc/`, like
  rustdoc: per-item pages (structs, unions, enums, traits, functions,
  macros, constants, type aliases) with highlighted declarations, Markdown
  doc comments, intra-doc links (`[Vec::push]`), code examples and source
  links; a module tree (one module per header); highlighted source pages;
  client-side search (a SafeC program compiled to wasm32 on `std/wasm/dom`);
  light and dark themes; links into std's published reference; a link check
  of the result. `--serve` serves it over `std/http`, `--std` documents std
  itself.
- `safec --emit-doc-json`: the declarations of the inputs — kinds, names,
  namespaces, generics and bounds, parameters, fields, methods, enumerators,
  macros, signatures and locations — with their doc comments, as JSON, from
  the real front end (parse only). `--doc-root` / `--doc-extern` /
  `--doc-private` select what is listed.
- Doc comment conventions: `///`, `/** */`, trailing `///<`, `//!` module
  docs, and std's plain `//` comments above or beside a declaration;
  `@hidden`. See the SafeC docs' *Doc Comments* page.
- `safeguard/doctool`: **safec-doc**, the generator `safeguard doc` runs, a
  SafeC program on `std/serial/json`, `std/serial/markdown` (render hooks),
  scx templates, `std/dir`, `std/path`, `std/http` and `std/wasm/dom`.
  Releases ship it prebuilt (`bin/safec-doc`, `share/safec/doctool/`);
  from a source checkout it is built on first use and cached.
- `safeguard publish` prepares a library's git repository for publishing:
  `git init`, `.gitignore`, README/LICENSE/metadata checks, `--bump`
  (semver), and CI for GitHub Actions (default), GitLab CI, Gitea or Forgejo
  Actions (`--ci=`) that builds and tests on Linux/macOS/Windows and, on a
  tag, attaches prebuilt static and shared libraries per OS/arch to a
  release and deploys `safeguard doc` to the repository's Pages. A dry run
  unless `--write`; `--tag` commits and tags, `--push` pushes.
- `Package.toml`: `[package]` `repository`, `homepage`, `documentation`,
  `readme`, `keywords`; a `[publish]` table (`ci`, `branch`, `tag_prefix`,
  `pages`, `release`).
- `safeguard build --crate-type staticlib|cdylib|bin` overrides
  `[build] crate_type`.
- `examples/doc_demo`: a documented example library.

### Added — ownership checks (`Drop`) and thread safety (`Send`/`Sync`)

- **`Drop`**: a type with a `void drop()` method, or one holding such a
  value, must be dropped or moved exactly once on every path (`return`,
  `break`, `continue`, `goto`, `match`, `defer`, loops). Leaks, use after
  drop or move, double drops, overwriting a live value, and dropping or
  moving a value out of borrowed memory are compile errors, with a note at
  the declaration. Checked per generic instantiation; panicking paths are
  exempt. No run-time cost: nothing is inserted.
- `consume` methods (`int close() consume;`) destroy their receiver like
  `drop()`; destructors must take apart every owned field. `drop(x)` (drop
  glue for any type), `needs_drop(T)`, conditional `void drop() if (cond);`,
  and `forget(x)` (only in `unsafe`). `std/ownership.h`: `std::replace`,
  `std::swap`.
- A reference a method returns into its receiver borrows from it: using it
  after the receiver is dropped or moved is an error (`MutexGuard::get()`,
  `Vec::at()`).
- **`Send` / `Sync`**, derived from each type's structure, overridable with
  `unsafe impl Send/Sync;` and `impl !Send/!Sync;`, usable as generic
  bounds. `spawn` takes only static data that is Sync (or synchronizes its
  own writes); `spawn_scoped` lends a local until it is joined (no access
  meanwhile for a mutable lend; the local must outlive the join);
  `chan_send` needs a Send value and moves it.
- `std/thread.h`: `thread_spawn` / `SpawnedThread<T, R>` (moves its
  argument into the thread, `join()` returns the result), `Mutex<T>` /
  `MutexGuard<T>` (the value only through the guard, unlocked by its
  `drop()`), `RwLock<T>`. `std/atomic.h`: `AtomicInt`, `AtomicLong`.
- `Result::unwrap()` / `unwrap_err()` (consume).

### Changed — ownership

- The standard library's owning types implement `Drop`: collections,
  `File`, directories, sockets, TLS, HTTP, threads, sync primitives, GUI,
  audio/MIDI, async (futures, tasks, wakers, executors, channels), ML
  containers. `free()`/`close()`/`destroy()` stay as `consume` aliases.
  `Vec::clear()`/`drop()`/`set()` drop the elements they discard.
- `ThreadPool::spawn`/`block_on`, `Chan<T>`, `AsyncChannel<T>` and
  `Oneshot<T>` require Send values; GUI windows are `!Send`.
- Dropping a spawned, unfinished `AsyncTask` panics instead of leaving its
  executor to poll freed memory.

### Fixed — found by the ownership checks

- Leaks, double frees and use-after-frees across std, tests and examples,
  among them: `TlsStream::client()`/`TlsAcceptor::new()` failures kept
  their backend state; a failed client-certificate CertificateVerify, the
  GUI SVG renderer (a use after free of a by-value `String`), widget and
  PNG loaders, static-file and router helpers, URL/header/query removal,
  `Audio::open` (a value used after being moved), LLM and LoRA/JiT ML
  paths (double frees), and safec-doc's link checker.

### Changed

- `safeguard test` in a library package (`staticlib`/`cdylib`) builds the
  library and links it into each test, with `src/` on the include path.
