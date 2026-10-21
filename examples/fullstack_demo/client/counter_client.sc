// fullstack_demo — WASM client: hydrates the /counter page's interactive
// counter (see src/pages.scx's render_counter()). Compiled *separately*
// from the rest of this project, straight to wasm32, together with the std
// sources it uses (#include names only headers; each .sc is its own
// translation unit) —
//
//   for f in client/counter_client.sc <SafeC>/std/wasm/wasm_rt.sc \
//            <SafeC>/std/wasm/dom.sc <SafeC>/std/wasm/hydrate.sc \
//            <SafeC>/std/mem.sc <SafeC>/std/panic.sc; do
//       safec "$f" -I <SafeC repo root> --target=wasm32-unknown-unknown \
//           --freestanding --emit-llvm -o "build/$(basename "$f" .sc).ll"
//   done
//   <llvm-with-wasm-target>/bin/clang --target=wasm32-unknown-unknown \
//       -nostdlib -fuse-ld=<path-to-wasm-ld> -Wl,--no-entry \
//       -Wl,--allow-undefined build/*.ll -o build/counter_client.wasm
//
// (Exports come from __attribute__((export_name)) — hydrate_init below and
// std/wasm/dom.h's dom_dispatch_event — plus the linear memory, which
// wasm-ld always exports.)
//
// It lives outside src/ so the hosted safeguard build never compiles it.
// main.sc serves the resulting build/counter_client.wasm as a static
// asset; see render_counter()'s <script> for the JS-side glue that
// provides the env.js_* imports this file calls into.
#include <std/wasm/wasm_rt.h>
#include <std/reactive/signal.h>
#include <std/wasm/dom.h>
#include <std/wasm/hydrate.h>

static struct Signal<int> gCounter;

static void on_increment(?&Signal<int> s) {
    match (s) { case some(c): c.set(c.get() + 1); case null: { } }
}

static void on_decrement(?&Signal<int> s) {
    match (s) { case some(c): c.set(c.get() - 1); case null: { } }
}

// Exported: the page's inline <script> calls this once, right after
// WebAssembly.instantiate() resolves.
__attribute__((export_name)) int hydrate_init() {
    gCounter = std::Signal::new(0);

    int countHandle = std::dom_get_element("#count");
    std::dom_set_text(countHandle, "0");
    std::hydrate_bind_text_int_handle(&gCounter, countHandle);

    unsafe {
        std::dom_on_click("#increment", on_increment, (?&Signal<int>)&gCounter);
        std::dom_on_click("#decrement", on_decrement, (?&Signal<int>)&gCounter);
    }

    return countHandle;
}
