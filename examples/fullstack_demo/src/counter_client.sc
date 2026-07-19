// fullstack_demo — WASM client: hydrates the /counter page's interactive
// counter (see src/pages.scx's render_counter()). Compiled *separately*
// from the rest of this project, straight to wasm32 —
//
//   safec src/counter_client.sc -I <SafeC repo root> \
//       --target=wasm32-unknown-unknown --compat-preprocessor \
//       --emit-llvm -o build/counter_client.ll
//   <llvm-with-wasm-target>/bin/clang --target=wasm32-unknown-unknown \
//       -nostdlib -fuse-ld=<path-to-wasm-ld> -Wl,--no-entry \
//       -Wl,--allow-undefined -Wl,--export=hydrate_init \
//       -Wl,--export=std_dom_dispatch_event -Wl,--export=memory \
//       build/counter_client.ll -o build/counter_client.wasm
//
// (not wired into Package.toml/[features] — safeguard's build system has
// no multi-target/cross-compile notion yet, same documented limitation
// as std/gui's Win32/X11 backends being type-checked but not build-
// system-integrated). main.sc serves the resulting build/counter_client.wasm
// as a static asset; see render_counter()'s <script> for the JS-side glue
// that provides the env.js_* imports this file calls into.
#include <std/wasm/wasm_rt.sc>
#include <std/reactive/signal.h>
#include <std/reactive/signal.sc>
#include <std/wasm/dom.h>
#include <std/wasm/dom.sc>
#include <std/wasm/hydrate.h>
#include <std/wasm/hydrate.sc>

static struct Signal gCounter;

static void on_increment(void* userdata) {
    struct Signal* s;
    unsafe { s = (struct Signal*)userdata; }
    int cur;
    unsafe { cur = signal_get_t(s, int); }
    unsafe { std::signal_set_t(s, cur + 1); }
}

static void on_decrement(void* userdata) {
    struct Signal* s;
    unsafe { s = (struct Signal*)userdata; }
    int cur;
    unsafe { cur = signal_get_t(s, int); }
    unsafe { std::signal_set_t(s, cur - 1); }
}

// Exported (--export=hydrate_init): the page's inline <script> calls this
// once, right after WebAssembly.instantiate() resolves.
int hydrate_init() {
    gCounter = std::signal_new(0);

    int countHandle = std::dom_get_element("#count");
    std::dom_set_text(countHandle, "0");
    std::hydrate_bind_text_int_handle(&gCounter, countHandle);

    unsafe { std::dom_on_click("#increment", on_increment, (void*)&gCounter); }
    unsafe { std::dom_on_click("#decrement", on_decrement, (void*)&gCounter); }

    return countHandle;
}
