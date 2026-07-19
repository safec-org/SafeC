#pragma once
// SafeC Standard Library — Server functions.
//
// The "write it once, call it from anywhere" pattern (Dioxus/Leptos/Next.js
// server actions) — a thin JSON-marshaling layer over std::rpc (see
// rpc.h). SafeC has no macros/codegen to make a single function
// definition automatically usable on both sides the way those frameworks'
// #[server]/"use server" attributes do — you still define the server-side
// implementation and register it, and a client calls it by name — but the
// wire format (JSON in, JSON out, via std::Value — see
// std/serial/value.h, the same tree json.h/csv.h/yaml.h use) is handled
// for you, so a "server function" is just:
//
//   struct Value add_fn(const struct Value* args) {
//       long long a = args->object_get("a")->as_int();
//       long long b = args->object_get("b")->as_int();
//       return value_int(a + b);
//   }
//   server_fn_register("add", add_fn);
//   server_fn_serve(8080);   // on the server
//
//   int ok;
//   struct Value args = value_object();
//   value_object_set(&args, "a", value_int(2));
//   value_object_set(&args, "b", value_int(3));
//   struct Value result = server_fn_call("localhost", 8080, "add", &args, &ok);
//   // result.as_int() == 5
//
// All registered functions are served over one RPC path ("/fn/call");
// the target function name travels inside the JSON payload
// ({"fn": "...", "args": ...}), not the URL, since std::rpc dispatches by
// exact path and a plain C function pointer (RpcMethodHandler) has no
// closure/userdata slot to carry a per-registration name through.
#include <std/serial/value.h>

namespace std {

typedef fn struct Value(const struct Value* args) ServerFn;

// Registers 'fn', callable as 'name' via server_fn_call()/over the wire.
// Call before server_fn_serve() — like rpc_register(), no registration
// after the server starts. Up to 64 server functions per process.
void server_fn_register(const char* name, ServerFn handler);

// Blocking accept loop (see rpc_serve()) serving every function
// server_fn_register()'d so far.
int server_fn_serve(unsigned short port);

// Calls the server function named 'name' at host:port with 'args' (may be
// NULL for a no-argument function — sent as a JSON null). *ok is set to 0
// (result is VAL_NULL) on any transport/handler/JSON failure.
struct Value server_fn_call(const char* host, unsigned short port, const char* name,
                             const struct Value* args, int* ok);

} // namespace std
