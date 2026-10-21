// SafeC Standard Library — Server functions implementation (see
// server_fn.h).
#pragma once
#include <std/rpc/server_fn.h>
#include <std/rpc/rpc.h>
#include <std/serial/value.h>
#include <std/serial/value.h>
#include <std/serial/json.h>
#include <std/serial/json.h>
#include <std/collections/string.h>
#include <std/collections/string.h>
#include <std/mem.h>

namespace std {

#define SERVER_FN_MAX 64

struct ServerFnEntry {
    struct String name;
    ServerFn handler;
};

static struct ServerFnEntry gServerFns[SERVER_FN_MAX];
static int gServerFnCount = 0;

void server_fn_register(const &char name, ServerFn handler) {
    if (gServerFnCount >= SERVER_FN_MAX) { return; }
    unsafe {
        gServerFns[gServerFnCount].name = String::from(name);
        gServerFns[gServerFnCount].handler = handler;
    }
    gServerFnCount = gServerFnCount + 1;
}

static ServerFn __server_fn_lookup(const char* name) {
    int i = 0;
    while (i < gServerFnCount) {
        int eq; unsafe { eq = gServerFns[i].name.eq_cstr(name); }
        if (eq) { ServerFn h; unsafe { h = gServerFns[i].handler; } return h; }
        i = i + 1;
    }
    return (ServerFn)0;
}

static struct RpcResult __server_fn_dispatch(const unsigned char* data, unsigned long len) {
    struct RpcResult result;
    result.ok = 0;
    result.data = (unsigned char*)0;
    result.len = 0UL;
    result.error_message = String::new();

    struct String reqText;
    unsafe { reqText = String::from_n((const char*)data, len); }

    int parseOk;
    struct Value root;
    unsafe { root = json_parse((const char*)reqText.data, &parseOk); }
    unsafe { reqText.free(); }
    if (!parseOk) {
        unsafe { result.error_message = String::from("malformed JSON request"); root.free(); }
        return result;
    }

    struct Value* nameVal; unsafe { nameVal = root.object_get("fn"); }
    if (nameVal == (struct Value*)0) {
        unsafe { result.error_message = String::from("missing 'fn' field"); root.free(); }
        return result;
    }
    const char* fnName; unsafe { fnName = nameVal->as_string(); }
    ServerFn handler = __server_fn_lookup(fnName);
    if (handler == (ServerFn)0) {
        unsafe { result.error_message = String::from("unknown server function"); root.free(); }
        return result;
    }

    struct Value* argsField; unsafe { argsField = root.object_get("args"); }
    struct Value argsVal;
    if (argsField == (struct Value*)0) {
        unsafe { argsVal = value_null(); }
    } else {
        unsafe { argsVal = argsField->clone(); }
    }

    struct Value resultVal;
    unsafe { resultVal = handler(&argsVal); }
    unsafe { argsVal.free(); }
    unsafe { root.free(); }

    struct String respText;
    unsafe { respText = value_to_json(&resultVal); }
    unsafe { resultVal.free(); }

    unsigned long respLen; unsafe { respLen = respText.len; }
    unsigned char* respBuf;
    unsafe {
        respBuf = (unsigned char*)alloc_raw_(respLen);
        unsigned long i = 0UL;
        while (i < respLen) { respBuf[i] = (unsigned char)respText.data[i]; i = i + 1UL; }
        respText.free();
    }

    result.ok = 1;
    result.data = respBuf;
    result.len = respLen;
    return result;
}

int server_fn_serve(unsigned short port) {
    rpc_register("/fn/call", __server_fn_dispatch);
    return rpc_serve(port);
}

struct Value server_fn_call(const &char host, unsigned short port, const &char name,
                             const ?&Value args, ?&int ok) {
    struct Value envelope;
    unsafe {
        envelope = value_object();
        envelope.object_set("fn", value_string(name));
        match (args) {
            case some(a): { envelope.object_set("args", a.clone()); }
            case null:    { envelope.object_set("args", value_null()); }
        }
    }
    struct String reqText;
    unsafe { reqText = value_to_json(&envelope); envelope.free(); }

    unsigned long reqLen; unsafe { reqLen = reqText.len; }
    struct RpcResult rr;
    unsafe { rr = rpc_call(host, port, "/fn/call", (const unsigned char*)reqText.data, reqLen); }
    unsafe { reqText.free(); }

    if (!rr.ok) {
        unsafe { *ok = 0; }
        unsafe { rr.free(); }
        return value_null();
    }

    struct String respText;
    unsafe { respText = String::from_n((const char*)rr.data, rr.len); }
    unsafe { rr.free(); }

    int parseOk;
    struct Value result;
    unsafe { result = json_parse((const char*)respText.data, &parseOk); respText.free(); }
    unsafe { *ok = parseOk; }
    return result;
}

} // namespace std
