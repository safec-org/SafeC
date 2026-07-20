// SafeC Standard Library — LLM orchestration implementation (see llm.h).
#pragma once
#include <std/ml/llm.h>
#include <std/serial/value.h>
#include <std/serial/value.sc>
#include <std/serial/json.h>
#include <std/serial/json.sc>
#include <std/collections/vec.h>
#include <std/collections/vec.sc>
#include <std/collections/string.h>
#include <std/collections/string.sc>
#include <std/http/http.h>
#include <std/http/http.sc>
#include <std/mem.sc>
#include <std/fmt.sc>
#include <std/convert.sc>
#include <std/str.sc>

namespace std {

// ── LlmClient ────────────────────────────────────────────────────────────────

struct LlmClient llm_client_new(const char* host, unsigned short port,
                                 const char* model, const char* apiKey) {
    struct LlmClient c;
    c.host = string_from(host);
    c.port = port;
    c.model = string_from(model);
    c.apiKey = string_from(apiKey);
    return c;
}

struct String llm_chat(&LlmClient client, &Vec messages, int* ok) {
    struct Value req;
    unsafe {
        req = value_object();
        value_object_set(&req, "model", value_string(client.model.as_ptr()));
        struct Value msgArr = value_array();
        unsigned long n = messages.length();
        unsigned long i = 0UL;
        while (i < n) {
            struct LlmMessage* m = (struct LlmMessage*)messages.get_raw(i);
            struct Value msgObj = value_object();
            value_object_set(&msgObj, "role", value_string(m->role.as_ptr()));
            value_object_set(&msgObj, "content", value_string(m->content.as_ptr()));
            value_array_push(&msgArr, msgObj);
            i = i + 1UL;
        }
        value_object_set(&req, "messages", msgArr);
    }

    struct String body;
    unsafe { body = value_to_json(&req); value_free(&req); }

    struct String headers = string_new();
    unsafe {
        int hasKey = client.apiKey.length() > 0UL;
        if (hasKey) {
            headers.push("Authorization: Bearer ");
            headers.push(client.apiKey.as_ptr());
            headers.push("\r\n");
        }
    }

    int httpOk = 0;
    struct HttpResponse resp;
    unsafe {
        resp = http_request(client.host.as_ptr(), client.port, "POST", "/v1/chat/completions",
                             headers.as_ptr(), (const unsigned char*)body.data, body.len, &httpOk);
    }
    unsafe { headers.free(); body.free(); }

    if (!httpOk || resp.status < 200 || resp.status >= 300) {
        unsafe { *ok = 0; resp.free(); }
        return string_new();
    }

    int parseOk;
    struct Value root;
    unsafe { root = json_parse(resp.body.as_ptr(), &parseOk); resp.free(); }
    if (!parseOk) {
        unsafe { *ok = 0; value_free(&root); }
        return string_new();
    }

    struct String result = string_new();
    unsafe {
        struct Value* choices = root.object_get("choices");
        if (choices != (struct Value*)0 && choices->array_len() > 0UL) {
            struct Value* first = choices->array_at(0UL);
            struct Value* message = first->object_get("message");
            if (message != (struct Value*)0) {
                struct Value* content = message->object_get("content");
                if (content != (struct Value*)0) {
                    result = string_from(content->as_string());
                    *ok = 1;
                } else { *ok = 0; }
            } else { *ok = 0; }
        } else { *ok = 0; }
        value_free(&root);
    }
    return result;
}

// ── PromptTemplate ───────────────────────────────────────────────────────────

struct String prompt_template_render(const char* tmpl, const &Value vars) {
    struct String out = string_new();
    unsafe {
        unsigned long i = 0UL;
        unsigned long len = 0UL;
        while (tmpl[len] != (char)0) { len = len + 1UL; }
        while (i < len) {
            if (tmpl[i] == '{') {
                unsigned long j = i + 1UL;
                while (j < len && tmpl[j] != '}') { j = j + 1UL; }
                if (j < len) {
                    unsigned long keyLen = j - i - 1UL;
                    struct String key = string_from_n((const char*)tmpl + i + 1UL, keyLen);
                    struct Value* v = vars.object_get(key.as_ptr());
                    if (v != (struct Value*)0) {
                        if (v->kind == VAL_STRING) { out.push(v->as_string()); }
                        else if (v->kind == VAL_INT) { out.push_int(v->as_int()); }
                        else if (v->kind == VAL_FLOAT) { out.push_float(v->as_float(), 6); }
                        else if (v->kind == VAL_BOOL) { out.push_bool(v->as_bool()); }
                        else { out.push("{"); out.push(key.as_ptr()); out.push("}"); }
                    } else {
                        out.push("{"); out.push(key.as_ptr()); out.push("}");
                    }
                    key.free();
                    i = j + 1UL;
                } else {
                    out.push_char(tmpl[i]);
                    i = i + 1UL;
                }
            } else {
                out.push_char(tmpl[i]);
                i = i + 1UL;
            }
        }
    }
    return out;
}

// ── Chain ────────────────────────────────────────────────────────────────────

struct Chain chain_new() {
    struct Chain c;
    unsafe { c.steps = vec_new(sizeof(ChainStepFn)); }
    return c;
}

void chain_add_step(&Chain c, ChainStepFn step) {
    unsafe { c.steps.push((const void*)&step); }
}

struct Value chain_run(&Chain c, const &Value input) {
    struct Value current;
    unsafe { current = value_clone(input); }
    unsigned long n;
    unsafe { n = c.steps.length(); }
    unsigned long i = 0UL;
    while (i < n) {
        unsafe {
            ChainStepFn* fp = (ChainStepFn*)c.steps.get_raw(i);
            struct Value next = (*fp)(&current);
            value_free(&current);
            current = next;
        }
        i = i + 1UL;
    }
    return current;
}

void chain_free(&Chain c) {
    unsafe { c.steps.free(); }
}

// ── Graph executor ───────────────────────────────────────────────────────────

struct Graph graph_new(const char* entryNode) {
    struct Graph g;
    unsafe { g.nodes = vec_new(sizeof(struct GraphNode)); }
    g.entryNode = string_from(entryNode);
    return g;
}

static void __graph_add(&Graph g, const char* name, GraphNodeFn nodeFn,
                         GraphRouterFn router, const char* nextNode) {
    struct GraphNode node;
    node.name = string_from(name);
    node.nodeFn = nodeFn;
    node.router = router;
    node.nextNode = string_from(nextNode);
    unsafe { g.nodes.push((const void*)&node); }
}

void graph_add_node(&Graph g, const char* name, GraphNodeFn nodeFn, const char* nextNode) {
    __graph_add(g, name, nodeFn, (GraphRouterFn)0, nextNode);
}

void graph_add_conditional_node(&Graph g, const char* name, GraphNodeFn nodeFn, GraphRouterFn router) {
    __graph_add(g, name, nodeFn, router, "__end__");
}

static struct GraphNode* __graph_find(&Graph g, const char* name) {
    unsigned long n;
    unsafe { n = g.nodes.length(); }
    unsigned long i = 0UL;
    while (i < n) {
        unsafe {
            struct GraphNode* node = (struct GraphNode*)g.nodes.get_raw(i);
            if (node->name.eq_cstr(name)) return node;
        }
        i = i + 1UL;
    }
    return (struct GraphNode*)0;
}

struct Value graph_run(&Graph g, struct Value initialState, int maxSteps) {
    struct Value state = initialState;
    struct String currentName;
    unsafe { currentName = string_from(g.entryNode.as_ptr()); }

    int step = 0;
    while (step < maxSteps) {
        int isEnd;
        unsafe { isEnd = currentName.eq_cstr("__end__"); }
        if (isEnd) break;

        struct GraphNode* node;
        unsafe { node = __graph_find(g, currentName.as_ptr()); }
        if (node == (struct GraphNode*)0) break;

        struct Value nextState;
        unsafe { nextState = node->nodeFn(&state); value_free(&state); }
        state = nextState;

        struct String nextName;
        unsafe {
            if (node->router != (GraphRouterFn)0) {
                nextName = node->router(&state);
            } else {
                nextName = string_from(node->nextNode.as_ptr());
            }
        }
        unsafe { currentName.free(); }
        currentName = nextName;
        step = step + 1;
    }
    unsafe { currentName.free(); }
    return state;
}

void graph_free(&Graph g) {
    unsigned long n;
    unsafe { n = g.nodes.length(); }
    unsigned long i = 0UL;
    while (i < n) {
        unsafe {
            struct GraphNode* node = (struct GraphNode*)g.nodes.get_raw(i);
            node->name.free();
            node->nextNode.free();
        }
        i = i + 1UL;
    }
    unsafe { g.nodes.free(); }
    unsafe { g.entryNode.free(); }
}

// ── Tracer ───────────────────────────────────────────────────────────────────

struct Tracer tracer_new() {
    struct Tracer t;
    unsafe { t.events = vec_new(sizeof(struct TraceEvent)); }
    return t;
}

void tracer_record(&Tracer t, const char* name, const &Value input,
                    const &Value output, double durationMs) {
    struct TraceEvent ev;
    ev.name = string_from(name);
    unsafe { ev.inputJson = value_to_json(input); ev.outputJson = value_to_json(output); }
    ev.durationMs = durationMs;
    unsafe { t.events.push((const void*)&ev); }
}

struct String tracer_to_json(const &Tracer t) {
    struct String out = string_new();
    unsafe {
        out.push("[");
        unsigned long n = t.events.length();
        unsigned long i = 0UL;
        while (i < n) {
            struct TraceEvent* ev = (struct TraceEvent*)t.events.get_raw(i);
            if (i > 0UL) out.push(",");
            out.push("{\"name\":\"");
            out.push(ev->name.as_ptr());
            out.push("\",\"input\":");
            out.push(ev->inputJson.as_ptr());
            out.push(",\"output\":");
            out.push(ev->outputJson.as_ptr());
            out.push(",\"durationMs\":");
            out.push_float(ev->durationMs, 3);
            out.push("}");
            i = i + 1UL;
        }
        out.push("]");
    }
    return out;
}

void tracer_free(&Tracer t) {
    unsigned long n;
    unsafe { n = t.events.length(); }
    unsigned long i = 0UL;
    while (i < n) {
        unsafe {
            struct TraceEvent* ev = (struct TraceEvent*)t.events.get_raw(i);
            ev->name.free();
            ev->inputJson.free();
            ev->outputJson.free();
        }
        i = i + 1UL;
    }
    unsafe { t.events.free(); }
}

} // namespace std
