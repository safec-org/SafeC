// SafeC Standard Library — LLM orchestration implementation (see llm.h).
#pragma once
#include <std/ml/llm.h>
#include <std/serial/value.h>
#include <std/serial/value.h>
#include <std/serial/json.h>
#include <std/serial/json.h>
#include <std/collections/vec.h>
#include <std/collections/vec.h>
#include <std/collections/string.h>
#include <std/collections/string.h>
#include <std/http/http.h>
#include <std/http/http.h>
#include <std/mem.h>
#include <std/fmt.h>
#include <std/convert.h>
#include <std/str.h>

namespace std {

// ── LlmClient ────────────────────────────────────────────────────────────────

struct LlmClient LlmClient::new(const &char host, unsigned short port, const &char model, const &char apiKey) {
    struct LlmClient c;
    c.host = string_from(host);
    c.port = port;
    c.model = string_from(model);
    c.apiKey = string_from(apiKey);
    return c;
}

struct String LlmClient::chat(&Vec<struct LlmMessage> messages, ?&int ok) {
    struct Value req;
    unsafe {
        req = value_object();
        value_object_set(&req, "model", value_string(self.model.as_ptr()));
        struct Value msgArr = value_array();
        unsigned long n = messages.length();
        unsigned long i = 0UL;
        while (i < n) {
            struct LlmMessage* m = messages.at(i);
            struct Value msgObj = value_object();
            value_object_set(&msgObj, "role", value_string(m->role.as_ptr()));
            value_object_set(&msgObj, "content", value_string(m->content.as_ptr()));
            value_array_push(&msgArr, msgObj);
            i = i + 1UL;
        }
        value_object_set(&req, "messages", msgArr);
    }

    struct String body;
    unsafe { body = value_to_json(&req); req.free(); }

    struct String headers = string_new();
    unsafe {
        int hasKey = self.apiKey.length() > 0UL;
        if (hasKey) {
            headers.push("Authorization: Bearer ");
            headers.push(self.apiKey.as_ptr());
            headers.push("\r\n");
        }
    }

    int httpOk = 0;
    struct HttpResponse resp;
    unsafe {
        resp = http_request(self.host.as_ptr(), self.port, "POST", "/v1/chat/completions",
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
        unsafe { *ok = 0; root.free(); }
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
        root.free();
    }
    return result;
}

// ── PromptTemplate ───────────────────────────────────────────────────────────

struct String prompt_template_render(const &char tmpl_ref, const &Value vars) { const char* tmpl = tmpl_ref;
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

struct Chain Chain::new() {
    struct Chain c;
    unsafe { c.steps = vec_new(); }
    return c;
}

void Chain::add_step(ChainStepFn step) {
    unsafe { self.steps.push(step); }
}

struct Value Chain::run(const &Value input) {
    struct Value current;
    unsafe { current = value_clone(input); }
    unsigned long n;
    unsafe { n = self.steps.length(); }
    unsigned long i = 0UL;
    while (i < n) {
        unsafe {
            ChainStepFn* fp = self.steps.at(i);
            struct Value next = (*fp)(&current);
            current.free();
            current = next;
        }
        i = i + 1UL;
    }
    return current;
}

void Chain::free() { self.drop(); }

void Chain::drop() {
    unsafe { self.steps.free(); }
}

// ── Graph executor ───────────────────────────────────────────────────────────

struct Graph Graph::new(const &char entryNode) {
    struct Graph g;
    unsafe { g.nodes = vec_new(); }
    g.entryNode = string_from(entryNode);
    return g;
}

static void __graph_add(&Graph g, const char* name, GraphNodeFn nodeFn,
                         GraphRouterFn router, const char* nextNode) {
    struct GraphNode node;
    unsafe { node.name = string_from(name); }
    node.nodeFn = nodeFn;
    node.router = router;
    unsafe { node.nextNode = string_from(nextNode); }
    unsafe { g.nodes.push(node); }
}

void Graph::add_node(const &char name, GraphNodeFn nodeFn, const &char nextNode) {
    __graph_add(self, name, nodeFn, (GraphRouterFn)0, nextNode);
}

void Graph::add_conditional_node(const &char name, GraphNodeFn nodeFn, GraphRouterFn router) {
    __graph_add(self, name, nodeFn, router, "__end__");
}

static struct GraphNode* __graph_find(&Graph g, const char* name) {
    unsigned long n;
    unsafe { n = g.nodes.length(); }
    unsigned long i = 0UL;
    while (i < n) {
        unsafe {
            struct GraphNode* node = g.nodes.at(i);
            if (node->name.eq_cstr(name)) return node;
        }
        i = i + 1UL;
    }
    return (struct GraphNode*)0;
}

struct Value Graph::run(struct Value initialState, int maxSteps) {
    struct Value state = initialState;
    struct String currentName;
    unsafe { currentName = string_from(self.entryNode.as_ptr()); }

    int step = 0;
    while (step < maxSteps) {
        int isEnd;
        unsafe { isEnd = currentName.eq_cstr("__end__"); }
        if (isEnd) break;

        struct GraphNode* node;
        unsafe { node = __graph_find(self, currentName.as_ptr()); }
        if (node == (struct GraphNode*)0) break;

        struct Value nextState;
        unsafe { nextState = node->nodeFn(&state); state.free(); }
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

void Graph::free() { self.drop(); }

void Graph::drop() {
    unsigned long n;
    unsafe { n = self.nodes.length(); }
    unsigned long i = 0UL;
    while (i < n) {
        unsafe {
            struct GraphNode* node = self.nodes.at(i);
            node->name.free();
            node->nextNode.free();
        }
        i = i + 1UL;
    }
    unsafe { self.nodes.free(); }
    unsafe { self.entryNode.free(); }
}

// ── Tracer ───────────────────────────────────────────────────────────────────

struct Tracer Tracer::new() {
    struct Tracer t;
    unsafe { t.events = vec_new(); }
    return t;
}

void Tracer::record(const &char name, const &Value input, const &Value output, double durationMs) {
    struct TraceEvent ev;
    ev.name = string_from(name);
    unsafe { ev.inputJson = value_to_json(input); ev.outputJson = value_to_json(output); }
    ev.durationMs = durationMs;
    unsafe { self.events.push(ev); }
}

struct String Tracer::to_json() const {
    struct String out = string_new();
    unsafe {
        out.push("[");
        unsigned long n = self.events.length();
        unsigned long i = 0UL;
        while (i < n) {
            struct TraceEvent* ev = self.events.at(i);
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

void Tracer::free() { self.drop(); }

void Tracer::drop() {
    unsigned long n;
    unsafe { n = self.events.length(); }
    unsigned long i = 0UL;
    while (i < n) {
        unsafe {
            struct TraceEvent* ev = self.events.at(i);
            ev->name.free();
            ev->inputJson.free();
            ev->outputJson.free();
        }
        i = i + 1UL;
    }
    unsafe { self.events.free(); }
}

} // namespace std
