#pragma once
// SafeC Standard Library — LLM orchestration (LangChain/LangGraph/
// LangSmith-inspired).
//
// A thin layer over std::http/std::Value: LlmClient speaks the
// OpenAI-chat-completions wire shape (the de facto standard most local
// and hosted LLM servers — vLLM's own OpenAI-compatible server included —
// implement), PromptTemplate does '{var}' substitution, Chain runs a
// fixed step sequence, the Graph executor runs named nodes connected by
// fixed or conditional edges over a shared Value state (LangGraph's core
// idea), and Tracer keeps an in-memory, JSON-dumpable log of named
// input/output/timing events (LangSmith's core idea) — none of these
// call out to any hosted service themselves; verification in this
// session used a local mock HTTP server standing in for an LLM endpoint,
// the same technique std/rpc/server_fn.h's own tests used.
#include <std/serial/value.h>
#include <std/collections/vec.h>
#include <std/collections/string.h>

namespace std {

// ── LlmClient ────────────────────────────────────────────────────────────────
struct LlmMessage {
    struct String role;    // "system" | "user" | "assistant"
    struct String content;
};

struct LlmClient {
    struct String host;
    unsigned short port;
    struct String model;
    struct String apiKey;  // sent as 'Authorization: Bearer <apiKey>' when non-empty
};

struct LlmClient llm_client_new(const char* host, unsigned short port,
                                 const char* model, const char* apiKey);

// POSTs 'messages' (Vec<struct LlmMessage>) to
// '<client.host>:<client.port>/v1/chat/completions' as
// '{"model":..., "messages":[{"role":...,"content":...}, ...]}' and
// returns 'choices[0].message.content' from the JSON response.
// *ok is set to 0 on any transport, HTTP-status, or JSON-shape failure.
struct String llm_chat(struct LlmClient* client, struct Vec* messages, int* ok);

// ── PromptTemplate ───────────────────────────────────────────────────────────
// Replaces every '{name}' occurrence in 'tmpl' with 'vars' object field
// 'name' (stringified: strings pass through as-is, numbers/bools render
// via their natural text form). A '{name}' whose key isn't present in
// 'vars' is left as literal text, unexpanded.
struct String prompt_template_render(const char* tmpl, const struct Value* vars);

// ── Chain ────────────────────────────────────────────────────────────────────
// A fixed sequence of Value -> Value steps, run in order — the output of
// each step becomes the input to the next.
typedef fn struct Value(const struct Value* input) ChainStepFn;

struct Chain {
    struct Vec steps; // Vec<ChainStepFn>
};

struct Chain chain_new();
void chain_add_step(struct Chain* c, ChainStepFn step);
struct Value chain_run(struct Chain* c, const struct Value* input);
void chain_free(struct Chain* c);

// ── Graph executor ───────────────────────────────────────────────────────────
// Named nodes transform a shared Value state; each node either always
// advances to a fixed 'nextNode', or (if 'router' is set) calls 'router'
// on the post-node state to pick the next node by name. graph_run()
// starts at the graph's entry node and keeps following edges until a
// node returns/routes to "__end__", a named node isn't found (also
// treated as end), or 'maxSteps' is hit (guards against an accidental
// infinite loop in a misconfigured graph).
typedef fn struct Value(const struct Value* state) GraphNodeFn;
typedef fn struct String(const struct Value* state) GraphRouterFn;

struct GraphNode {
    struct String   name;
    GraphNodeFn     nodeFn;
    GraphRouterFn   router;    // NULL -> use 'nextNode' unconditionally
    struct String   nextNode;
};

struct Graph {
    struct Vec    nodes; // Vec<struct GraphNode>
    struct String entryNode;
};

struct Graph graph_new(const char* entryNode);
void graph_add_node(struct Graph* g, const char* name, GraphNodeFn nodeFn, const char* nextNode);
void graph_add_conditional_node(struct Graph* g, const char* name, GraphNodeFn nodeFn, GraphRouterFn router);
struct Value graph_run(struct Graph* g, struct Value initialState, int maxSteps);
void graph_free(struct Graph* g);

// ── Tracer ───────────────────────────────────────────────────────────────────
struct TraceEvent {
    struct String name;
    struct String inputJson;
    struct String outputJson;
    double        durationMs;
};

struct Tracer {
    struct Vec events; // Vec<struct TraceEvent>
};

struct Tracer tracer_new();
void tracer_record(struct Tracer* t, const char* name, const struct Value* input,
                    const struct Value* output, double durationMs);
// Dumps every recorded event as a JSON array: '[{"name":...,"input":...,
// "output":...,"durationMs":...}, ...]'.
struct String tracer_to_json(const struct Tracer* t);
void tracer_free(struct Tracer* t);

} // namespace std
