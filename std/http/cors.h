#pragma once
// SafeC Standard Library — CORS (Cross-Origin Resource Sharing) helpers.
//
// Not a wrapping middleware combinator (std::http's HttpHandler is a bare
// function pointer with no closure/userdata slot — see std/rpc/server_fn.h's
// header comment for the same constraint) — call these directly from your
// own handler instead:
//
//   struct HttpResponse handle(struct HttpRequest* req) {
//       if (cors_is_preflight(req)) return cors_preflight_response("*", "GET, POST, OPTIONS", "Content-Type, Authorization");
//       struct HttpResponse resp = ...;
//       cors_apply_headers(&resp, "*");
//       return resp;
//   }
#include <std/http/http.h>
#include <std/collections/string.h>

namespace std {

// True if 'req' is a CORS preflight request (OPTIONS with an
// 'Access-Control-Request-Method' header).
int cors_is_preflight(const &HttpRequest req);

// Adds 'Access-Control-Allow-Origin: <allowedOrigin>' (and
// '...-Credentials: true' when 'allowedOrigin' isn't "*") to 'resp's
// headers — call on every response (not just preflights) so the browser
// accepts the actual response, not just the preflight.
void cors_apply_headers(&HttpResponse resp, const char* allowedOrigin);

// Builds a complete 204 response for a preflight OPTIONS request, with
// Allow-Origin/-Methods/-Headers/-Max-Age all set.
struct HttpResponse cors_preflight_response(const char* allowedOrigin,
                                             const char* allowedMethods,
                                             const char* allowedHeaders);

} // namespace std
