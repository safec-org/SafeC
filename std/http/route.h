// SafeC Standard Library — URL path patterns (the matching half of
// std/http/router.h, usable on its own: no sockets, so a wasm32 client can
// route on location.pathname with the same patterns the server uses).
//
// Pattern syntax, segment by segment ('/'-separated):
//   literal      "users"      matches exactly that segment
//   :name        ":id"        matches any one non-empty segment, captured as "id"
//   *            "*"          (last segment only) matches the rest of the path,
//                             possibly empty, captured as "*"
//   *name        "*path"      the same, captured as "path"
// A trailing '/' on either side is ignored ("/a/" matches "/a"), except
// that "/" itself is the root. Captured values are percent-decoded.
#pragma once
#include <std/collections/string.h>
#include <std/collections/vec.h>

namespace std {

struct RouteParam {
    struct String name;
    struct String value;   // percent-decoded
};

// Matches 'path' (no query string) against 'pattern', appending the
// captures to 'params' on success. Returns 0 for no match, or a positive
// specificity score — higher for more literal segments earlier on — that
// the router uses to prefer "/users/me" over "/users/:id" over "/users/*".
long long http_route_match(const &char pattern, const &char path, &Vec<RouteParam> params);

// The value captured as 'name', or null.
const ?&char http_route_param(const &Vec<RouteParam> params, const &char name);

// Frees every capture and empties the vector (keeps its buffer).
void http_route_params_clear(&Vec<RouteParam> params);

} // namespace std
