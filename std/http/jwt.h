#pragma once
// SafeC Standard Library — JWT (JSON Web Tokens), HS256.
//
// Encodes/verifies compact-serialization JWTs
// (base64url(header).base64url(payload).base64url(signature)) signed
// with HMAC-SHA256 (std::hmac_sha256 — see std/crypto/hmac.h). Only
// HS256 — no RS256/ES256 (those need RSA/ECDSA, out of scope here) — the
// header is always the fixed '{"alg":"HS256","typ":"JWT"}'.
#include <std/serial/value.h>
#include <std/collections/string.h>

namespace std {

// Builds and signs a JWT: header + 'claims' (a Value object — e.g.
// '{"sub":"alice","exp":1234567890}') as the payload, HMAC-SHA256'd with
// 'secret' under the fixed HS256 header. Returns the full compact-
// serialization token string.
struct String jwt_sign(const struct Value* claims, const char* secret);

// Verifies 'token's signature against 'secret' and that its header says
// HS256. Does NOT check 'exp'/'nbf'/etc. itself (claims validation is
// left to the caller — read them back from the returned Value, e.g.
// 'claims->object_get("exp")') — only the cryptographic signature and
// JWT structure (three base64url segments) are checked here. *ok is set
// to 0 (result is VAL_NULL) on a malformed token or signature mismatch;
// on success, returns the decoded payload Value (caller owns it — call
// value_free()).
struct Value jwt_verify(const char* token, const char* secret, int* ok);

} // namespace std
