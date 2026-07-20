#pragma once
// SafeC Standard Library — OAuth2 client (authorization-code + refresh
// grants, RFC 6749).
//
// Provider-agnostic: every function here just POSTs
// 'application/x-www-form-urlencoded' to whatever token endpoint
// host/port/path you give it, so it works against any real OAuth2
// provider's token endpoint (Google, GitHub, a self-hosted
// authorization server, ...) — nothing here is tied to one provider.
// This is the token-exchange half only; building the authorization
// *redirect* URL (the browser-facing '/authorize?...' step) is one
// string-format call away and deliberately left to the caller, since
// it's provider-specific query parameters, not a protocol exchange.
#include <std/serial/value.h>
#include <std/collections/string.h>

namespace std {

struct OAuth2Token {
    struct String accessToken;
    struct String refreshToken; // empty if the provider didn't return one
    struct String tokenType;    // usually "Bearer"
    long long     expiresIn;    // seconds; 0 if not returned
};

// Exchanges an authorization code for tokens: POSTs
// 'grant_type=authorization_code&code=...&redirect_uri=...&client_id=...
// &client_secret=...' to '<tokenHost>:<tokenPort><tokenPath>'.
// *ok is set to 0 on any transport/HTTP-status/JSON-shape failure.
struct OAuth2Token oauth2_exchange_code(const char* tokenHost, unsigned short tokenPort,
                                        const char* tokenPath, const char* clientId,
                                        const char* clientSecret, const char* code,
                                        const char* redirectUri, int* ok);

// Exchanges a refresh token for a fresh access token
// ('grant_type=refresh_token&refresh_token=...&client_id=...&client_secret=...').
struct OAuth2Token oauth2_refresh(const char* tokenHost, unsigned short tokenPort,
                                  const char* tokenPath, const char* clientId,
                                  const char* clientSecret, const char* refreshToken, int* ok);

void oauth2_token_free(&OAuth2Token t);

} // namespace std
