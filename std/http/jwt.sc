// SafeC Standard Library — JWT implementation (see jwt.h).
#pragma once
#include <std/http/jwt.h>
#include <std/crypto/hmac.h>
#include <std/crypto/hmac.h>
#include <std/encoding/base64.h>
#include <std/encoding/base64.h>
#include <std/serial/value.h>
#include <std/serial/value.h>
#include <std/serial/json.h>
#include <std/serial/json.h>
#include <std/collections/string.h>
#include <std/collections/string.h>
#include <std/collections/vec.h>
#include <std/collections/vec.h>
#include <std/mem.h>
#include <std/str.h>
#include <std/str.h>

namespace std {

static struct String __b64url_encode(const unsigned char* data, unsigned long len) {
    struct String s; unsafe { s = base64_encode(data, len); }
    unsafe {
        s.replace_char('+', '-');
        s.replace_char('/', '_');
        while (s.length() > 0UL && s.char_at(s.length() - 1UL) == (int)'=') {
            s.pop_char();
        }
    }
    return s;
}

static struct Vec<unsigned char> __b64url_decode(const char* s, int* ok) {
    struct String padded; unsafe { padded = String::from(s); }
    unsafe {
        padded.replace_char('-', '+');
        padded.replace_char('_', '/');
        while (padded.length() % 4UL != 0UL) { padded.push_char('='); }
    }
    struct Vec<unsigned char> out;
    unsafe { out = base64_decode(padded.as_ptr(), ok); }
    unsafe { padded.free(); }
    return out;
}

struct String jwt_sign(const &Value claims, const &char secret) {
    struct Value header;
    unsafe {
        header = Value::object();
        header.object_set("alg", Value::from_str("HS256"));
        header.object_set("typ", Value::from_str("JWT"));
    }
    struct String headerJson; struct String payloadJson;
    unsafe {
        headerJson = value_to_json(&header);
        payloadJson = value_to_json(claims);
        header.free();
    }

    struct String headerB64;
    struct String payloadB64;
    unsafe {
        headerB64 = __b64url_encode((const unsigned char*)headerJson.data, headerJson.len);
        payloadB64 = __b64url_encode((const unsigned char*)payloadJson.data, payloadJson.len);
        headerJson.free(); payloadJson.free();
    }

    struct String signingInput = String::new();
    unsafe {
        signingInput.push_str(&headerB64);
        signingInput.push_char('.');
        signingInput.push_str(&payloadB64);
    }

    unsigned char sig[HMAC_SHA256_SIZE];
    unsafe {
        hmac_sha256((const unsigned char*)secret, str_len(secret),
                    (const unsigned char*)signingInput.data, signingInput.len, sig);
    }
    struct String sigB64 = __b64url_encode((const unsigned char*)sig, (unsigned long)HMAC_SHA256_SIZE);

    struct String token = String::new();
    unsafe {
        token.push_str(&signingInput);
        token.push_char('.');
        token.push_str(&sigB64);
    }
    unsafe { headerB64.free(); payloadB64.free(); signingInput.free(); sigB64.free(); }
    return token;
}

// Finds the index of the Nth (0-based) occurrence of 'c' in 's', or -1.
static long long __nth_char_index(const char* s, char c, int n) {
    long long idx = -1LL;
    int found = 0;
    unsigned long i = 0UL;
    unsafe {
        while (s[i] != (char)0) {
            if (s[i] == c) {
                if (found == n) { idx = (long long)i; break; }
                found = found + 1;
            }
            i = i + 1UL;
        }
    }
    return idx;
}

struct Value jwt_verify(const &char token_ref, const &char secret, ?&int ok) { const char* token = token_ref;
    long long dot1 = __nth_char_index(token, '.', 0);
    long long dot2 = __nth_char_index(token, '.', 1);
    if (dot1 < 0LL || dot2 < 0LL) {
        unsafe { *ok = 0; }
        return Value::null();
    }

    struct String headerB64;
    struct String payloadB64;
    struct String sigB64;
    struct String signingInput;
    unsafe {
        headerB64 = String::from_n(token, (unsigned long)dot1);
        payloadB64 = String::from_n((const char*)token + dot1 + 1LL, (unsigned long)(dot2 - dot1 - 1LL));
        sigB64 = String::from((const char*)token + dot2 + 1LL);
        signingInput = String::from_n(token, (unsigned long)dot2);
    }

    // Header must decode to alg="HS256" — the only algorithm this file
    // signs with or trusts; a token's own claim to use anything else
    // (including "none") is rejected, not honored.
    int headerOk;
    struct Vec<unsigned char> headerBytes = __b64url_decode(headerB64.as_ptr(), &headerOk);
    int headerValid = 0;
    if (headerOk) {
        struct String headerJson;
        unsafe { headerJson = String::from_n((const char*)headerBytes.data, headerBytes.len); }
        int parseOk;
        struct Value header;
        unsafe { header = json_parse(headerJson.as_ptr(), &parseOk); headerJson.free(); }
        if (parseOk) {
            struct Value* alg;
            unsafe { alg = header.object_get("alg"); }
            if (alg != (struct Value*)0) {
                const char* algStr;
                unsafe { algStr = alg->as_string(); }
                unsafe { headerValid = str_eq(algStr, "HS256"); }
            }
        }
        unsafe { header.free(); }
    }
    unsafe { headerBytes.free(); }

    unsigned char expectedSig[HMAC_SHA256_SIZE];
    unsafe {
        hmac_sha256((const unsigned char*)secret, str_len(secret),
                    (const unsigned char*)signingInput.data, signingInput.len, expectedSig);
    }
    struct String expectedSigB64 = __b64url_encode((const unsigned char*)expectedSig, (unsigned long)HMAC_SHA256_SIZE);

    int sigMatch;
    unsafe { sigMatch = sigB64.eq_cstr(expectedSigB64.as_ptr()); }

    struct Value result = Value::null();
    if (headerValid && sigMatch) {
        int payloadOk;
        struct Vec<unsigned char> payloadBytes = __b64url_decode(payloadB64.as_ptr(), &payloadOk);
        if (payloadOk) {
            struct String payloadJson;
            unsafe { payloadJson = String::from_n((const char*)payloadBytes.data, payloadBytes.len); }
            int parseOk;
            unsafe { result = json_parse(payloadJson.as_ptr(), &parseOk); payloadJson.free(); }
            unsafe { *ok = parseOk; }
        } else {
            unsafe { *ok = 0; }
        }
        unsafe { payloadBytes.free(); }
    } else {
        unsafe { *ok = 0; }
    }

    unsafe {
        headerB64.free(); payloadB64.free(); sigB64.free(); signingInput.free();
        expectedSigB64.free();
    }
    return result;
}

} // namespace std
