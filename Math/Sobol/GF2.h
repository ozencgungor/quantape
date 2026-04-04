#pragma once
/**
 * @file GF2.h
 * @brief GF(2) polynomial arithmetic, irreducibility and primitivity testing.
 *
 * Polynomials are represented as uint64_t bitmasks:
 *   bit i = coefficient of x^i.
 *   e.g., x^3 + x + 1 = 0b1011 = 11.
 *
 * Supports degrees up to 63 (though Sobol extension only needs ~25).
 */

#include <cstdint>
#include <vector>

namespace mc {
namespace gf2 {

// ═══════════════════════════════════════════════════════════════════════════
// Core arithmetic
// ═══════════════════════════════════════════════════════════════════════════

inline int degree(uint64_t p) {
    return p == 0 ? -1 : 63 - __builtin_clzll(p);
}

inline uint64_t mul(uint64_t a, uint64_t b) {
    // Carry-less multiplication.
    // For production: use PCLMULQDQ via _mm_clmulepi64_si128 on x86.
    // This scalar fallback handles degrees up to ~31 safely in uint64_t.
    uint64_t r = 0;
    while (b) {
        if (b & 1) r ^= a;
        a <<= 1;
        b >>= 1;
    }
    return r;
}

#if defined(__PCLMUL__) && defined(__SSE2__)
#include <immintrin.h>
inline uint64_t mul_hw(uint64_t a, uint64_t b) {
    __m128i va = _mm_set_epi64x(0, a);
    __m128i vb = _mm_set_epi64x(0, b);
    __m128i r  = _mm_clmulepi64_si128(va, vb, 0x00);
    // For degree ≤ 31 inputs, result fits in low 64 bits.
    // For higher degrees, high bits spill into r[127:64].
    return _mm_extract_epi64(r, 0);
}
#endif

inline uint64_t mod(uint64_t a, uint64_t m) {
    int dm = degree(m);
    for (;;) {
        int da = degree(a);
        if (da < dm) return a;
        a ^= (m << (da - dm));
    }
}

inline uint64_t gcd(uint64_t a, uint64_t b) {
    while (b) { uint64_t t = mod(a, b); a = b; b = t; }
    return a;
}

// x^exp mod m  (x = polynomial 0b10)
inline uint64_t powmod_x(uint64_t exp, uint64_t m) {
    uint64_t base = mod(2, m);
    uint64_t result = 1;
    while (exp > 0) {
        if (exp & 1) result = mod(mul(result, base), m);
        base = mod(mul(base, base), m);
        exp >>= 1;
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// Factoring (trial division — sufficient for 2^s-1, s ≤ 30)
// ═══════════════════════════════════════════════════════════════════════════

inline std::vector<uint64_t> prime_factors(uint64_t n) {
    std::vector<uint64_t> f;
    for (uint64_t d = 2; d * d <= n; ++d) {
        if (n % d == 0) {
            f.push_back(d);
            while (n % d == 0) n /= d;
        }
    }
    if (n > 1) f.push_back(n);
    return f;
}

// ═══════════════════════════════════════════════════════════════════════════
// Irreducibility and primitivity tests
// ═══════════════════════════════════════════════════════════════════════════

// p(x) of degree s is irreducible over GF(2) iff:
//   1. x^(2^s) ≡ x  (mod p)
//   2. gcd(x^(2^(s/q)) - x, p) = 1 for each prime factor q of s
inline bool is_irreducible(uint64_t p, int s) {
    if (s <= 0) return false;

    // x^(2^s) mod p via repeated squaring of x
    uint64_t xpow = 2;
    for (int i = 0; i < s; ++i)
        xpow = mod(mul(xpow, xpow), p);
    if (xpow != 2) return false;

    for (uint64_t q : prime_factors(s)) {
        uint64_t xp = 2;
        for (int i = 0; i < s / (int)q; ++i)
            xp = mod(mul(xp, xp), p);
        if (gcd(xp ^ 2, p) != 1) return false;  // xp - x = xp ^ x = xp ^ 2 (polynomial "x" = 0b10)
    }
    return true;
}

// p(x) of degree s is primitive iff:
//   1. irreducible
//   2. ord(x mod p) = 2^s - 1
//      i.e., x^((2^s-1)/q) ≢ 1 (mod p) for each prime q | (2^s - 1)
inline bool is_primitive(uint64_t p, int s) {
    if (!is_irreducible(p, s)) return false;

    uint64_t order = (1ULL << s) - 1;
    for (uint64_t q : prime_factors(order)) {
        if (powmod_x(order / q, p) == 1) return false;
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// Enumeration of primitive polynomials
// ═══════════════════════════════════════════════════════════════════════════

// All primitive polynomials of degree s.
// Format: x^s + (middle bits) + 1, so leading and trailing bits are always 1.
inline std::vector<uint64_t> enumerate_primitive(int s) {
    std::vector<uint64_t> result;
    uint64_t hi = 1ULL << s;
    // Enumerate all possible middle coefficients
    for (uint64_t mid = 0; mid < (1ULL << (s - 1)); ++mid) {
        uint64_t poly = hi | (mid << 1) | 1;
        if (is_primitive(poly, s))
            result.push_back(poly);
    }
    return result;
}

// Count of primitive polynomials of degree s = φ(2^s - 1) / s
inline uint64_t count_primitive(int s) {
    uint64_t n = (1ULL << s) - 1;
    uint64_t phi = n;
    for (uint64_t q : prime_factors(n))
        phi = phi / q * (q - 1);
    return phi / s;
}

// ═══════════════════════════════════════════════════════════════════════════
// Joe-Kuo polynomial encoding
// ═══════════════════════════════════════════════════════════════════════════

// Joe-Kuo 'a' field: strip leading x^s and constant 1, keep middle bits
// poly = x^s + c_1 x^{s-1} + ... + c_{s-1} x + 1
// a = bits [1..s-1] of poly = (poly >> 1) & ((1 << (s-1)) - 1)
inline uint64_t encode_a(uint64_t poly, int s) {
    return (poly >> 1) & ((1ULL << (s - 1)) - 1);
}

// Recover polynomial from degree s and encoded a
inline uint64_t decode_poly(int s, uint64_t a) {
    return (1ULL << s) | (a << 1) | 1;
}

} // namespace gf2
} // namespace mc
