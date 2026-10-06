// test_number_format.cpp — zmij wrapper: round trip, printf compatibility, hex
#include "quantape/format/Number.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

#include "support/GtestSupport.h"

using quantape::format::FloatFormat;

TEST(NumberFormat, toStringFormats) {
    EXPECT_EQ(quantape::format::toString(0.1), "0.1");
    EXPECT_EQ(quantape::format::toString(1.0), "1");
    EXPECT_EQ(quantape::format::toString(-2.5), "-2.5");
    EXPECT_EQ(quantape::format::toString(1.5e300, FloatFormat::Scientific, 2), "1.50e+300");
    EXPECT_EQ(quantape::format::toString(0.5, FloatFormat::Fixed, 3), "0.500");
    EXPECT_EQ(quantape::format::toString(1234.0, FloatFormat::General, 3), "1.23e+03");
    EXPECT_TRUE(quantape::format::toString(2.0, FloatFormat::Hex, -1).find("0x") ==
                std::string::npos);
}

TEST(NumberFormat, roundTripsRandomFiniteBitPatterns) {
    char buffer[quantape::format::kMaxDoubleChars];

    // Shortest form must round-trip every finite double bit pattern.
    std::mt19937_64 rng(20260928);
    int tested = 0;
    for (int i = 0; i < 200000; ++i) {
        const std::uint64_t bits = rng();
        double value = 0.0;
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value)) {
            continue;
        }
        char* end = quantape::format::write(value, buffer, sizeof(buffer));
        ASSERT_TRUE(end > buffer && end < buffer + sizeof(buffer));
        *end = '\0';
        char* parse_end = nullptr;
        const double back = std::strtod(buffer, &parse_end);
        EXPECT_EQ(parse_end, end);
        EXPECT_EQ(back, value);
        ++tested;
    }
    EXPECT_GT(tested, 190000);
    RecordProperty("tested", tested);
}

TEST(NumberFormat, matchesPrintfFixedPrecision) {
    // printf compatibility for fixed precision on a deterministic grid.
    for (int precision = 0; precision <= 12; ++precision) {
        SCOPED_TRACE(::testing::Message() << "precision=" << precision);
        double value = -1000.0;
        while (value <= 1000.0) {
            const std::string actual =
                quantape::format::toString(value, FloatFormat::Fixed, precision);
            char expected[quantape::format::kMaxDoubleChars];
            std::snprintf(expected, sizeof(expected), "%.*f", precision, value);
            EXPECT_EQ(actual, expected);
            value += 7.38905609893065;
        }
    }
}

TEST(NumberFormat, toCharsBufferExhaustion) {
    // Buffer exhaustion reports value_too_large without writing out of bounds.
    char tiny[4];
    const auto overflow = quantape::format::to_chars(tiny, tiny + sizeof(tiny), 1.0 / 3.0);
    EXPECT_EQ(overflow.ec, std::errc::value_too_large);
    const auto fits = quantape::format::to_chars(tiny, tiny + sizeof(tiny), 1.0);
    EXPECT_EQ(fits.ec, std::errc{});
}
