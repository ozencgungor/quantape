#pragma once

// Shared test harness: logging + number formatting for every test.
//
// Include this header and drop any local CHECK macro definition. Failures are
// reported through the logging facade (component quantape.test) and exit
// without static teardown: failing while Stan nested tapes are still live
// would otherwise crash during exit-time destruction instead of reporting.

#include "quantape/format/Number.h"
#include "quantape/log/Log.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#ifndef CHECK
#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            QTA_LOG_ERROR("test", "CHECK failed at {}:{}: {}", __FILE__, __LINE__,        \
                          #cond);                                                                  \
            ::quantape::log::shutdown();                                                           \
            std::fflush(nullptr);                                                                  \
            std::_Exit(1);                                                                         \
        }                                                                                          \
    } while (0)
#endif

namespace quantape_test {

/// Shortest round-trip text for a double, for status and diagnostic messages.
inline std::string num(double value) {
    return quantape::format::toString(value);
}

/// Fixed-precision text for approximate statistics (printf-compatible).
inline std::string num(double value, int precision) {
    return quantape::format::toString(value, quantape::format::FloatFormat::General, precision);
}

} // namespace quantape_test
