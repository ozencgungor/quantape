# instruments/: cashflow vocabulary, money-market and FX instruments. The AD
# gates instantiate Stan scalars; test_fx_quote is double-only.

add_executable(test_instruments ${CMAKE_SOURCE_DIR}/tests/test_instruments.cpp)
target_link_libraries(test_instruments PRIVATE quantape)
quantape_enable_stan(test_instruments)
quantape_set_warnings(test_instruments)
quantape_register_test(test_instruments)

add_executable(test_ir_instruments ${CMAKE_SOURCE_DIR}/tests/test_ir_instruments.cpp)
target_link_libraries(test_ir_instruments PRIVATE quantape)
quantape_enable_stan(test_ir_instruments)
quantape_set_warnings(test_ir_instruments)
quantape_register_test(test_ir_instruments)

# FX quote records and settlement helpers (double-based, no Stan scalars).
add_executable(test_fx_quote ${CMAKE_SOURCE_DIR}/tests/test_fx_quote.cpp)
target_link_libraries(test_fx_quote PRIVATE quantape)
quantape_set_warnings(test_fx_quote)
quantape_register_test(test_fx_quote)

# FX pricing, reset-aware cross-currency risk rows and FX stack risk; the
# double/var/fvar<var> gates instantiate Stan scalars.
add_executable(test_fx_pricing ${CMAKE_SOURCE_DIR}/tests/test_fx_pricing.cpp)
target_link_libraries(test_fx_pricing PRIVATE quantape)
quantape_enable_stan(test_fx_pricing)
quantape_set_warnings(test_fx_pricing)
quantape_register_test(test_fx_pricing)

# Dated FX instruments: settlement-lag factories, cashflow materialization,
# date-aware CIP/PV helpers and their var/fvar<var> gates.
add_executable(test_fx_instruments ${CMAKE_SOURCE_DIR}/tests/test_fx_instruments.cpp)
target_link_libraries(test_fx_instruments PRIVATE quantape)
quantape_enable_stan(test_fx_instruments)
quantape_set_warnings(test_fx_instruments)
quantape_register_test(test_fx_instruments)

quantape_collect_executables()
