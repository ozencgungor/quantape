# markets/: curve, quote, futures and survival gates. All plain except
# test_markets, which instantiates `fvar<var>` for the survival curve.

add_executable(test_curve ${CMAKE_SOURCE_DIR}/tests/test_curve.cpp)
target_link_libraries(test_curve PRIVATE quantape)
quantape_set_warnings(test_curve)
quantape_register_test(test_curve)

add_executable(test_instrument_options ${CMAKE_SOURCE_DIR}/tests/test_instrument_options.cpp)
target_link_libraries(test_instrument_options PRIVATE quantape)
quantape_set_warnings(test_instrument_options)
quantape_register_test(test_instrument_options)

add_executable(test_forecast_pillars ${CMAKE_SOURCE_DIR}/tests/test_forecast_pillars.cpp)
target_link_libraries(test_forecast_pillars PRIVATE quantape)
quantape_set_warnings(test_forecast_pillars)
quantape_register_test(test_forecast_pillars)

add_executable(test_synthetic_quotes ${CMAKE_SOURCE_DIR}/tests/test_synthetic_quotes.cpp)
target_link_libraries(test_synthetic_quotes PRIVATE quantape)
quantape_set_warnings(test_synthetic_quotes)
quantape_register_test(test_synthetic_quotes)

add_executable(test_futures ${CMAKE_SOURCE_DIR}/tests/test_futures.cpp)
target_link_libraries(test_futures PRIVATE quantape)
quantape_set_warnings(test_futures)
quantape_register_test(test_futures)

add_executable(test_bootstrap_validation ${CMAKE_SOURCE_DIR}/tests/test_bootstrap_validation.cpp)
target_link_libraries(test_bootstrap_validation PRIVATE quantape)
quantape_set_warnings(test_bootstrap_validation)
quantape_register_test(test_bootstrap_validation)

# test_markets instantiates `fvar<var>` for the survival-probability curve.
add_executable(test_markets ${CMAKE_SOURCE_DIR}/tests/test_markets.cpp)
target_link_libraries(test_markets PRIVATE quantape)
quantape_enable_stan(test_markets)
quantape_set_warnings(test_markets)
quantape_register_test(test_markets)

quantape_collect_executables()
