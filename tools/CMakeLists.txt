# Command-line tools. Python scripts under aws/ and market_data/ are not built.
# sobol_to_binary lives in the top-level CMakeLists: it is a build-time asset
# generator for the library, not a user tool.

add_executable(extend_sobol extend_sobol.cpp)
target_link_libraries(extend_sobol PRIVATE quantape_log)

add_executable(verify_sobol verify_sobol.cpp)
target_link_libraries(verify_sobol PRIVATE quantape_log)

add_executable(analyze_sobol analyze_sobol.cpp)
target_link_libraries(analyze_sobol PRIVATE quantape_log)

add_executable(refine_sobol refine_sobol.cpp)
target_link_libraries(refine_sobol PRIVATE quantape_log)

quantape_collect_executables()
