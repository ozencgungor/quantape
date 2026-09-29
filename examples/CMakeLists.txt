# Minimal usage example of the library.
add_executable(example_usage example_usage.cpp)
target_link_libraries(example_usage PRIVATE quantape)
quantape_set_warnings(example_usage)

quantape_collect_executables()
