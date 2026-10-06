# infra/: logging facade. The facade test must see every level at compile
# time; -U overrides the project compile-time level (Release strips
# debug/trace by default).

add_executable(test_logging ${CMAKE_SOURCE_DIR}/tests/test_logging.cpp)
target_link_libraries(test_logging PRIVATE quantape)
target_compile_options(test_logging PRIVATE
    -UQUILL_COMPILE_ACTIVE_LOG_LEVEL
    -DQUILL_COMPILE_ACTIVE_LOG_LEVEL=QUILL_COMPILE_ACTIVE_LOG_LEVEL_TRACE_L3)
quantape_set_warnings(test_logging)
quantape_register_test(test_logging)

quantape_collect_executables()
