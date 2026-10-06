# infra/: logging facade. The facade test must see every level at compile
# time; -U overrides the project compile-time level (Release strips
# debug/trace by default). Flag order matters: undefine before define.
quantape_add_gtest(qta_test_infra
    SOURCES test_logging.cpp
    FLAGS -UQUILL_COMPILE_ACTIVE_LOG_LEVEL
          -DQUILL_COMPILE_ACTIVE_LOG_LEVEL=QUILL_COMPILE_ACTIVE_LOG_LEVEL_TRACE_L3
    LABELS unit infra fast)

quantape_collect_executables()
