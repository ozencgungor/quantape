# datetime/: date, day count, calendar and schedule gates (no engine link).

foreach(_test test_date test_daycount test_calendar test_schedule)
    add_executable(${_test} ${CMAKE_SOURCE_DIR}/tests/${_test}.cpp)
    target_link_libraries(${_test} PRIVATE quantape_log quantape_format)
    quantape_set_warnings(${_test})
    quantape_register_test(${_test})
endforeach()

quantape_collect_executables()
