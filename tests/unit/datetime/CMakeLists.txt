# datetime/: date, day count, calendar and schedule gates (plain, smoke tier).
quantape_add_gtest(qta_test_datetime
    SOURCES test_date.cpp test_daycount.cpp test_calendar.cpp test_schedule.cpp
    LABELS unit datetime fast smoke)

quantape_collect_executables()
