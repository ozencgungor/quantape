# markets/ad: AD curve bootstrap, the paper EUR replication fixture and the
# portfolio risk stack. test_paper_eur_curves is double-only; the rest link
# full Stan. test_curve_ad is the sole writer of the portfolio risk table.

add_executable(test_curve_risk ${CMAKE_SOURCE_DIR}/tests/test_curve_risk.cpp)
target_link_libraries(test_curve_risk PRIVATE quantape)
quantape_set_warnings(test_curve_risk)
quantape_register_test(test_curve_risk)

add_executable(test_xccy ${CMAKE_SOURCE_DIR}/tests/test_xccy.cpp)
target_link_libraries(test_xccy PRIVATE quantape)
quantape_set_warnings(test_xccy)
quantape_register_test(test_xccy)

add_executable(test_curve_risk_report ${CMAKE_SOURCE_DIR}/tests/test_curve_risk_report.cpp)
target_link_libraries(test_curve_risk_report PRIVATE quantape)
quantape_set_warnings(test_curve_risk_report)
quantape_register_test(test_curve_risk_report)

add_executable(test_hull_white_convexity ${CMAKE_SOURCE_DIR}/tests/test_hull_white_convexity.cpp)
target_link_libraries(test_hull_white_convexity PRIVATE quantape)
quantape_enable_stan(test_hull_white_convexity)
quantape_set_warnings(test_hull_white_convexity)
quantape_register_test(test_hull_white_convexity)

# Double-only curve replication: shares the data/builder/dump fixture header
# with the AD test and never instantiates Stan scalars.
add_executable(test_paper_eur_curves ${CMAKE_SOURCE_DIR}/tests/test_paper_eur_curves.cpp)
target_link_libraries(test_paper_eur_curves PRIVATE quantape)
quantape_set_warnings(test_paper_eur_curves)
quantape_register_test(test_paper_eur_curves)

# Curve evaluation is AD-templated; this target instantiates Stan scalars and
# gates solver-derived bootstrap sensitivities. It is the sole writer of the
# portfolio risk table, so no test ordering or parallelism can race on it.
add_executable(test_curve_ad ${CMAKE_SOURCE_DIR}/tests/test_curve_ad.cpp)
target_link_libraries(test_curve_ad PRIVATE quantape)
quantape_enable_stan(test_curve_ad)
quantape_set_warnings(test_curve_ad)
quantape_register_test(test_curve_ad)

quantape_collect_executables()
