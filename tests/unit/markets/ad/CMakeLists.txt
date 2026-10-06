# markets/ad: AD curve bootstrap, the paper EUR replication fixture and the
# portfolio risk stack. qta_test_curve_ad links full Stan (curve_ad and
# hull_white instantiate Stan scalars; paper is double-only but shares the
# fixture); qta_test_risk_ad is the double-only risk/xccy trio.

# AD curve bootstrap, the paper EUR replication fixture and the Hull-White
# convexity reference. test_curve_ad is the sole writer of the portfolio risk
# table; the write lands under a per-test temp directory.
quantape_add_gtest(qta_test_curve_ad
    SOURCES test_curve_ad.cpp test_paper_eur_curves.cpp test_hull_white_convexity.cpp
    STAN
    LABELS unit markets ad slow)

# Double-only curve risk: quote/stack risk, the risk report and the
# cross-currency/FX bootstraps; no Stan scalars are instantiated, so the
# target stays on the plain link profile and drops the `ad` label.
quantape_add_gtest(qta_test_risk_ad
    SOURCES test_curve_risk.cpp test_curve_risk_report.cpp test_xccy.cpp
    LABELS unit markets risk slow)

quantape_collect_executables()
