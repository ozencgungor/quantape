# markets/ad: AD curve bootstrap, the paper EUR replication fixture and the
# portfolio risk stack. qta_test_curve_ad links full Stan (curve_ad and
# hull_white instantiate Stan scalars; paper is double-only but shares the
# fixture); the risk/xccy trio follows in the plain qta_test_risk_ad target.

# AD curve bootstrap, the paper EUR replication fixture and the Hull-White
# convexity reference. test_curve_ad is the sole writer of the portfolio risk
# table; the write lands under a per-test temp directory.
quantape_add_gtest(qta_test_curve_ad
    SOURCES test_curve_ad.cpp test_paper_eur_curves.cpp test_hull_white_convexity.cpp
    STAN
    LABELS unit markets ad slow)

quantape_collect_executables()
