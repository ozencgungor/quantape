#ifndef GAUSS_LOBATTO_INTEGRATOR_H
#define GAUSS_LOBATTO_INTEGRATOR_H

#include "Integrator.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace Math {
    /**
     * @brief Adaptive Gauss-Lobatto quadrature integration
     *
     * This integrator uses an adaptive Gauss-Lobatto formula that automatically
     * subdivides the integration interval where needed for accuracy.
     *
     * Based on the algorithm by W. Gander and W. Gautschi:
     * "Adaptive Quadrature - Revisited", BIT 40(1):84-101, 2000
     *
     * Features:
     * - Adaptive subdivision (focuses computational effort where needed)
     * - Convergence estimation for improved error control
     * - Excellent for smooth functions with localized features
     * - More efficient than uniform refinement methods
     *
     * @tparam DoubleT Numeric type (double or stan::math::var)
     */
    template<typename DoubleT>
    class GaussLobattoIntegrator : public Integrator<DoubleT> {
    public:
        using FunctionType = typename Integrator<DoubleT>::FunctionType;

        /**
         * @brief Construct Gauss-Lobatto integrator
         * @param absAccuracy Absolute accuracy target
         * @param maxIterations Maximum number of function evaluations
         * @param relAccuracy Relative accuracy target (optional, 0 = not used)
         * @param useConvergenceEstimate Use convergence estimation for better error control
         */
        GaussLobattoIntegrator(double absAccuracy,
                               size_t maxIterations,
                               double relAccuracy = 0.0,
                               bool useConvergenceEstimate = true)
            : Integrator<DoubleT>(absAccuracy, maxIterations)
              , m_relAccuracy(relAccuracy)
              , m_useConvergenceEstimate(useConvergenceEstimate) {
        }

    protected:
        DoubleT integrate(const FunctionType &f, DoubleT a, DoubleT b) const override {
            this->setNumberOfEvaluations(0);
            const double calcAbsTolerance = calculateAbsTolerance(f, a, b);

            this->increaseNumberOfEvaluations(2);
            return adaptiveGaussLobattoStep(f, a, b, f(a), f(b), calcAbsTolerance);
        }

    private:
        double m_relAccuracy;
        bool m_useConvergenceEstimate;

        // Gauss-Lobatto quadrature points
        static constexpr double ALPHA = 0.8164965809277260; // sqrt(2/3)
        static constexpr double BETA = 0.4472135954999579; // 1/sqrt(5)
        static constexpr double X1 = 0.94288241569547971906;
        static constexpr double X2 = 0.64185334234578130578;
        static constexpr double X3 = 0.23638319966214988028;

        /**
         * @brief Calculate absolute tolerance based on initial integral estimate
         */
        double calculateAbsTolerance(const FunctionType &f, DoubleT a, DoubleT b) const {
            constexpr double EPSILON = std::numeric_limits<double>::epsilon();
            double relTol = std::max(m_relAccuracy, EPSILON);

            DoubleT m = (a + b) / DoubleT(2.0);
            DoubleT h = (b - a) / DoubleT(2.0);

            // 13-point Gauss-Lobatto rule evaluation
            DoubleT y1 = f(a);
            DoubleT y3 = f(m - DoubleT(ALPHA) * h);
            DoubleT y5 = f(m - DoubleT(BETA) * h);
            DoubleT y7 = f(m);
            DoubleT y9 = f(m + DoubleT(BETA) * h);
            DoubleT y11 = f(m + DoubleT(ALPHA) * h);
            DoubleT y13 = f(b);

            DoubleT f1 = f(m - DoubleT(X1) * h);
            DoubleT f2 = f(m + DoubleT(X1) * h);
            DoubleT f3 = f(m - DoubleT(X2) * h);
            DoubleT f4 = f(m + DoubleT(X2) * h);
            DoubleT f5 = f(m - DoubleT(X3) * h);
            DoubleT f6 = f(m + DoubleT(X3) * h);

            DoubleT acc = h * (DoubleT(0.0158271919734801831) * (y1 + y13)
                               + DoubleT(0.0942738402188500455) * (f1 + f2)
                               + DoubleT(0.1550719873365853963) * (y3 + y11)
                               + DoubleT(0.1888215739601824544) * (f3 + f4)
                               + DoubleT(0.1997734052268585268) * (y5 + y9)
                               + DoubleT(0.2249264653333395270) * (f5 + f6)
                               + DoubleT(0.2426110719014077338) * y7);

            this->increaseNumberOfEvaluations(13);

            double acc_val = value(acc);
            if (acc_val == 0.0 && (value(f1) != 0.0 || value(f2) != 0.0 ||
                                   value(f3) != 0.0 || value(f4) != 0.0 ||
                                   value(f5) != 0.0 || value(f6) != 0.0)) {
                throw std::runtime_error("Cannot calculate absolute accuracy from relative accuracy");
            }

            double r = 1.0;
            if (m_useConvergenceEstimate) {
                // Estimate convergence rate
                DoubleT integral2 = (h / DoubleT(6.0)) * (y1 + y13 + DoubleT(5.0) * (y5 + y9));
                DoubleT integral1 = (h / DoubleT(1470.0)) * (DoubleT(77.0) * (y1 + y13)
                                                             + DoubleT(432.0) * (y3 + y11)
                                                             + DoubleT(625.0) * (y5 + y9)
                                                             + DoubleT(672.0) * y7);

                double diff = std::fabs(value(integral2) - acc_val);
                if (diff != 0.0) {
                    r = std::fabs(value(integral1) - acc_val) / diff;
                }
                if (r == 0.0 || r > 1.0) {
                    r = 1.0;
                }
            }

            if (m_relAccuracy != 0.0) {
                return std::min(this->absoluteAccuracy(), acc_val * relTol) / (r * EPSILON);
            } else {
                return this->absoluteAccuracy() / (r * EPSILON);
            }
        }

        /**
         * @brief Adaptive Gauss-Lobatto step with recursive subdivision
         */
        DoubleT adaptiveGaussLobattoStep(const FunctionType &f,
                                         DoubleT a, DoubleT b,
                                         DoubleT fa, DoubleT fb,
                                         double acc) const {
            if (this->numberOfEvaluations() >= this->maxEvaluations()) {
                throw std::runtime_error("GaussLobattoIntegrator: max number of iterations reached");
            }

            DoubleT h = (b - a) / DoubleT(2.0);
            DoubleT m = (a + b) / DoubleT(2.0);

            DoubleT mll = m - DoubleT(ALPHA) * h;
            DoubleT ml = m - DoubleT(BETA) * h;
            DoubleT mr = m + DoubleT(BETA) * h;
            DoubleT mrr = m + DoubleT(ALPHA) * h;

            DoubleT fmll = f(mll);
            DoubleT fml = f(ml);
            DoubleT fm = f(m);
            DoubleT fmr = f(mr);
            DoubleT fmrr = f(mrr);
            this->increaseNumberOfEvaluations(5);

            // Two integral estimates with different orders
            DoubleT integral2 = (h / DoubleT(6.0)) * (fa + fb + DoubleT(5.0) * (fml + fmr));
            DoubleT integral1 = (h / DoubleT(1470.0)) * (DoubleT(77.0) * (fa + fb)
                                                         + DoubleT(432.0) * (fmll + fmrr)
                                                         + DoubleT(625.0) * (fml + fmr)
                                                         + DoubleT(672.0) * fm);

            // Avoid 80-bit logic on x86 CPU
            double dist = acc + value(integral1 - integral2);

            if (dist == acc || value(mll) <= value(a) || value(b) <= value(mrr)) {
                if (!(value(m) > value(a) && value(b) > value(m))) {
                    throw std::runtime_error("Interval contains no more machine numbers");
                }
                return integral1;
            } else {
                // Subdivide interval into 6 parts for better accuracy
                return adaptiveGaussLobattoStep(f, a, mll, fa, fmll, acc)
                       + adaptiveGaussLobattoStep(f, mll, ml, fmll, fml, acc)
                       + adaptiveGaussLobattoStep(f, ml, m, fml, fm, acc)
                       + adaptiveGaussLobattoStep(f, m, mr, fm, fmr, acc)
                       + adaptiveGaussLobattoStep(f, mr, mrr, fmr, fmrr, acc)
                       + adaptiveGaussLobattoStep(f, mrr, b, fmrr, fb, acc);
            }
        }

        static double value(const DoubleT &x) {
            if constexpr (std::is_same_v<DoubleT, double>) {
                return x;
            } else {
                return x.val();
            }
        }
    };
} // namespace Math

#endif // GAUSS_LOBATTO_INTEGRATOR_H
