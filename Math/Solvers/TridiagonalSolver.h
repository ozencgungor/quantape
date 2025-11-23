#ifndef TRIDIAGONAL_SOLVER_H
#define TRIDIAGONAL_SOLVER_H

#include <stdexcept>
#include <vector>

namespace Math {

template <typename DoubleT>
class TridiagonalSolver {
public:
    static std::vector<DoubleT> solve(const std::vector<DoubleT>& a, const std::vector<DoubleT>& b,
                                      const std::vector<DoubleT>& c,
                                      const std::vector<DoubleT>& d) {
        int n = d.size();
        if (n == 0) {
            return std::vector<DoubleT>();
        }
        if (a.size() != n || b.size() != n || c.size() != n) {
            throw std::runtime_error("Invalid input sizes for tridiagonal solver");
        }

        std::vector<DoubleT> c_prime(n);
        std::vector<DoubleT> d_prime(n);
        std::vector<DoubleT> x(n);

        c_prime[0] = c[0] / b[0];
        d_prime[0] = d[0] / b[0];

        for (int i = 1; i < n; i++) {
            DoubleT m = 1.0 / (b[i] - a[i] * c_prime[i - 1]);
            c_prime[i] = c[i] * m;
            d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) * m;
        }

        x[n - 1] = d_prime[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            x[i] = d_prime[i] - c_prime[i] * x[i + 1];
        }

        return x;
    }
};

} // namespace Math

#endif // TRIDIAGONAL_SOLVER_H
