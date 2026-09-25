#ifndef GENERIC_MC_STAN_MATH_H
#define GENERIC_MC_STAN_MATH_H

/**
 * @file StanMath.h
 * @brief Single include for Stan AD — plugin-safe Eigen ordering
 *
 * Stan injects MatrixBase::val()/adj() into Eigen through
 * EIGEN_MATRIXBASE_PLUGIN (prim/fun/Eigen.hpp), and the plugin only takes
 * effect if Eigen is first parsed AFTER the define. Including <Eigen/Dense>
 * before <stan/math.hpp> sets Eigen's include guards first, silently drops
 * the plugin, and every matrix-var path fails to compile.
 *
 * This header fixes the order (stan -> mix -> Eigen) and is guarded with
 * clang-format off so `format_code.sh` can never re-sort it. Project code
 * includes THIS header instead of raw <stan/math.hpp> + <Eigen/Dense>:
 * the ordering problem disappears entirely. <Eigen/Dense> is provided
 * transitively (documented dependency, like stan/math.hpp provides its own
 * Eigen wrapper).
 *
 * .clang-format's IncludeCategories sort project headers ("Math/...") before
 * third-party ones (<Eigen/...>), so this include also stays first within
 * any include block it appears in.
 */

// clang-format off
#include <stan/math.hpp>
#include <stan/math/mix.hpp>
#include <Eigen/Dense>
// clang-format on

#endif // GENERIC_MC_STAN_MATH_H
