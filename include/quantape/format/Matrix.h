#pragma once

// Compact text rendering of Eigen matrices/vectors for logs and diagnostics.
//
//   QTA_LOG_INFO("test", "hessian\n{}", format::matrix(H));
//   QTA_LOG_INFO("test", "grad {}", format::matrix(g, {.precision = 6}));
//
// Scalars are rendered through format::Number (zmij). Long or wide results are
// elided with a "(N more)" marker. The wrapper owns a copy of the matrix:
// quill formats arguments on the backend thread, so a reference-holding view
// would dangle before the message is rendered.

#include "quantape/format/Number.h"

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace quantape::format {

struct MatrixOptions {
    int precision = 4;  // significant digits; < 0 = shortest round-trip
    int maxRows = 12;   // 0 = unlimited
    int maxCols = 12;   // 0 = unlimited
};

/// Owning, async-safe snapshot of an Eigen matrix/vector.
struct Matrix {
    std::shared_ptr<const Eigen::MatrixXd> data;
    MatrixOptions options;
};

template <typename Derived>
Matrix matrix(const Eigen::MatrixBase<Derived>& m, MatrixOptions options = {}) {
    using Scalar = typename Derived::Scalar;
    static_assert(std::is_arithmetic_v<Scalar>,
                  "format::matrix supports arithmetic Eigen scalars only");
    return Matrix{std::make_shared<const Eigen::MatrixXd>(m.template cast<double>()), options};
}

namespace detail {

inline std::string formatCell(double value, int precision) {
    return precision < 0 ? toString(value, FloatFormat::Shortest)
                         : toString(value, FloatFormat::General, precision);
}

inline std::string formatMatrix(const Eigen::MatrixXd& m, const MatrixOptions& options) {
    const Eigen::Index rows = m.rows();
    const Eigen::Index cols = m.cols();
    if (rows == 0 || cols == 0) {
        return "[]";
    }
    const Eigen::Index shownRows =
        options.maxRows > 0 ? std::min<Eigen::Index>(rows, options.maxRows) : rows;
    const Eigen::Index shownCols =
        options.maxCols > 0 ? std::min<Eigen::Index>(cols, options.maxCols) : cols;
    const bool elideRows = shownRows < rows;
    const bool elideCols = shownCols < cols;

    // Vectors are rendered on a single line.
    if (rows == 1 || cols == 1) {
        const Eigen::Index length = std::max(rows, cols);
        const Eigen::Index shown = std::max(shownRows, shownCols);
        std::string out = "[";
        for (Eigen::Index i = 0; i < shown; ++i) {
            out += ' ';
            out += formatCell(rows == 1 ? m(0, i) : m(i, 0), options.precision);
        }
        if (shown < length) {
            out += " ... (";
            out += std::to_string(length - shown);
            out += " more)";
        }
        out += " ]";
        return out;
    }

    std::vector<std::string> cells(static_cast<std::size_t>(shownRows * shownCols));
    std::vector<std::size_t> widths(static_cast<std::size_t>(shownCols), 0);
    for (Eigen::Index r = 0; r < shownRows; ++r) {
        for (Eigen::Index c = 0; c < shownCols; ++c) {
            std::string cell = formatCell(m(r, c), options.precision);
            widths[static_cast<std::size_t>(c)] =
                std::max(widths[static_cast<std::size_t>(c)], cell.size());
            cells[static_cast<std::size_t>(r * shownCols + c)] = std::move(cell);
        }
    }

    std::string out = "[";
    for (Eigen::Index r = 0; r < shownRows; ++r) {
        out += (r == 0) ? "[" : " [";
        for (Eigen::Index c = 0; c < shownCols; ++c) {
            const std::string& cell = cells[static_cast<std::size_t>(r * shownCols + c)];
            out += ' ';
            out.append(widths[static_cast<std::size_t>(c)] - cell.size(), ' ');
            out += cell;
        }
        if (elideCols) {
            out += " ...";
        }
        out += (r + 1 < shownRows) ? "]\n" : "]";
    }
    out += "]";
    if (elideRows) {
        out += "\n ... (";
        out += std::to_string(rows - shownRows);
        out += " more rows";
        if (elideCols) {
            out += ", ";
            out += std::to_string(cols - shownCols);
            out += " more cols";
        }
        out += ")";
    }
    return out;
}

}  // namespace detail

}  // namespace quantape::format
