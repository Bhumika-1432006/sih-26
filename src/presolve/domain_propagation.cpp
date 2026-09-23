// SPDX-License-Identifier: Apache-2.0
// SANKHYA - whole-matrix activity-based domain propagation. See domain_propagation.hpp for
// the citation, the rationale and how this differs from the row-by-row propagators already
// in the tree.

#include "domain_propagation.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "sankhya/sparse.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::presolve {
namespace {

/// Round a derived bound INWARD for an integer column, exactly as presolve.cpp's
/// round_integer_lower/round_integer_upper and branch_and_bound_node.cpp's propagate() do:
/// narrowing a box past a feasible integer silently drops the optimum, so the tolerance only
/// ever moves the bound towards the interior.
[[nodiscard]] double round_integer_lower(double value) {
  return std::ceil(value - tol::kIntegrality);
}
[[nodiscard]] double round_integer_upper(double value) {
  return std::floor(value + tol::kIntegrality);
}

/// Did a bound that can only ever tighten (see the Jacobi seeding below) actually move by
/// more than the feasibility tolerance? `before` is the fixpoint's incoming value; `after` is
/// never looser than `before`, by construction of the caller's std::min / std::max reduction,
/// so the only two cases are "unchanged", "left an infinity", and "moved within the finite
/// range".
[[nodiscard]] bool bound_tightened(double before, double after) {
  if (before == after) return false;
  if (!is_finite_bound(before)) return true;  // was unbounded, is now finite
  return std::fabs(after - before) > tol::kPrimalFeasibility * std::max(1.0, std::fabs(before));
}

}  // namespace

PropagationResult propagate_bounds_to_fixpoint(const Model& model, Bounds in) {
  const Index rows = model.num_rows();
  const Index cols = model.num_cols();
  assert(static_cast<Index>(in.lower.size()) == cols);
  assert(static_cast<Index>(in.upper.size()) == cols);

  PropagationResult result;
  result.bounds = std::move(in);

  // #328's fix, generalised to the whole matrix: an integer column with no row of its own
  // (legal MPS - objective-only) is rounded here, once, rather than only ever inside a row
  // pass it never takes part in.
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (model.col_type[u] != VarType::kInteger) continue;
    if (is_finite_bound(result.bounds.lower[u])) {
      result.bounds.lower[u] = round_integer_lower(result.bounds.lower[u]);
    }
    if (is_finite_bound(result.bounds.upper[u])) {
      result.bounds.upper[u] = round_integer_upper(result.bounds.upper[u]);
    }
  }

  const CsrView by_row(model.matrix);
  Bounds candidate;
  candidate.lower.resize(static_cast<std::size_t>(cols));
  candidate.upper.resize(static_cast<std::size_t>(cols));

  for (int round = 0; round < tol::kDomainPropagationMaxRounds; ++round) {
    ++result.rounds;

    // JACOBI SEEDING: every row-thread of this pass reads `result.bounds` (this pass's
    // starting point, untouched by any row processed earlier in the SAME pass) and only ever
    // tightens `candidate` towards it - never loosens past the incoming value. That is what
    // makes a running std::min / std::max over `candidate` the CPU analogue of the paper's
    // per-column atomic reduction across parallel row-threads: order within a pass cannot
    // matter, because every row is looking at the same snapshot.
    candidate.lower = result.bounds.lower;
    candidate.upper = result.bounds.upper;

    for (Index i = 0; i < rows; ++i) {
      const auto ui = static_cast<std::size_t>(i);
      const ColumnView row = by_row.row(i);

      // ---- Row activity from the current bounds (Brearley, Mitra & Williams 1975). --------
      double min_activity = 0.0;
      double max_activity = 0.0;
      bool min_infinite = false;
      bool max_infinite = false;
      for (Index k = 0; k < row.size; ++k) {
        const auto j = static_cast<std::size_t>(row.rows[k]);
        const double a = row.values[k];
        const double lo = result.bounds.lower[j];
        const double hi = result.bounds.upper[j];
        const double low_term = a > 0.0 ? a * lo : a * hi;
        const double high_term = a > 0.0 ? a * hi : a * lo;
        if (is_infinite(low_term)) {
          min_infinite = true;
        } else {
          min_activity += low_term;
        }
        if (is_infinite(high_term)) {
          max_infinite = true;
        } else {
          max_activity += high_term;
        }
      }

      const double row_lo = model.row_lower[ui];
      const double row_up = model.row_upper[ui];

      // Infeasible by activity alone: no assignment inside the current box can satisfy this
      // row, regardless of what any other row wants. Recorded, not returned immediately -
      // every other row still gets its pass so `result.bounds` ends the round at a real,
      // reproducible fixpoint rather than wherever the row loop happened to be.
      if (!min_infinite && is_finite_bound(row_up) &&
          min_activity > row_up + tol::kPrimalFeasibility) {
        result.infeasible = true;
      }
      if (!max_infinite && is_finite_bound(row_lo) &&
          max_activity < row_lo - tol::kPrimalFeasibility) {
        result.infeasible = true;
      }

      // ---- Per-column candidates this row implies (Savelsbergh 1994). ---------------------
      // For column j with coefficient a: pull its own contribution out of the row's min/max
      // activity (own_low / own_high) to get the range the OTHER columns can still cover, and
      // divide what is left of the row's own bound by a. own_low / own_high reduce to the
      // whole of min_activity / max_activity exactly when the row is a singleton on j, which
      // is why a singleton row's implied bound here is the same division presolve.cpp's
      // kSingletonRow reduction performs.
      for (Index k = 0; k < row.size; ++k) {
        const auto j = static_cast<std::size_t>(row.rows[k]);
        const double a = row.values[k];
        if (std::fabs(a) < tol::kZeroDrop) continue;
        const double lo = result.bounds.lower[j];
        const double hi = result.bounds.upper[j];
        const double own_low = a > 0.0 ? a * lo : a * hi;
        const double own_high = a > 0.0 ? a * hi : a * lo;

        if (!min_infinite && is_finite_bound(row_up) && !is_infinite(own_low)) {
          const double slack = row_up - (min_activity - own_low);
          const double implied = slack / a;
          if (a > 0.0) {
            candidate.upper[j] = std::min(candidate.upper[j], implied);
          } else {
            candidate.lower[j] = std::max(candidate.lower[j], implied);
          }
        }
        if (!max_infinite && is_finite_bound(row_lo) && !is_infinite(own_high)) {
          const double slack = row_lo - (max_activity - own_high);
          const double implied = slack / a;
          if (a > 0.0) {
            candidate.lower[j] = std::max(candidate.lower[j], implied);
          } else {
            candidate.upper[j] = std::min(candidate.upper[j], implied);
          }
        }
      }
    }

    // Integer rounding and the crossed-bound check happen once per pass, after every row's
    // candidate has landed - rounding mid-pass would make later rows in the same pass see a
    // value no row-thread actually produced.
    bool changed = false;
    for (Index j = 0; j < cols; ++j) {
      const auto u = static_cast<std::size_t>(j);
      if (model.col_type[u] == VarType::kInteger) {
        if (is_finite_bound(candidate.lower[u])) {
          candidate.lower[u] = round_integer_lower(candidate.lower[u]);
        }
        if (is_finite_bound(candidate.upper[u])) {
          candidate.upper[u] = round_integer_upper(candidate.upper[u]);
        }
      }
      if (candidate.lower[u] > candidate.upper[u] + tol::kPrimalFeasibility) {
        result.infeasible = true;
      }
      if (bound_tightened(result.bounds.lower[u], candidate.lower[u]) ||
          bound_tightened(result.bounds.upper[u], candidate.upper[u])) {
        changed = true;
      }
    }

    result.bounds.lower.swap(candidate.lower);
    result.bounds.upper.swap(candidate.upper);

    if (result.infeasible || !changed) break;
  }

  return result;
}

}  // namespace sankhya::presolve
