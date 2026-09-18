// SPDX-License-Identifier: Apache-2.0
// SANKHYA - PDHG convergence evaluation implementation.

#include "pdhg_evaluate.hpp"

#include <cmath>
#include <limits>

#include "sankhya/types.hpp"

namespace sankhya::pdhg {

double euclidean_norm(const std::vector<double>& v) {
  double s = 0.0;
  for (const double x : v) s += x * x;
  return std::sqrt(s);
}

Residuals evaluate(const Problem& problem, const std::vector<double>& x,
                   const std::vector<double>& y, std::vector<double>& activity,
                   std::vector<double>& reduced) {
  const Model& model = *problem.model;
  const Index rows = model.num_rows();
  const Index cols = model.num_cols();

  Residuals r;

  // ---- Primal: how far Ax falls outside the row bounds. x is projected every iteration,
  // so the column bounds hold by construction and contribute nothing.
  activity.assign(static_cast<std::size_t>(rows), 0.0);
  if (rows > 0) model.matrix.multiply(x.data(), activity.data());
  double primal_violation = 0.0;
  for (Index i = 0; i < rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double a = activity[u];
    double violation = 0.0;
    if (is_finite_bound(model.row_lower[u])) {
      violation = std::max(violation, model.row_lower[u] - a);
    }
    if (is_finite_bound(model.row_upper[u])) {
      violation = std::max(violation, a - model.row_upper[u]);
    }
    primal_violation += violation * violation;
  }
  r.absolute_primal = std::sqrt(primal_violation);
  r.primal = r.absolute_primal / (1.0 + problem.bound_norm);

  // ---- Dual: d = c + A'y. A component of d is only a violation where no bound can absorb
  // it, i.e. a positive reduced cost on a variable with no lower bound, or a negative one
  // on a variable with no upper bound.
  reduced.assign(static_cast<std::size_t>(cols), 0.0);
  for (Index j = 0; j < cols; ++j) {
    reduced[static_cast<std::size_t>(j)] = problem.cost[static_cast<std::size_t>(j)];
  }
  if (rows > 0) model.matrix.transpose_multiply_add(y.data(), reduced.data());

  double dual_violation = 0.0;
  double bound_contribution = 0.0;
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double d = reduced[u];
    if (d > 0.0) {
      if (is_finite_bound(model.col_lower[u])) {
        bound_contribution += d * model.col_lower[u];
      } else {
        dual_violation += d * d;
      }
    } else if (d < 0.0) {
      if (is_finite_bound(model.col_upper[u])) {
        bound_contribution += d * model.col_upper[u];
      } else {
        dual_violation += d * d;
      }
    }
  }
  // The dual residual is assigned once, below, after the row multipliers have had
  // their chance to contribute a violation too.

  // ---- Objectives. The dual objective is the Lagrangian bound:
  //   sum_j (d_j > 0 ? d_j l_j : d_j u_j)  -  sigma_C(y)
  double support = 0.0;
  for (Index i = 0; i < rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double yi = y[u];
    if (yi > 0.0) {
      if (is_finite_bound(model.row_upper[u])) {
        support += yi * model.row_upper[u];
      } else {
        dual_violation += yi * yi;  // no bound to price against: the dual is infeasible
      }
    } else if (yi < 0.0) {
      if (is_finite_bound(model.row_lower[u])) {
        support += yi * model.row_lower[u];
      } else {
        dual_violation += yi * yi;
      }
    }
  }
  r.absolute_dual = std::sqrt(dual_violation);
  r.dual = r.absolute_dual / (1.0 + problem.cost_norm);

  // Complementary slackness, in the product form the verifier uses.
  for (Index i = 0; i < rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (model.row_lower[u] == model.row_upper[u]) continue;  // equality: always tight
    const double lower_slack = is_finite_bound(model.row_lower[u])
                                   ? activity[u] - model.row_lower[u]
                                   : std::numeric_limits<double>::infinity();
    const double upper_slack = is_finite_bound(model.row_upper[u])
                                   ? model.row_upper[u] - activity[u]
                                   : std::numeric_limits<double>::infinity();
    r.complementarity =
        std::max(r.complementarity, std::fabs(y[u]) * std::min(lower_slack, upper_slack));
  }
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (model.col_lower[u] == model.col_upper[u]) continue;  // fixed column
    const double lower_slack = is_finite_bound(model.col_lower[u])
                                   ? x[u] - model.col_lower[u]
                                   : std::numeric_limits<double>::infinity();
    const double upper_slack = is_finite_bound(model.col_upper[u])
                                   ? model.col_upper[u] - x[u]
                                   : std::numeric_limits<double>::infinity();
    r.complementarity =
        std::max(r.complementarity, std::fabs(reduced[u]) * std::min(lower_slack, upper_slack));
  }

  double primal_objective = 0.0;
  for (Index j = 0; j < cols; ++j) {
    primal_objective +=
        problem.cost[static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
  }
  r.primal_objective = primal_objective;
  r.dual_objective = bound_contribution - support;
  const double absolute_gap = std::fabs(r.primal_objective - r.dual_objective);
  r.gap = absolute_gap / (1.0 + std::fabs(r.primal_objective) + std::fabs(r.dual_objective));
  r.gap_as_verified = absolute_gap / std::max(1.0, std::fabs(r.primal_objective));
  return r;
}

}  // namespace sankhya::pdhg
