// SPDX-License-Identifier: Apache-2.0
// SANKHYA - batched restarted primal-dual hybrid gradient for safe LP-relaxation bounding
// across many branch-and-bound nodes that share one constraint matrix (#520).
//
// References, all written from the papers (ENGINEERING_RULES.md red line: no solver source
// consulted, here or anywhere else in this file):
//   [CP11]  Chambolle & Pock, "A first-order primal-dual algorithm for convex problems with
//           applications to imaging", JMIV 40(1), 2011. Algorithm 1 is the iteration below,
//           the same one src/pdhg/pdhg.cpp runs for a single LP.
//   [PDLP]  Applegate, Diaz, Hinder, Lu, Lubin, O'Donoghue, Schudy, "Practical Large-Scale
//           Linear Programming using Primal-Dual Hybrid Gradient", NeurIPS 2021, section 3.1
//           for the adaptive step size used below.
//   [arXiv:2601.21990] batched first-order strong branching on GPU - the observation this
//           issue (#520) and this file are scoped from: a single node LP is too small to
//           fill a GPU, but K of them sharing one constraint matrix become one SpMM per
//           iteration instead of K matrix-vector products, and strong branching's 2K
//           children are exactly such a batch.
//
// WHAT "SAFE" MEANS HERE. For ANY y (not only an optimal one), Lagrangian weak duality gives
//
//     min_{x in X} c'x  >=  min_{x in X} (c'x + y'Ax) - sup_{v in C} y'v      for  Ax in C
//
// and the right-hand side is finite (rather than -infinity) exactly when, for every column
// j, the reduced cost d_j = c_j + (A'y)_j has a bound on the side it would otherwise want to
// run away to - i.e. y is DUAL FEASIBLE. src/pdhg/pdhg_evaluate.cpp's `evaluate()` already
// computes precisely this quantity (`dual_objective`) and precisely this feasibility
// residual (`absolute_dual`) for the single-LP engine; the code below recomputes the same
// two formulas directly against an explicit (col_lower, col_upper) pair instead of a Model,
// which is what lets K nodes share one scaling and one spectral-norm estimate without a
// Model copy per node. A run that never reaches absolute_dual <= tol::kDualFeasibility
// within its iteration budget reports `safe = false`: an unproven node, not an optimistic
// bound. That is the acceptance criterion #520 asks for (every batched prune re-checked by
// the rational oracle on small instances) made possible: a bound this file calls safe is
// safe or the run reports nothing at all.
//
// THE LOOP BELOW IS THE GPU KERNEL'S CPU REFERENCE, NOT A STAND-IN FOR IT. Every node's
// iterate is independent given the shared matrix, so the per-node loop is exactly the
// operation a CUDA port replaces with one SpMM against an n x K (respectively m x K) dense
// block per iteration: same projections, same step-size rule, same convergence test, done
// K-wide instead of once per node. Nothing about the per-node math would change; only the
// data layout and the two matrix products would move onto the GPU. That is why this loop,
// not a stub, is the correct reference to check a future kernel against - and it is also
// already useful on its own: it is what src/mip/branch_and_bound.cpp calls, behind the
// batched_node_bounding and batched_strong_branching options (both default off), to prune
// or score several nodes without a simplex solve.
//
// OUT OF SCOPE HERE, DELIBERATELY (#520): the CUDA kernel itself (this sandbox has no CUDA
// toolkit, see ENGINEERING_RULES.md) and the rented-GPU throughput numbers the issue's
// second acceptance criterion asks for. Nothing in this file claims either.

#include "pdhg_batch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "sankhya/tolerances.hpp"

#include "../la/scaling.hpp"
#include "pdhg_evaluate.hpp"

namespace sankhya::pdhg {
namespace {

constexpr int kRuizIterations = 10;
constexpr int kPowerIterations = 30;
/// How often the (unscaled) dual-feasibility test runs. Smaller than pdhg.cpp's
/// kEvaluationInterval (40): a batch node's whole budget is
/// tol::kPdhgBatchIterationLimit iterations, and checking only every 40 of those would spend
/// most of a 2,000-iteration run without ever looking.
constexpr Count kBatchEvaluationInterval = 10;

double project(double value, double lower, double upper) {
  if (is_finite_bound(lower) && value < lower) return lower;
  if (is_finite_bound(upper) && value > upper) return upper;
  return value;
}

/// `bound / multiplier`, or `bound` unchanged when it is already infinite (scaling.hpp's
/// scale_bound(), duplicated here because it is anonymous-namespace private to scaling.cpp;
/// one three-line formula, not the scaling algorithm, so this is not the kind of duplication
/// ENGINEERING_RULES.md's provenance rule is about).
double scaled_bound(double bound, double multiplier) {
  return is_finite_bound(bound) ? bound * multiplier : bound;
}

/// The dual objective and its feasibility residual at (unscaled) y, for one node's column
/// box. Same two formulas as pdhg_evaluate.cpp's evaluate() computes for `dual_objective` and
/// `absolute_dual` (see the file comment above for why they coincide), parameterised on an
/// explicit column box instead of a Model so the batch never copies the shared matrix.
struct DualBound {
  double objective = 0.0;
  double absolute_infeasibility = 0.0;
};

DualBound dual_bound_at(const Model& model, const std::vector<double>& cost,
                        const std::vector<double>& col_lower,
                        const std::vector<double>& col_upper, const std::vector<double>& y,
                        std::vector<double>& reduced_scratch) {
  const Index rows = model.num_rows();
  const Index cols = model.num_cols();

  reduced_scratch.assign(static_cast<std::size_t>(cols), 0.0);
  for (Index j = 0; j < cols; ++j)
    reduced_scratch[static_cast<std::size_t>(j)] = cost[static_cast<std::size_t>(j)];
  if (rows > 0) model.matrix.transpose_multiply_add(y.data(), reduced_scratch.data());

  double dual_violation = 0.0;
  double bound_contribution = 0.0;
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double d = reduced_scratch[u];
    if (d > 0.0) {
      if (is_finite_bound(col_lower[u])) {
        bound_contribution += d * col_lower[u];
      } else {
        dual_violation += d * d;
      }
    } else if (d < 0.0) {
      if (is_finite_bound(col_upper[u])) {
        bound_contribution += d * col_upper[u];
      } else {
        dual_violation += d * d;
      }
    }
  }

  double support = 0.0;
  for (Index i = 0; i < rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double yi = y[u];
    if (yi > 0.0) {
      if (is_finite_bound(model.row_upper[u])) {
        support += yi * model.row_upper[u];
      } else {
        dual_violation += yi * yi;
      }
    } else if (yi < 0.0) {
      if (is_finite_bound(model.row_lower[u])) {
        support += yi * model.row_lower[u];
      } else {
        dual_violation += yi * yi;
      }
    }
  }

  DualBound result;
  result.objective = bound_contribution - support;
  result.absolute_infeasibility = std::sqrt(dual_violation);
  return result;
}

/// One node's restarted-PDHG run, sharing `scaling` and `spectral_norm` with every other
/// node in the batch. [CP11] Algorithm 1 with [PDLP]'s adaptive step size, same as
/// src/pdhg/pdhg.cpp's main loop, minus the running-average and restart machinery: those are
/// convergence ACCELERATORS for reporting a tight final answer, and every one of them stays
/// correct to add later, but a safe bound only needs ONE dual-feasible iterate, and the
/// batch's whole point is to stay cheap, not to reproduce pdhg.cpp line for line.
BatchNodeBound solve_one(const Model& model, const std::vector<double>& cost,
                         const Scaling& scaling, double spectral_norm,
                         const BatchNodeBounds& node, Count iteration_limit) {
  const Index rows = model.num_rows();
  const Index cols = model.num_cols();
  const auto n = static_cast<std::size_t>(cols);
  const auto m = static_cast<std::size_t>(rows);

  std::vector<double> col_lower_hat(n);
  std::vector<double> col_upper_hat(n);
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double dc = scaling.column[u];
    col_lower_hat[u] = scaled_bound(node.col_lower[u], 1.0 / dc);
    col_upper_hat[u] = scaled_bound(node.col_upper[u], 1.0 / dc);
  }

  std::vector<double> x(n, 0.0);
  std::vector<double> y(m, 0.0);
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    x[u] = project(0.0, col_lower_hat[u], col_upper_hat[u]);
  }

  std::vector<double> x_next(n, 0.0);
  std::vector<double> y_next(m, 0.0);
  std::vector<double> extrapolated(n, 0.0);
  std::vector<double> at_y(n, 0.0);
  std::vector<double> a_x(m, 0.0);
  std::vector<double> y_unscaled(m, 0.0);
  std::vector<double> reduced_scratch;

  double eta = spectral_norm > 0.0 ? 1.0 / spectral_norm : 1.0;
  const double omega = 1.0;  // no primal-weight adaptation: see the function comment

  BatchNodeBound best;
  Count iteration = 0;
  while (iteration < iteration_limit) {
    const double tau = eta / omega;
    const double sigma = eta * omega;

    for (Index j = 0; j < cols; ++j) at_y[static_cast<std::size_t>(j)] = 0.0;
    if (rows > 0) scaling.matrix.transpose_multiply(y.data(), at_y.data());
    for (Index j = 0; j < cols; ++j) {
      const auto u = static_cast<std::size_t>(j);
      const double gradient = scaling.cost[u] + at_y[u];
      x_next[u] = project(x[u] - tau * gradient, col_lower_hat[u], col_upper_hat[u]);
      extrapolated[u] = 2.0 * x_next[u] - x[u];
    }

    if (rows > 0) scaling.matrix.multiply(extrapolated.data(), a_x.data());
    for (Index i = 0; i < rows; ++i) {
      const auto u = static_cast<std::size_t>(i);
      const double v = y[u] + sigma * a_x[u];
      y_next[u] = v - sigma * project(v / sigma, scaling.row_lower[u], scaling.row_upper[u]);
    }

    double movement = 0.0;
    for (Index j = 0; j < cols; ++j) {
      const double d = x_next[static_cast<std::size_t>(j)] - x[static_cast<std::size_t>(j)];
      movement += 0.5 * omega * d * d;
    }
    for (Index i = 0; i < rows; ++i) {
      const double d = y_next[static_cast<std::size_t>(i)] - y[static_cast<std::size_t>(i)];
      movement += 0.5 * d * d / omega;
    }

    double interaction = 0.0;
    if (rows > 0) {
      std::vector<double> dx(n);
      for (Index j = 0; j < cols; ++j) {
        dx[static_cast<std::size_t>(j)] =
            x_next[static_cast<std::size_t>(j)] - x[static_cast<std::size_t>(j)];
      }
      std::vector<double> adx(m, 0.0);
      scaling.matrix.multiply(dx.data(), adx.data());
      for (Index i = 0; i < rows; ++i) {
        const auto u = static_cast<std::size_t>(i);
        interaction += (y_next[u] - y[u]) * adx[u];
      }
      interaction = std::fabs(interaction);
    }

    // Same reasoning as pdhg.cpp's identically-named branch: zero interaction carries no
    // information about how large eta may safely be, so the honest response is to accept
    // the step and leave eta where it was rather than let either obvious alternative
    // collapse it (see pdhg.cpp for the full argument and the Netlib evidence).
    const bool no_information = interaction <= 0.0;
    const double limit =
        no_information ? std::numeric_limits<double>::infinity() : movement / interaction;
    const double exponent = static_cast<double>(std::max<Count>(2, iteration + 1));
    const double shrink = 1.0 - std::pow(exponent, -0.3);
    const double grow = 1.0 + std::pow(exponent, -0.6);
    const double proposed = std::min(shrink * limit, grow * eta);

    if (eta <= limit) {
      x.swap(x_next);
      y.swap(y_next);
      ++iteration;
    }
    const double eta_ceiling = 1.0e3 / std::max(spectral_norm, 1e-12);
    if (!no_information) eta = std::clamp(proposed, 1e-12, eta_ceiling);

    if (iteration == 0) continue;
    if (iteration % kBatchEvaluationInterval != 0 && !no_information) continue;

    for (Index i = 0; i < rows; ++i) {
      y_unscaled[static_cast<std::size_t>(i)] =
          y[static_cast<std::size_t>(i)] * scaling.row[static_cast<std::size_t>(i)];
    }
    const DualBound candidate =
        dual_bound_at(model, cost, node.col_lower, node.col_upper, y_unscaled, reduced_scratch);
    best.iterations = iteration;
    if (candidate.absolute_infeasibility <= tol::kDualFeasibility &&
        (!best.safe || candidate.objective > best.bound)) {
      best.safe = true;
      best.bound = candidate.objective;
    }
  }
  best.iterations = iteration;
  return best;
}

}  // namespace

std::vector<BatchNodeBound> solve_batch(const Model& model,
                                        const std::vector<BatchNodeBounds>& nodes,
                                        const Options& options, Logger& logger) {
  std::vector<BatchNodeBound> results(nodes.size());
  if (nodes.empty()) return results;

  const Index cols = model.num_cols();
  std::vector<double> cost(static_cast<std::size_t>(cols));
  const double sense = model.sense_multiplier();
  for (Index j = 0; j < cols; ++j) {
    cost[static_cast<std::size_t>(j)] = sense * model.col_cost[static_cast<std::size_t>(j)];
  }

  // EVERYTHING BELOW THIS LINE IS COMPUTED ONCE FOR THE WHOLE BATCH: the scaling passes, the
  // scaled matrix, and the spectral-norm estimate depend only on model.matrix, never on a
  // node's column box (src/la/scaling.cpp builds Dr, Dc from the matrix alone and applies
  // them to the bounds afterwards - see build_scaling()'s comment). This is the shared cost
  // the batch exists to amortise: on the GPU kernel this issue is scoped from, it is the
  // step that turns K separate small solves into one.
  const Scaling scaling = build_scaling(model, cost, kRuizIterations);
  const double spectral_norm =
      estimate_spectral_norm(scaling.matrix, kPowerIterations,
                             static_cast<unsigned>(options.get_int("random_seed")) + 1u);

  const Count iteration_limit_option = options.get_int("pdhg_batch_iterations");
  const Count iteration_limit = iteration_limit_option > 0
                                    ? iteration_limit_option
                                    : Count{tol::kPdhgBatchIterationLimit};

  logger.verbose("batched PDHG bounding: {} nodes, {} rows, {} columns, budget {} iterations",
                 nodes.size(), model.num_rows(), cols, iteration_limit);

  // THE LOOP A GPU KERNEL REPLACES WITH ONE SpMM PER ITERATION ACROSS THE WHOLE BATCH (see
  // the file comment). Each node is independent given the shared scaling above, so this loop
  // has no cross-node data dependency - it is already exactly the shape the GPU port needs,
  // just not yet vectorised across K.
  for (std::size_t k = 0; k < nodes.size(); ++k) {
    results[k] = solve_one(model, cost, scaling, spectral_norm, nodes[k], iteration_limit);
  }
  return results;
}

}  // namespace sankhya::pdhg
