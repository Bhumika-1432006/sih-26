// SPDX-License-Identifier: Apache-2.0
// SANKHYA - tests for batched restarted-PDHG safe bounding (#520).
//
// This is acceptance criterion 1 of issue #520 made real: "every pruning by a batched bound
// re-checked by the rational oracle on small instances, held by a test". A small MIP is
// built once; a batch of branch-and-bound-style node bound vectors is generated from it
// (the same shared matrix, one column box per node); pdhg::solve_batch() bounds every node
// in the batch; and each node is ALSO solved exactly, in rational arithmetic, by
// oracles::solve_exact(). Two properties are checked against that exact answer, on every
// node the batch reports a bound for:
//
//   1. THE BOUND IS SAFE: it never exceeds the exact LP relaxation optimum (a lower bound
//      that overshoots the truth is not a lower bound).
//   2. A PRUNE THE BOUND WOULD MAKE IS CORRECT: whenever the safe bound is no better than a
//      test incumbent, the exact optimum is no better than the incumbent either, so pruning
//      the node never discards the true optimum.
//
// Both follow mathematically from weak Lagrangian duality once the bound is dual-feasible
// (see pdhg_batch.cpp's file comment), but the point of this test is to check the CODE, not
// re-derive the theorem: an implementation bug in dual_bound_at() or in the scaling algebra
// would show up here as a bound that beats the oracle's exact answer.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

#include "oracles/rational_simplex.hpp"
#include "pdhg/pdhg_batch.hpp"

namespace sankhya::pdhg {
namespace {

using sankhya::oracle::GeneratedLp;
using sankhya::oracle::kNoUpperBound;
using sankhya::oracle::OracleResult;
using sankhya::oracle::OracleStatus;
using sankhya::oracle::solve_exact;
using sankhya::oracle::to_model;

/// min 2 x0 + 3 x1 + x2
///   s.t.  x0 + x1        >= 4
///                x1 + x2 >= 3
///         x0        + x2 >= 2
///   0 <= x0, x1, x2 <= 10
///
/// Small enough that a node's exact optimum can be sanity-checked by hand, big enough to
/// have a real feasible region PDHG has to find. x0 and x1 are marked integer purely to fit
/// the "small MIP" framing of #520's acceptance criterion - solve_batch() only ever bounds
/// the CONTINUOUS relaxation, so integrality plays no role in what is being checked here.
GeneratedLp base_instance() {
  GeneratedLp lp;
  lp.num_rows = 3;
  lp.num_cols = 3;
  lp.a = {{1, 1, 0}, {0, 1, 1}, {1, 0, 1}};
  lp.b = {4, 3, 2};
  lp.c = {2, 3, 1};
  lp.upper = {10, 10, 10};
  return lp;
}

Model base_model() {
  Model model = to_model(base_instance());
  model.col_type = {VarType::kInteger, VarType::kInteger, VarType::kContinuous};
  return model;
}

Options batch_options() {
  Options options;
  // Generous relative to tol::kPdhgBatchIterationLimit: these instances are tiny, and the
  // test wants most of the generated nodes to actually reach a dual-feasible iterate, not
  // merely to exercise the "gave up early" path.
  options.set_int("pdhg_batch_iterations", 20000);
  options.set_int("random_seed", 20260923);
  return options;
}

/// One batch node, in both forms: the double bound vectors solve_batch() takes, and the
/// int64 lp.lower / lp.upper the exact oracle takes on a copy of base_instance(). Building
/// both from the same two int vectors is what makes the comparison below meaningful - a
/// typo that gave the oracle a different box than the batch saw would make this test measure
/// nothing.
struct TestNode {
  std::string name;
  std::vector<std::int64_t> lower;
  std::vector<std::int64_t> upper;
};

BatchNodeBounds to_batch_bounds(const TestNode& node) {
  BatchNodeBounds bounds;
  bounds.col_lower.resize(node.lower.size());
  bounds.col_upper.resize(node.upper.size());
  for (std::size_t j = 0; j < node.lower.size(); ++j) {
    bounds.col_lower[j] = static_cast<double>(node.lower[j]);
    bounds.col_upper[j] =
        node.upper[j] == kNoUpperBound ? kInfinity : static_cast<double>(node.upper[j]);
  }
  return bounds;
}

OracleResult exact_optimum(const TestNode& node) {
  GeneratedLp lp = base_instance();
  lp.lower = node.lower;
  lp.upper = node.upper;
  return solve_exact(lp);
}

/// A batch of node boxes shaped like what branch_and_bound.cpp actually generates: the root,
/// two children of one branch on x0, a grandchild that branches again on x1, and one node
/// whose box is empty (x0 and x1 both capped at 0 against the row that needs x0 + x1 >= 4) -
/// the infeasible case a batched bound must not silently mishandle.
std::vector<TestNode> node_batch() {
  return {
      {"root", {0, 0, 0}, {10, 10, 10}},
      {"x0<=1 (down)", {0, 0, 0}, {1, 10, 10}},
      {"x0>=2 (up)", {2, 0, 0}, {10, 10, 10}},
      {"x0>=2, x1<=1", {2, 0, 0}, {10, 1, 10}},
      {"x0>=2, x1>=2", {2, 2, 0}, {10, 10, 10}},
      {"infeasible: x0<=0, x1<=0", {0, 0, 0}, {0, 0, 10}},
  };
}

TEST(PdhgBatch, SafeBoundsNeverExceedTheExactOracleOptimum) {
  const Model model = base_model();
  const std::vector<TestNode> nodes = node_batch();

  std::vector<BatchNodeBounds> batch;
  batch.reserve(nodes.size());
  for (const TestNode& node : nodes) batch.push_back(to_batch_bounds(node));

  Logger logger(nullptr);
  const std::vector<BatchNodeBound> results =
      solve_batch(model, batch, batch_options(), logger);
  ASSERT_EQ(results.size(), nodes.size());

  int safe_count = 0;
  int optimal_count = 0;
  for (std::size_t k = 0; k < nodes.size(); ++k) {
    const OracleResult exact = exact_optimum(nodes[k]);
    SCOPED_TRACE(nodes[k].name);

    if (exact.status == OracleStatus::kOptimal) {
      ++optimal_count;
      if (results[k].safe) {
        ++safe_count;
        // THE SAFETY PROPERTY: a batched bound reported safe must not be more optimistic
        // (for this MINIMISE problem, strictly greater) than the true optimum, to well
        // inside the dual-feasibility tolerance the bound was accepted at.
        EXPECT_LE(results[k].bound, exact.objective.to_double() + 1e-6)
            << "batched bound " << results[k].bound << " beats the exact optimum "
            << exact.objective.to_double();
      }
    } else {
      EXPECT_EQ(exact.status, OracleStatus::kInfeasible) << "unexpected oracle status for a "
                                                            "hand-built tiny instance";
      // An infeasible node's true optimum is +infinity for this minimise problem, so ANY
      // finite bound is vacuously safe; nothing to compare against beyond "did not crash".
    }
  }

  // Every feasible node in node_batch() is tiny and well inside pdhg_batch_iterations, so a
  // reasonable implementation reaches dual feasibility on most of them. This is not a
  // property being tested for its own sake, only a guard against the whole batch silently
  // reporting `safe = false` everywhere, which would make the EXPECT_LE loop above vacuous.
  EXPECT_GT(safe_count, optimal_count / 2)
      << "too few nodes reached a safe bound for the comparison above to mean anything";
}

TEST(PdhgBatch, EveryBatchedPruneDecisionAgreesWithTheExactOracle) {
  const Model model = base_model();
  const std::vector<TestNode> nodes = node_batch();

  std::vector<BatchNodeBounds> batch;
  batch.reserve(nodes.size());
  for (const TestNode& node : nodes) batch.push_back(to_batch_bounds(node));

  Logger logger(nullptr);
  const std::vector<BatchNodeBound> results =
      solve_batch(model, batch, batch_options(), logger);

  // An incumbent chosen so it sits strictly between some nodes' true optima and others': the
  // point is for the batch to actually prune SOME of these nodes and open others, not for
  // every node to land on the same side.
  constexpr double kIncumbent = 12.0;
  constexpr double kMargin = tol::kMipAbsoluteGap;

  int pruned = 0;
  int opened = 0;
  for (std::size_t k = 0; k < nodes.size(); ++k) {
    SCOPED_TRACE(nodes[k].name);
    if (!results[k].safe) continue;  // an unproven node prunes nothing, by construction

    const bool batch_prunes = results[k].bound >= kIncumbent - kMargin;
    const OracleResult exact = exact_optimum(nodes[k]);

    if (batch_prunes) {
      ++pruned;
      // THE ACCEPTANCE CRITERION: a node the batch would prune must not actually contain a
      // solution better than the incumbent. An infeasible node trivially satisfies this
      // (there is nothing in it to discard); an optimal one must have its exact objective
      // on the pruned side too.
      if (exact.status == OracleStatus::kOptimal) {
        EXPECT_GE(exact.objective.to_double(), kIncumbent - kMargin - 1e-6)
            << "batch pruned a node whose exact optimum " << exact.objective.to_double()
            << " beats the incumbent " << kIncumbent;
      }
    } else {
      ++opened;
    }
  }
  EXPECT_GT(pruned, 0) << "the incumbent was chosen to prune at least one node";
  EXPECT_GT(opened, 0) << "the incumbent was chosen to leave at least one node open";
}

TEST(PdhgBatch, EmptyBatchReturnsNoResults) {
  const Model model = base_model();
  Logger logger(nullptr);
  const std::vector<BatchNodeBound> results = solve_batch(model, {}, batch_options(), logger);
  EXPECT_TRUE(results.empty());
}

}  // namespace
}  // namespace sankhya::pdhg
