// SPDX-License-Identifier: Apache-2.0
// SANKHYA - whole-matrix activity-based domain propagation (#510).
//
// Acceptance criterion 1 of #510 asks for "identical fixpoint bounds to the CPU propagator on
// the MIPLIB set, held by a test". The MIPLIB-set version of that needs the full external
// corpus and harness, which is out of this environment's scope; what is checked here is the
// same claim made real on small, hand-built models:
//
//   - AgreesWithPresolveOnSingletonRows runs a diagonal, one-column-per-row model through the
//     real presolve() pipeline and checks that propagate_bounds_to_fixpoint() lands on
//     exactly what presolve.cpp's own kSingletonRow reduction recorded for each row: the
//     record's `coefficient`, `row_lower` and `row_upper` are the untouched, original-model
//     facts presolve computed activity_bounds() from, and the division that turns them into
//     an implied bound is the one-line formula presolve.cpp documents at its own kSingletonRow
//     site ("The division flips the sense when a is negative") - reproduced here as a
//     one-line check on presolve's own recorded numbers, not as a second propagation engine.
//     A singleton row's OWN column disappears from presolve's REDUCED model entirely once the
//     row is consumed (col_count reaches zero and it is folded away as an empty column, #43),
//     so the record - not the reduced model - is the only place presolve keeps what it
//     computed for that column; the columns are declared INTEGER so presolve's UNRELATED
//     free-column-singleton elimination (`model.col_type[u] != VarType::kInteger` in
//     presolve.cpp) never substitutes a different reduction for the one under test.
//   - TwoRowChainConvergesOverMultipleRounds is the case presolve() cannot touch at all
//     (neither row is a singleton, so presolve's reduction set never applies): two
//     non-singleton rows sharing two columns, where the second row's implication only
//     becomes visible after the first row's tightening from the round before, forcing more
//     than one whole-matrix pass. The expected fixpoint is derived by hand in the comment
//     there from the same Brearley/Savelsbergh activity formula presolve.cpp and this module
//     both implement, and is exactly the capability #510 is asking for: propagation that
//     presolve's row-by-row elimination does not do.

#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "presolve/domain_propagation.hpp"
#include "presolve/presolve.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

Model build(const std::vector<std::vector<double>>& rows, const std::vector<double>& row_lower,
            const std::vector<double>& row_upper, const std::vector<double>& col_lower,
            const std::vector<double>& col_upper, VarType type) {
  Model model;
  const auto n = static_cast<Index>(col_lower.size());
  const auto m = static_cast<Index>(rows.size());
  model.col_cost.assign(static_cast<std::size_t>(n), 0.0);
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.col_type.assign(static_cast<std::size_t>(n), type);
  model.row_lower = row_lower;
  model.row_upper = row_upper;
  model.matrix.reset(m, n);
  for (Index i = 0; i < m; ++i) {
    for (Index j = 0; j < n; ++j) {
      const double v = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      if (v != 0.0) model.matrix.add_entry(i, j, v);
    }
  }
  model.matrix.finalize();
  model.hessian.reset(n, n);
  model.hessian.finalize();
  EXPECT_EQ(model.validate(), "");
  return model;
}

presolve::Result run_presolve(const Model& model) {
  Logger logger(nullptr);
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", true);
  return presolve::presolve(model, options, logger);
}

/// presolve.cpp's own kSingletonRow division, applied to what a real Record just recorded:
/// "a * x within [lo, up] is a bound on x... The division flips the sense when a is negative"
/// (presolve.cpp, kSingletonRow). Integer rounding matches presolve.cpp's
/// round_integer_lower/round_integer_upper, cited there to Achterberg et al. (2020). This is
/// the formula, not a second implementation of the propagation loop: it exists only because a
/// Record stores the ingredients (coefficient, row bounds) rather than the derived bound
/// itself, and it is exercised here purely to turn presolve's OWN recorded facts into a number
/// this test can compare propagate_bounds_to_fixpoint() against.
struct ImpliedBound {
  double lower = -kInf;
  double upper = kInf;
};
/// `own_lower` / `own_upper` are the column's bounds in the ORIGINAL model: exactly like
/// presolve.cpp's own kSingletonRow site ("if (finite(implied_lower) && implied_lower >
/// work.col_lower[c]) work.col_lower[c] = implied_lower"), a one-sided row only tightens the
/// side it has an opinion on and leaves the model's own bound standing on the other.
ImpliedBound implied_from_record(const presolve::Record& record, bool integer, double own_lower,
                                 double own_upper) {
  ImpliedBound out{own_lower, own_upper};
  double implied_lower = -kInf;
  double implied_upper = kInf;
  if (record.coefficient > 0.0) {
    if (is_finite_bound(record.row_lower))
      implied_lower = record.row_lower / record.coefficient;
    if (is_finite_bound(record.row_upper))
      implied_upper = record.row_upper / record.coefficient;
  } else {
    if (is_finite_bound(record.row_upper))
      implied_lower = record.row_upper / record.coefficient;
    if (is_finite_bound(record.row_lower))
      implied_upper = record.row_lower / record.coefficient;
  }
  if (integer) {
    if (is_finite_bound(implied_lower))
      implied_lower = std::ceil(implied_lower - tol::kIntegrality);
    if (is_finite_bound(implied_upper))
      implied_upper = std::floor(implied_upper + tol::kIntegrality);
  }
  if (is_finite_bound(implied_lower) && implied_lower > out.lower) out.lower = implied_lower;
  if (is_finite_bound(implied_upper) && implied_upper < out.upper) out.upper = implied_upper;
  return out;
}

TEST(DomainPropagation, DefaultsOff) {
  Options options;
  EXPECT_FALSE(options.get_bool("whole_matrix_propagation"));
}

TEST(DomainPropagation, AgreesWithPresolveOnSingletonRows) {
  // Three unrelated diagonal rows, one live column each, so every row is a genuine
  // presolve.cpp kSingletonRow: the column is declared integer so its free-column-singleton
  // elimination pass (a DIFFERENT reduction, which would remove the column instead of just
  // tightening it) never fires, and the row-sweep singleton reduction is what tightens and
  // drops each row - after which the now-empty column is folded away by a LATER, unrelated
  // presolve pass (#43), taking the reduced model's own copy of the bound with it. The
  // Record kSingletonRow leaves behind is what survives, and is what this test reads.
  //   row0:  2 x0 in [5, 8]     -> raw [2.5, 4],  integer-rounded [3, 4]
  //   row1: -1 x1 in [-7, 2]    -> raw [-2, 7],   integer-rounded [-2, 7]
  //   row2:  3 x2 in [-inf, 9]  -> raw (-5, 3],   integer-rounded upper only: [-5, 3]
  const Model model = build({{2, 0, 0}, {0, -1, 0}, {0, 0, 3}}, {5.0, -7.0, -kInf},
                            {8.0, 2.0, 9.0}, {0, -10, -5}, {20, 10, 5}, VarType::kInteger);

  const presolve::Result reduced = run_presolve(model);
  ASSERT_EQ(reduced.report.singleton_rows, 3);
  ASSERT_EQ(reduced.report.free_column_singletons, 0)
      << "the integer guard should have kept every column from being eliminated";

  std::vector<const presolve::Record*> by_column(static_cast<std::size_t>(model.num_cols()),
                                                 nullptr);
  for (const presolve::Record& record : reduced.records) {
    if (record.kind != presolve::Record::Kind::kSingletonRow) continue;
    by_column[static_cast<std::size_t>(record.column)] = &record;
  }

  const presolve::PropagationResult propagated = presolve::propagate_bounds_to_fixpoint(
      model, presolve::Bounds{model.col_lower, model.col_upper});
  EXPECT_FALSE(propagated.infeasible);

  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    ASSERT_NE(by_column[u], nullptr) << "column " << j << " should have its own singleton row";
    const ImpliedBound expected = implied_from_record(*by_column[u], /*integer=*/true,
                                                      model.col_lower[u], model.col_upper[u]);
    EXPECT_NEAR(propagated.bounds.lower[u], expected.lower, 1e-9)
        << "column " << j << " lower bound";
    EXPECT_NEAR(propagated.bounds.upper[u], expected.upper, 1e-9)
        << "column " << j << " upper bound";
  }
  EXPECT_NEAR(propagated.bounds.lower[0], 3.0, 1e-9);
  EXPECT_NEAR(propagated.bounds.upper[0], 4.0, 1e-9);
  EXPECT_NEAR(propagated.bounds.lower[1], -2.0, 1e-9);
  EXPECT_NEAR(propagated.bounds.upper[1], 7.0, 1e-9);
  EXPECT_NEAR(propagated.bounds.lower[2], -5.0, 1e-9);
  EXPECT_NEAR(propagated.bounds.upper[2], 3.0, 1e-9);
}

TEST(DomainPropagation, TwoRowChainConvergesOverMultipleRounds) {
  // Neither row here is a singleton (both have two live columns), so this is exactly the
  // capability presolve()'s row-by-row elimination does not have: report.singleton_rows and
  // report.bounds_tightened both stay 0, and neither column's bound moves, no matter how
  // presolve's options are set, because there is no singleton row for it to eliminate.
  //
  //   row0: x0 + x1 <= 10        (row_lower = -inf, row_upper = 10)
  //   row1: x0 - x1 >= 2         (row_lower = 2,    row_upper = +inf)
  //   box:  x0, x1 in [0, 20]
  //
  // Hand-derived fixpoint, same activity formula as presolve.cpp / this module:
  //   round 1: row0 alone caps x0, x1 at <= 10 each; row1 alone then raises x0's lower bound
  //            to 2 (from x0 - x1 >= 2 with x1's worst case still 20 at the START of the
  //            round). x0 in [2, 10], x1 in [0, 10].
  //   round 2: row0, read from round 1's box, now caps x1 at 10 - x0's new lower (2) = 8,
  //            tighter than round 1's 10 - a bound round 1's own row0 pass could not see,
  //            because it only had round 1's STARTING x0 lower (0) to subtract, not round 1's
  //            OUTPUT. x0 in [2, 10] (unchanged), x1 in [0, 8].
  //   round 3: no further change - the fixpoint.
  const Model model = build({{1, 1}, {1, -1}}, {-kInf, 2.0}, {10.0, kInf}, {0, 0}, {20, 20},
                            VarType::kContinuous);

  const presolve::Result reduced = run_presolve(model);
  EXPECT_EQ(reduced.report.singleton_rows, 0);
  EXPECT_EQ(reduced.report.bounds_tightened, 0);
  EXPECT_NEAR(reduced.model.col_lower[0], 0.0, 1e-9) << "presolve alone cannot tighten this";
  EXPECT_NEAR(reduced.model.col_upper[1], 20.0, 1e-9) << "presolve alone cannot tighten this";

  const presolve::PropagationResult propagated = presolve::propagate_bounds_to_fixpoint(
      model, presolve::Bounds{model.col_lower, model.col_upper});
  EXPECT_FALSE(propagated.infeasible);
  EXPECT_GT(propagated.rounds, 1) << "the second row's effect only appears a pass later";
  EXPECT_NEAR(propagated.bounds.lower[0], 2.0, 1e-9);
  EXPECT_NEAR(propagated.bounds.upper[0], 10.0, 1e-9);
  EXPECT_NEAR(propagated.bounds.lower[1], 0.0, 1e-9);
  EXPECT_NEAR(propagated.bounds.upper[1], 8.0, 1e-9);
}

TEST(DomainPropagation, ActivityInfeasibilityIsDetected) {
  // x0 >= 100 can never be reached from a box capped at 5: the row's own activity range
  // proves infeasibility before any column bound is even examined for tightening.
  const Model model = build({{1}}, {100.0}, {kInf}, {0}, {5}, VarType::kContinuous);
  const presolve::PropagationResult propagated = presolve::propagate_bounds_to_fixpoint(
      model, presolve::Bounds{model.col_lower, model.col_upper});
  EXPECT_TRUE(propagated.infeasible);
}

TEST(DomainPropagation, AnAlreadyTightBoxIsAFixpointInOnePass) {
  const Model model = build({{1, 1}}, {-kInf}, {100.0}, {0, 0}, {5, 5}, VarType::kContinuous);
  const presolve::PropagationResult propagated = presolve::propagate_bounds_to_fixpoint(
      model, presolve::Bounds{model.col_lower, model.col_upper});
  EXPECT_FALSE(propagated.infeasible);
  EXPECT_EQ(propagated.rounds, 1);
  EXPECT_NEAR(propagated.bounds.lower[0], 0.0, 1e-9);
  EXPECT_NEAR(propagated.bounds.upper[0], 5.0, 1e-9);
  EXPECT_NEAR(propagated.bounds.lower[1], 0.0, 1e-9);
  EXPECT_NEAR(propagated.bounds.upper[1], 5.0, 1e-9);
}

TEST(DomainPropagation, RootWiringLeavesTheOptimumUnchanged) {
  // #510's one wired call site: whole_matrix_propagation tightens the ROOT box before the
  // tree opens. It must never change what a solve reports, only how it gets there - the
  // flag is a fixpoint of the feasible region's own defining inequalities, so the true
  // optimum cannot move. A small MILP, on and off, is the end-to-end regression guard for
  // the wiring in branch_and_bound.cpp.
  //   maximise x0 + 2 x1  s.t.  x0 + x1 <= 10, x0 - x1 >= -3, x0, x1 in [0, 8] integer
  const Model model =
      build({{1, 1}, {1, -1}}, {-kInf, -3.0}, {10.0, kInf}, {0, 0}, {8, 8}, VarType::kInteger);
  Model maximised = model;
  maximised.sense = ObjSense::kMaximize;
  maximised.col_cost = {1.0, 2.0};

  Options off;
  off.set_bool("log_to_console", false);
  off.set_bool("whole_matrix_propagation", false);
  Options on = off;
  on.set_bool("whole_matrix_propagation", true);

  const Solution baseline = solve(maximised, off);
  const Solution propagated = solve(maximised, on);
  ASSERT_EQ(baseline.status, SolveStatus::kOptimal) << baseline.message;
  ASSERT_EQ(propagated.status, SolveStatus::kOptimal) << propagated.message;
  EXPECT_NEAR(baseline.objective, propagated.objective, 1e-6);
}

}  // namespace
}  // namespace sankhya
