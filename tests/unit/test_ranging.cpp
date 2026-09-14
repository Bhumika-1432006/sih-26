// SPDX-License-Identifier: Apache-2.0
// SANKHYA - sensitivity ranging tests.
//
// Acceptance criteria from issue #220:
//   1. A cost perturbed INSIDE its reported range re-solves to the SAME basis.
//   2. A cost perturbed OUTSIDE its reported range re-solves to a DIFFERENT basis.
//   3. A RHS perturbed inside its reported range re-solves to the same basis.
//   4. A RHS perturbed outside its reported range re-solves to a different basis.
//   5. For nonbasic columns, verify_solution.py re-derives cost ranges from reduced costs;
//      this test mirrors that check directly.
//
// The LP used throughout:
//   min   -x1 - 2*x2
//   s.t.  x1 + x2  <= 4   (row 0)
//         x1       <= 3   (row 1)
//              x2  <= 3   (row 2)
//         x1, x2  >= 0
//
// Optimal: x1=1, x2=3, objective=-7. Basis: x2, slack1 (row1 slack), slack0 (row0 slack).
// Actually let me recalculate:
//   x1+x2=4, x2=3 => x1=1, objective = -1-6 = -7. Row1: x1=1<=3 (slack=2, basic).
//
// Cost ranging for x1 (basic): c1 can move in some interval without x2 leaving basis.
// Cost ranging for x2 (basic): c2 can move in some interval.
// RHS ranging for row 0 (b=4): feasibility interval.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

// Build the reference LP and return the base solution (with ranging populated).
Model make_lp() {
  // min -x1 - 2*x2
  // x1 + x2 <= 4
  // x1      <= 3
  //      x2 <= 3
  // x1, x2 >= 0
  Model m;
  m.resize_columns(2);
  m.resize_rows(3);
  m.col_cost = {-1.0, -2.0};
  m.col_lower = {0.0, 0.0};
  m.col_upper = {kInfinity, kInfinity};
  m.row_lower = {-kInfinity, -kInfinity, -kInfinity};
  m.row_upper = {4.0, 3.0, 3.0};
  // Row 0: x1 + x2 <= 4
  m.matrix.reset(3, 2);
  m.matrix.add_entry(0, 0, 1.0);
  m.matrix.add_entry(0, 1, 1.0);
  // Row 1: x1 <= 3
  m.matrix.add_entry(1, 0, 1.0);
  // Row 2: x2 <= 3
  m.matrix.add_entry(2, 1, 1.0);
  m.matrix.finalize();
  return m;
}

Options ranging_opts() {
  Options opts;
  opts.set_bool("ranging", true);
  return opts;
}

// Compare two basis status vectors for equality.
bool same_basis(const std::vector<BasisStatus>& a, const std::vector<BasisStatus>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i]) return false;
  return true;
}

TEST(Ranging, RangingPopulatedAtOptimality) {
  const Model m = make_lp();
  const Solution sol = solve(m, ranging_opts());
  EXPECT_EQ(sol.status, SolveStatus::kOptimal);
  EXPECT_EQ(static_cast<int>(sol.col_ranging_lower.size()), m.num_cols());
  EXPECT_EQ(static_cast<int>(sol.col_ranging_upper.size()), m.num_cols());
  EXPECT_EQ(static_cast<int>(sol.row_ranging_lower.size()), m.num_rows());
  EXPECT_EQ(static_cast<int>(sol.row_ranging_upper.size()), m.num_rows());
}

TEST(Ranging, NotPopulatedWhenFlagOff) {
  const Model m = make_lp();
  const Solution sol = solve(m, Options{});
  EXPECT_EQ(sol.status, SolveStatus::kOptimal);
  EXPECT_TRUE(sol.col_ranging_lower.empty());
  EXPECT_TRUE(sol.row_ranging_lower.empty());
}

TEST(Ranging, NonbasicCostRangeMatchesReducedCost) {
  // For a nonbasic column at lower bound, col_ranging_lower == reduced_cost (in model sense).
  // This is what verify_solution.py checks without the LU factors.
  const Model m = make_lp();
  const Solution sol = solve(m, ranging_opts());
  ASSERT_EQ(sol.status, SolveStatus::kOptimal);
  for (Index j = 0; j < m.num_cols(); ++j) {
    const auto jj = static_cast<std::size_t>(j);
    const BasisStatus st = sol.col_status[jj];
    if (st == BasisStatus::kAtLower) {
      // reduced cost >= 0; col_ranging_lower should equal it
      const double d = sol.col_dual[jj];  // in model sense, >= 0 for minimization at lower
      EXPECT_NEAR(sol.col_ranging_lower[jj], d, 1e-8)
          << "column " << j << ": ranging_lo != reduced_cost";
      EXPECT_EQ(sol.col_ranging_upper[jj], kInfinity);
    }
  }
}

TEST(Ranging, CostPerturbInsideRange_SameBasis) {
  // Perturb c_j inside the reported range: the re-solved basis must be identical.
  const Model base = make_lp();
  const Options opts = ranging_opts();
  const Solution base_sol = solve(base, opts);
  ASSERT_EQ(base_sol.status, SolveStatus::kOptimal);

  for (Index j = 0; j < base.num_cols(); ++j) {
    const auto jj = static_cast<std::size_t>(j);
    const double lo = base_sol.col_ranging_lower[jj];
    const double hi = base_sol.col_ranging_upper[jj];

    // Choose a small perturbation that is strictly inside (lo, hi).
    const double pert_lo = lo < kInfinity ? lo * 0.5 : 0.0;
    const double pert_hi = hi < kInfinity ? hi * 0.5 : 0.0;

    // Decrease by pert_lo.
    if (pert_lo > 1e-9) {
      Model m2 = base;
      m2.col_cost[jj] += pert_lo;
      const Solution s2 = solve(m2, opts);
      ASSERT_EQ(s2.status, SolveStatus::kOptimal)
          << "column " << j << " decrease inside range: not optimal";
      EXPECT_TRUE(same_basis(s2.col_status, base_sol.col_status) &&
                  same_basis(s2.row_status, base_sol.row_status))
          << "column " << j << " decrease by " << pert_lo << " (inside lo=" << lo
          << "): basis changed";
    }

    // Increase by pert_hi.
    if (pert_hi > 1e-9) {
      Model m2 = base;
      m2.col_cost[jj] -= pert_hi;
      const Solution s2 = solve(m2, opts);
      ASSERT_EQ(s2.status, SolveStatus::kOptimal)
          << "column " << j << " increase inside range: not optimal";
      EXPECT_TRUE(same_basis(s2.col_status, base_sol.col_status) &&
                  same_basis(s2.row_status, base_sol.row_status))
          << "column " << j << " increase by " << pert_hi << " (inside hi=" << hi
          << "): basis changed";
    }
  }
}

TEST(Ranging, CostPerturbOutsideRange_BasisChanges) {
  // Push c_j well outside the reported range: the re-solved basis must differ.
  const Model base = make_lp();
  const Options opts = ranging_opts();
  const Solution base_sol = solve(base, opts);
  ASSERT_EQ(base_sol.status, SolveStatus::kOptimal);

  // Find a column with a finite ranging bound to violate.
  bool found = false;
  for (Index j = 0; j < base.num_cols(); ++j) {
    const auto jj = static_cast<std::size_t>(j);
    const double lo = base_sol.col_ranging_lower[jj];
    const double hi = base_sol.col_ranging_upper[jj];
    const double push = 5.0;  // a large enough push

    if (lo < kInfinity && lo > 1e-9) {
      Model m2 = base;
      m2.col_cost[jj] -= (lo + push);  // decrease c_j past the lo boundary
      const Solution s2 = solve(m2, opts);
      if (s2.status == SolveStatus::kOptimal) {
        EXPECT_FALSE(same_basis(s2.col_status, base_sol.col_status) &&
                     same_basis(s2.row_status, base_sol.row_status))
            << "column " << j << " perturbed outside lo=" << lo << " but basis did not change";
        found = true;
      }
    }
    if (hi < kInfinity && hi > 1e-9) {
      Model m2 = base;
      m2.col_cost[jj] += (hi + push);
      const Solution s2 = solve(m2, opts);
      if (s2.status == SolveStatus::kOptimal) {
        EXPECT_FALSE(same_basis(s2.col_status, base_sol.col_status) &&
                     same_basis(s2.row_status, base_sol.row_status))
            << "column " << j << " perturbed outside hi=" << hi << " but basis did not change";
        found = true;
      }
    }
  }
  EXPECT_TRUE(found) << "No finite bound found to violate; test is vacuous";
}

TEST(Ranging, RhsPerturbInsideRange_SameBasis) {
  const Model base = make_lp();
  const Options opts = ranging_opts();
  const Solution base_sol = solve(base, opts);
  ASSERT_EQ(base_sol.status, SolveStatus::kOptimal);

  for (Index i = 0; i < base.num_rows(); ++i) {
    const auto ii = static_cast<std::size_t>(i);
    const double lo = base_sol.row_ranging_lower[ii];
    const double hi = base_sol.row_ranging_upper[ii];

    const double pert_hi = hi < kInfinity ? hi * 0.5 : 0.0;
    const double pert_lo = lo < kInfinity ? lo * 0.5 : 0.0;

    if (pert_hi > 1e-9) {
      Model m2 = base;
      // Increase upper bound of row i (the active bound for <= rows).
      m2.row_upper[ii] += pert_hi;
      const Solution s2 = solve(m2, opts);
      ASSERT_EQ(s2.status, SolveStatus::kOptimal)
          << "row " << i << " rhs increase inside range: not optimal";
      EXPECT_TRUE(same_basis(s2.col_status, base_sol.col_status) &&
                  same_basis(s2.row_status, base_sol.row_status))
          << "row " << i << " rhs increase by " << pert_hi << " (inside hi=" << hi
          << "): basis changed";
    }

    if (pert_lo > 1e-9) {
      Model m2 = base;
      m2.row_upper[ii] -= pert_lo;
      const Solution s2 = solve(m2, opts);
      if (s2.status == SolveStatus::kOptimal) {
        EXPECT_TRUE(same_basis(s2.col_status, base_sol.col_status) &&
                    same_basis(s2.row_status, base_sol.row_status))
            << "row " << i << " rhs decrease by " << pert_lo << " (inside lo=" << lo
            << "): basis changed";
      }
    }
  }
}

TEST(Ranging, RhsPerturbOutsideRange_BasisChanges) {
  const Model base = make_lp();
  const Options opts = ranging_opts();
  const Solution base_sol = solve(base, opts);
  ASSERT_EQ(base_sol.status, SolveStatus::kOptimal);

  bool found = false;
  for (Index i = 0; i < base.num_rows(); ++i) {
    const auto ii = static_cast<std::size_t>(i);
    const double lo = base_sol.row_ranging_lower[ii];
    const double hi = base_sol.row_ranging_upper[ii];
    const double push = 10.0;

    if (hi < kInfinity && hi > 1e-9) {
      Model m2 = base;
      m2.row_upper[ii] += (hi + push);
      const Solution s2 = solve(m2, opts);
      if (s2.status == SolveStatus::kOptimal) {
        EXPECT_FALSE(same_basis(s2.col_status, base_sol.col_status) &&
                     same_basis(s2.row_status, base_sol.row_status))
            << "row " << i << " rhs perturbed outside hi=" << hi << " but basis did not change";
        found = true;
      }
    }

    if (lo < kInfinity && lo > 1e-9) {
      // Tightening the row bound past lo should make the basis infeasible or change it.
      Model m2 = base;
      if (m2.row_upper[ii] < kInfinity) {
        m2.row_upper[ii] -= (lo + push);
        const Solution s2 = solve(m2, opts);
        // Accept infeasible or a different basis.
        if (s2.status == SolveStatus::kOptimal) {
          EXPECT_FALSE(same_basis(s2.col_status, base_sol.col_status) &&
                       same_basis(s2.row_status, base_sol.row_status))
              << "row " << i << " rhs perturbed outside lo=" << lo
              << " but basis did not change";
        }
        found = true;
      }
    }
  }
  EXPECT_TRUE(found) << "No finite RHS bound found to violate; test is vacuous";
}

TEST(Ranging, AllValuesNonNegative) {
  // Ranging values must be non-negative (they are allowable changes, not signed deltas).
  const Model m = make_lp();
  const Solution sol = solve(m, ranging_opts());
  ASSERT_EQ(sol.status, SolveStatus::kOptimal);
  for (std::size_t j = 0; j < sol.col_ranging_lower.size(); ++j) {
    EXPECT_GE(sol.col_ranging_lower[j], 0.0) << "col_ranging_lower[" << j << "] < 0";
    EXPECT_GE(sol.col_ranging_upper[j], 0.0) << "col_ranging_upper[" << j << "] < 0";
  }
  for (std::size_t i = 0; i < sol.row_ranging_lower.size(); ++i) {
    EXPECT_GE(sol.row_ranging_lower[i], 0.0) << "row_ranging_lower[" << i << "] < 0";
    EXPECT_GE(sol.row_ranging_upper[i], 0.0) << "row_ranging_upper[" << i << "] < 0";
  }
}

}  // namespace
}  // namespace sankhya
