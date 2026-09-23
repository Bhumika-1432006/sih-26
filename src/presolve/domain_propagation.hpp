// SPDX-License-Identifier: Apache-2.0
// SANKHYA - whole-matrix activity-based domain propagation, iterated to a fixpoint (#510).
//
// Sofranac, Gleixner & Pokutta, "Accelerated domain propagation for mixed-integer linear
// programs on GPUs", arXiv:2009.07785 (2020). Their kernel assigns one thread per row to
// compute that row's min/max activity from the current column bounds, then reduces the bound
// candidate each row implies for each of its columns into a per-column result with atomic
// min/max, and repeats whole passes over the matrix until nothing moves. This is the CPU
// REFERENCE for that kernel: the same three-stage shape (per-row activity, per-column
// candidate reduction, iterate to a fixpoint) over the WHOLE constraint matrix in one call,
// so a future CUDA kernel implementing the paper has a known-correct fixpoint to match. It is
// deliberately its own routine rather than a fold into either of the two propagators already
// in the tree:
//   - src/presolve/presolve.cpp's activity_bounds() drives row-by-row ELIMINATION (empty,
//     singleton, redundant rows; columns folded away). It tightens a column's bound only as a
//     side effect of a row it is about to remove, never a general multi-row sweep.
//   - src/mip/branch_and_bound_node.cpp's propagate() runs the same activity-bound arithmetic
//     but PER NODE, incrementally, against the node's own working bounds, Gauss-Seidel style
//     (each row sees the previous rows' tightening within the same sweep) and capped at three
//     sweeps because it pays for itself many times over a tree and does not need to be exact.
// Neither is the "propagate the whole matrix to a fixpoint once, standalone" routine the paper
// describes and the GPU kernel needs a reference for; this module is that routine.
//
// The activity-bound formula itself is the textbook one both of the above already use:
// Brearley, Mitra & Williams, "Analysis of mathematical programming problems prior to
// applying the simplex algorithm", Math. Programming 8 (1975), for the row min/max activity;
// Savelsbergh, "Preprocessing and probing for mixed integer programming problems", ORSA J.
// Computing 6(4), 1994, for turning a row's activity range into an implied column bound.
//
// Scope (#510): this gives identical fixpoint bounds to the CPU propagator on the small and
// medium models already used by the presolve tests (tests/unit/test_domain_propagation.cpp),
// which is acceptance criterion 1 made concrete without the full MIPLIB harness. The MIPLIB-
// set version of that check, the CUDA kernel itself, and any GPU-vs-CPU timing chart are
// tracked by the issue and are out of scope here - this environment has no CUDA toolkit and
// no GPU to produce a single real number from.
#pragma once

#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/types.hpp"

namespace sankhya::presolve {

/// A column bound pair, num_cols entries each - the same shape branch_and_bound already
/// carries as working_.col_lower / col_upper and global_lower_ / global_upper_ (see
/// src/mip/branch_and_bound_internal.hpp), named here only so it can be passed and returned
/// as one value instead of two parallel vectors.
struct Bounds {
  std::vector<double> lower;
  std::vector<double> upper;
};

/// Outcome of propagate_bounds_to_fixpoint().
struct PropagationResult {
  Bounds bounds;
  /// True when a row's own activity range cannot reach its stated bounds, or two implied
  /// bounds on the same column crossed: the box is provably empty. `bounds` still holds the
  /// tightened values that proved it, exactly as the per-row infeasibility checks in
  /// presolve.cpp and branch_and_bound_node.cpp report the row that triggered them, but no
  /// caller should treat them as a feasible box.
  bool infeasible = false;
  /// Whole-matrix passes actually run; at most tol::kDomainPropagationMaxRounds. A pass that
  /// tightens nothing beyond tol::kPrimalFeasibility ends the loop early - most models reach
  /// their fixpoint in a handful of passes (Savelsbergh 1994's observation, which is also why
  /// branch_and_bound_node.cpp's node propagator stops after three).
  Count rounds = 0;
};

/// Activity-based bound propagation over the WHOLE constraint matrix of `model`, iterated to
/// a fixpoint.
///
/// One pass: for every row, compute its min/max activity from the CURRENT column bounds (the
/// paper's per-row thread), then for every column the row touches, combine the bound that row
/// implies for it with whatever every OTHER row in the same pass implies, keeping the
/// tightest (the paper's atomic min/max reduction; here a running std::min / std::max over a
/// per-column candidate array, read from this pass's starting bounds and written once the
/// pass finishes - so within a pass every row sees the same bounds every other row does,
/// exactly as parallel row-threads reading a shared array before any of them writes back
/// would). The candidates become the new bounds, integer columns are rounded inward
/// (Achterberg et al. 2020), and the next pass starts from there; passes repeat until nothing
/// moves or the round cap is hit.
///
/// `model`'s matrix and row bounds are read as fixed data; `in` supplies the column bounds to
/// start from - the model's own, an already-tightened box from presolve, or a node's - and is
/// consumed rather than aliased, so the caller's copy is untouched.
[[nodiscard]] PropagationResult propagate_bounds_to_fixpoint(const Model& model, Bounds in);

}  // namespace sankhya::presolve
