// SPDX-License-Identifier: Apache-2.0
// SANKHYA - batched restarted-PDHG safe bounding for branch-and-bound nodes (#520).
//
// See pdhg_batch.cpp for the algorithm and its citations. This header is the seam a future
// CUDA kernel plugs into: solve_batch() takes the same shared matrix / per-node column box
// split a GPU batched kernel would, so wiring in a device implementation later is a matter
// of replacing this function's body, not its callers.
#pragma once

#include <vector>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/types.hpp"

namespace sankhya::pdhg {

/// One node's column box. Everything else - the constraint matrix, the objective, the row
/// bounds - is shared across the whole batch via the `model` argument to solve_batch(): a
/// branch-and-bound node differs from its siblings in exactly this and nothing else.
struct BatchNodeBounds {
  std::vector<double> col_lower;  ///< size model.num_cols(), model's own units
  std::vector<double> col_upper;
};

/// What one node's batched run reports.
struct BatchNodeBound {
  /// A valid lower bound on the node's LP relaxation optimum, in MINIMISE space and
  /// EXCLUDING model.objective_offset - the same convention TreeNode::bound and
  /// BranchAndBound::incumbent_internal_ use (branch_and_bound_internal.hpp), chosen so a
  /// caller can compare this directly against those without a sense or offset conversion.
  /// Meaningful only when `safe` is true.
  double bound = -kInfinity;
  /// Whether `bound` actually IS a valid lower bound. False means the run never reached a
  /// dual-feasible iterate inside its budget, which is a normal outcome for an early-stopped
  /// method and not an error: the caller falls back to solving the node normally. Silence
  /// on an unsure node, never an optimistic one, is the whole safety property this type
  /// exists to carry.
  bool safe = false;
  Count iterations = 0;
};

/// Batched restarted-PDHG safe bounding (#520). One shared constraint matrix and row box
/// (`model.matrix`, `model.row_lower`, `model.row_upper`, `model.col_cost`), K column boxes
/// (`nodes`), one early-stopped restarted-PDHG run per node. See pdhg_batch.cpp for why an
/// early-stopped iterate's dual objective is a valid bound and for the literature this is
/// written from.
///
/// nodes[k].col_lower / col_upper must each have model.num_cols() entries. Returns one
/// BatchNodeBound per node, same order as `nodes`.
[[nodiscard]] std::vector<BatchNodeBound> solve_batch(const Model& model,
                                                      const std::vector<BatchNodeBounds>& nodes,
                                                      const Options& options, Logger& logger);

}  // namespace sankhya::pdhg
