// SPDX-License-Identifier: Apache-2.0
// SANKHYA - LP sensitivity ranging: cost and RHS intervals.
//
// Reference: Chvatal, "Linear Programming", ch. 10 (W. H. Freeman, 1983).
//   Cost ranging:  the interval [c_j - delta_lo, c_j + delta_hi] over which the current
//                  basis remains optimal.
//   RHS ranging:   the interval over which a row's active bound can move before the basis
//                  becomes primal infeasible.
//
// Populated only when the "ranging" option is true and the solve status is kOptimal.
// Works in original (unscaled) model space; call after solve_with_scaling() has unscaled.
#pragma once

namespace sankhya {
class Model;
class Options;
class Solution;
}  // namespace sankhya

namespace sankhya::detail {

void compute_ranging(const Model& model, const Options& options, Solution& solution);

}  // namespace sankhya::detail
