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
//
// THE RANGES ARE IN THE MODEL'S OWN SENSE. The arithmetic runs in minimization space,
// where the simplex lives; for a maximize model a decrease of the internal cost is an
// increase of the coefficient the user wrote, so the two sides are exchanged before they
// are reported. A degenerate optimal basis - a basic variable sitting on a bound - makes
// the ranges those of THIS basis rather than of the unique answer, and the solution says so
// (Solution::ranging_basis_degenerate) rather than leaving the reader to guess.
#pragma once

namespace sankhya {
class Logger;
class Model;
class Options;
class Solution;
}  // namespace sankhya

namespace sankhya::detail {

void compute_ranging(const Model& model, const Options& options, Logger& logger,
                     Solution& solution);

}  // namespace sankhya::detail
