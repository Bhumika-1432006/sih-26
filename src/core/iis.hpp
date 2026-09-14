// SPDX-License-Identifier: Apache-2.0
// SANKHYA - IIS computation (internal, not part of the public API).
#pragma once

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {

/// Compute the Irreducible Infeasible Subsystem when `solution` is kInfeasible with a
/// Farkas certificate, storing the result in solution->iis_rows / iis_col_lo / iis_col_hi.
///
/// Does nothing when the certificate is absent or the compute_iis option is false.
///
/// Algorithm: Chinneck & Dravnieks (1991) deletion filter - O(k) re-solves on models
/// restricted to the k constraints the Farkas certificate names. Each re-solve uses
/// `--option algorithm=dual-simplex` so it can exploit the dual feasibility of a
/// perturbed start and prove infeasibility quickly.
void compute_iis(const Model& model, Solution* solution, const Options& options,
                 Logger& logger);

}  // namespace sankhya
