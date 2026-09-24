// SPDX-License-Identifier: Apache-2.0
// Reflected restarted Halpern PDHG for LP (#481).
//
// References:
//   [LY24]  Lu & Yang, "Restarted Halpern PDHG for linear programming",
//           arXiv:2407.16144, 2024.
//   [H67]   Halpern, "Fixed points of nonexpanding maps", Bull. AMS 73, 1967.
//   [CSY24] Chen, Sun, Yuan, Zhang & Zhao, "HPR-LP: an implementation of an HPR
//           method for solving linear programming", arXiv:2408.12179, 2024.
//
// The Halpern iteration:
//   z_{k+1} = (k+1)/(k+2) * [(1+gamma)*T(z_k) - gamma*z_k] + 1/(k+2) * z_0
// where T is one PDHG step and z_0 is the anchor. Restart on the fixed-point
// residual r(z) = ||z - T(z)||_P; set the new anchor to the PDHG output.
//
// CURRENTLY A STUB: pdhg_halpern_step() returns false immediately.
// Default OFF (pdhg_halpern=false). Full implementation tracked in #481.
#pragma once

#include "sankhya/options.hpp"

namespace sankhya::pdhg {

/// Apply one Halpern-form step for the LP primal-dual iterate.
/// Returns true when the option is active; false (no-op) otherwise.
///
/// CURRENTLY A STUB: returns false immediately when pdhg_halpern=false (the
/// default). The caller in pdhg.cpp is unchanged until the option is on.
bool pdhg_halpern_step(const Options& options);

}  // namespace sankhya::pdhg
