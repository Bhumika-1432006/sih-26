// SPDX-License-Identifier: Apache-2.0
// SANKHYA - GPU-accelerated restarted PDHG for LP: public declaration.
//
// Compiled only when SANKHYA_ENABLE_CUDA is ON.  src/core/solve.cpp is the sole caller;
// it guards the call site with the same ifdef.
#pragma once

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"

namespace sankhya::gpu {

/// GPU-accelerated restarted PDHG.
///
/// Falls back transparently to pdhg::solve_pdhg() when no CUDA device is found, so the caller
/// never needs to probe the device itself.
[[nodiscard]] Solution solve_pdhg_gpu(const Model& model, const Options& options,
                                      Logger& logger, SolveControl* control = nullptr);

}  // namespace sankhya::gpu
