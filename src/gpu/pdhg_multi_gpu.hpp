// SPDX-License-Identifier: Apache-2.0
// SANKHYA - multi-GPU restarted PDHG for LP (#295): public declaration.
//
// Compiled only when SANKHYA_ENABLE_CUDA is ON.
#pragma once

#include <vector>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"

namespace sankhya::gpu {

/// Multi-GPU row-partitioned restarted PDHG.
///
/// Distributes the constraint matrix rows across `device_ids` (one row block per device).
/// The primal iterate x is replicated on every device; the dual iterate y is partitioned.
/// Cross-GPU allreduce for A^T*y is host-mediated (P2P when available for d-to-d copies).
///
/// Falls back to solve_pdhg_gpu() on device 0 when:
///   - device_ids has only one entry, OR
///   - any device fails to allocate or initialize.
[[nodiscard]] Solution solve_pdhg_multi_gpu(const Model& model, const Options& options,
                                            const std::vector<int>& device_ids, Logger& logger,
                                            SolveControl* control = nullptr);

}  // namespace sankhya::gpu
