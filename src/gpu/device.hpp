// SPDX-License-Identifier: Apache-2.0
// SANKHYA - GPU device probe.
//
// Compiled only when SANKHYA_ENABLE_CUDA is set. Included behind that guard from any
// translation unit that needs to know whether a device is present before touching it.
#pragma once

#include <string>

namespace sankhya::gpu {

/// Return true when at least one CUDA-capable device is present and ready.
///
/// On success *description is a human-readable line:
///   "GeForce RTX 4050 Laptop GPU (compute 8.9, 6144 MiB VRAM)"
/// On failure *description holds the reason (no device, driver error, …).
/// description may be null.
[[nodiscard]] bool device_available(std::string* description);

}  // namespace sankhya::gpu
