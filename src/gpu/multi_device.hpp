// SPDX-License-Identifier: Apache-2.0
// SANKHYA - multi-GPU device enumeration and work partitioning (#295).
//
// Pure C++ API — no CUDA runtime types in the header. The CPU-safe subset
// (partition_rows, parse_device_ids) is compiled into sankhya_objects unconditionally;
// the CUDA subset (device_count, can_peer_access) lives in multi_device.cu and is only
// present in SANKHYA_ENABLE_CUDA builds.
#pragma once

#include <string>
#include <vector>

namespace sankhya::gpu {

// A half-open row range owned by one CUDA device.
struct RowPartition {
  int device_id;
  int row_start;  // inclusive
  int row_end;    // exclusive
  [[nodiscard]] int local_m() const { return row_end - row_start; }
};

// Partition m rows across device_ids as evenly as possible.
// Remainder rows go one-per-device to the first (remainder) devices.
[[nodiscard]] std::vector<RowPartition> partition_rows(int m,
                                                       const std::vector<int>& device_ids);

// Parse comma-separated device IDs (e.g. "0,1,2"). Returns {0} for empty or "auto".
// Negative values and non-integer tokens are silently skipped. Duplicates are removed.
[[nodiscard]] std::vector<int> parse_device_ids(const std::string& option);

#ifdef SANKHYA_ENABLE_CUDA
// Returns the number of CUDA-capable devices present. Returns 0 on driver error.
[[nodiscard]] int device_count();

// Returns true when device `from` can read device `to`'s memory via P2P / NVLink.
// `from == to` always returns true.
[[nodiscard]] bool can_peer_access(int from, int to);
#endif

}  // namespace sankhya::gpu
