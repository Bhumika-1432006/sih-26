// SPDX-License-Identifier: Apache-2.0
// SANKHYA - multi-GPU partition and option utilities (#295).
//
// CPU-only; no CUDA calls. Compiled unconditionally so that tests can exercise
// partition_rows / parse_device_ids without a CUDA build.

#include "multi_device.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace sankhya::gpu {

std::vector<RowPartition> partition_rows(int m, const std::vector<int>& device_ids) {
  const int k = static_cast<int>(device_ids.size());
  std::vector<RowPartition> parts;
  if (k <= 0) return parts;
  parts.reserve(static_cast<std::size_t>(k));
  const int base = (m >= 0 ? m : 0) / k;
  const int rem = (m >= 0 ? m : 0) % k;
  int row = 0;
  for (int i = 0; i < k; ++i) {
    const int sz = base + (i < rem ? 1 : 0);
    parts.push_back({device_ids[static_cast<std::size_t>(i)], row, row + sz});
    row += sz;
  }
  return parts;
}

std::vector<int> parse_device_ids(const std::string& opt) {
  if (opt.empty() || opt == "auto") return {0};
  std::vector<int> ids;
  std::size_t pos = 0;
  while (pos <= opt.size()) {
    const std::size_t comma = opt.find(',', pos);
    const std::size_t end = (comma == std::string::npos) ? opt.size() : comma;
    if (end > pos) {
      const std::string tok = opt.substr(pos, end - pos);
      try {
        const int id = std::stoi(tok);
        if (id >= 0) ids.push_back(id);
      } catch (...) {
      }
    }
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  if (ids.empty()) return {0};
  // Remove duplicates, keeping first occurrence.
  std::vector<int> unique_ids;
  for (int id : ids) {
    if (std::find(unique_ids.begin(), unique_ids.end(), id) == unique_ids.end())
      unique_ids.push_back(id);
  }
  return unique_ids;
}

}  // namespace sankhya::gpu
