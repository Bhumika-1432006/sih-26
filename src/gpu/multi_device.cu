// SPDX-License-Identifier: Apache-2.0
// SANKHYA - multi-GPU CUDA device utilities (#295).
//
// CUDA-requiring functions only. Compiled when SANKHYA_ENABLE_CUDA is ON via the
// CMakeLists glob of src/gpu/*.cu.

#include "multi_device.hpp"

#include <cuda_runtime.h>

namespace sankhya::gpu {

int device_count() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
  return count;
}

bool can_peer_access(int from, int to) {
  if (from == to) return true;
  int can = 0;
  if (cudaDeviceCanAccessPeer(&can, from, to) != cudaSuccess) return false;
  return can != 0;
}

}  // namespace sankhya::gpu
