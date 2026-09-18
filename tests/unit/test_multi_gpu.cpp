// SPDX-License-Identifier: Apache-2.0
// SANKHYA - unit tests for multi-GPU partitioning and device utilities (#295).
//
// The partition and option-parsing tests are CPU-only and run without a CUDA device.
// The device enumeration tests are guarded by SANKHYA_ENABLE_CUDA.

#include <gtest/gtest.h>

#include "gpu/multi_device.hpp"

namespace sankhya {
namespace {

using gpu::RowPartition;
using gpu::parse_device_ids;
using gpu::partition_rows;

// ---- parse_device_ids ----------------------------------------------------

TEST(MultiGpu, ParseAutoReturnsDevice0) {
  const auto ids = parse_device_ids("auto");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 0);
}

TEST(MultiGpu, ParseEmptyReturnsDevice0) {
  const auto ids = parse_device_ids("");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 0);
}

TEST(MultiGpu, ParseSingleDevice) {
  const auto ids = parse_device_ids("0");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 0);
}

TEST(MultiGpu, ParseTwoDevices) {
  const auto ids = parse_device_ids("0,1");
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_EQ(ids[0], 0);
  EXPECT_EQ(ids[1], 1);
}

TEST(MultiGpu, ParseThreeDevices) {
  const auto ids = parse_device_ids("0,1,2");
  ASSERT_EQ(ids.size(), 3u);
  EXPECT_EQ(ids[0], 0);
  EXPECT_EQ(ids[1], 1);
  EXPECT_EQ(ids[2], 2);
}

TEST(MultiGpu, ParseNonZeroStart) {
  const auto ids = parse_device_ids("2,3");
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_EQ(ids[0], 2);
  EXPECT_EQ(ids[1], 3);
}

TEST(MultiGpu, ParseDuplicatesRemoved) {
  const auto ids = parse_device_ids("0,0,1");
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_EQ(ids[0], 0);
  EXPECT_EQ(ids[1], 1);
}

TEST(MultiGpu, ParseNegativeSkipped) {
  // Negative IDs are silently dropped; result falls back to {0}.
  const auto ids = parse_device_ids("-1,0");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 0);
}

TEST(MultiGpu, ParseGarbageFallsBack) {
  const auto ids = parse_device_ids("abc");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 0);
}

// ---- partition_rows ------------------------------------------------------

TEST(MultiGpu, PartitionZeroRows) {
  const auto parts = partition_rows(0, {0, 1});
  ASSERT_EQ(parts.size(), 2u);
  EXPECT_EQ(parts[0].row_start, 0);
  EXPECT_EQ(parts[0].row_end, 0);
  EXPECT_EQ(parts[0].local_m(), 0);
  EXPECT_EQ(parts[1].row_start, 0);
  EXPECT_EQ(parts[1].row_end, 0);
}

TEST(MultiGpu, PartitionEvenRows) {
  // 10 rows across 2 devices -> 5 each.
  const auto parts = partition_rows(10, {0, 1});
  ASSERT_EQ(parts.size(), 2u);
  EXPECT_EQ(parts[0].device_id, 0);
  EXPECT_EQ(parts[0].row_start, 0);
  EXPECT_EQ(parts[0].row_end, 5);
  EXPECT_EQ(parts[1].device_id, 1);
  EXPECT_EQ(parts[1].row_start, 5);
  EXPECT_EQ(parts[1].row_end, 10);
}

TEST(MultiGpu, PartitionOddRows) {
  // 11 rows across 2 devices -> 6, 5.
  const auto parts = partition_rows(11, {0, 1});
  ASSERT_EQ(parts.size(), 2u);
  EXPECT_EQ(parts[0].local_m(), 6);
  EXPECT_EQ(parts[1].local_m(), 5);
  EXPECT_EQ(parts[0].row_end, parts[1].row_start);
  EXPECT_EQ(parts[1].row_end, 11);
}

TEST(MultiGpu, PartitionThreeDevices) {
  // 10 rows across 3 devices -> 4, 3, 3.
  const auto parts = partition_rows(10, {0, 1, 2});
  ASSERT_EQ(parts.size(), 3u);
  EXPECT_EQ(parts[0].local_m(), 4);
  EXPECT_EQ(parts[1].local_m(), 3);
  EXPECT_EQ(parts[2].local_m(), 3);
  // Contiguous.
  EXPECT_EQ(parts[0].row_end, parts[1].row_start);
  EXPECT_EQ(parts[1].row_end, parts[2].row_start);
  EXPECT_EQ(parts[2].row_end, 10);
}

TEST(MultiGpu, PartitionCoversAllRows) {
  for (int m = 1; m <= 20; ++m) {
    for (int k = 1; k <= 4; ++k) {
      std::vector<int> devs;
      devs.reserve(static_cast<std::size_t>(k));
      for (int i = 0; i < k; ++i) devs.push_back(i);
      const auto parts = partition_rows(m, devs);
      ASSERT_EQ(static_cast<int>(parts.size()), k);
      int total = 0;
      for (const auto& p : parts) total += p.local_m();
      EXPECT_EQ(total, m) << "m=" << m << " k=" << k;
    }
  }
}

TEST(MultiGpu, PartitionSingleDevice) {
  const auto parts = partition_rows(7, {3});
  ASSERT_EQ(parts.size(), 1u);
  EXPECT_EQ(parts[0].device_id, 3);
  EXPECT_EQ(parts[0].row_start, 0);
  EXPECT_EQ(parts[0].row_end, 7);
}

TEST(MultiGpu, PartitionEmptyDeviceList) {
  const auto parts = partition_rows(5, {});
  EXPECT_TRUE(parts.empty());
}

// ---- RowPartition helpers ------------------------------------------------

TEST(MultiGpu, LocalMIsCorrect) {
  RowPartition p{0, 10, 20};
  EXPECT_EQ(p.local_m(), 10);
}

// ---- CUDA-only tests (device_count, can_peer_access) ---------------------
#ifdef SANKHYA_ENABLE_CUDA

TEST(MultiGpu, DeviceCountNonNegative) { EXPECT_GE(gpu::device_count(), 0); }

TEST(MultiGpu, PeerAccessSelfIsTrue) {
  const int cnt = gpu::device_count();
  for (int i = 0; i < cnt; ++i) EXPECT_TRUE(gpu::can_peer_access(i, i));
}

#endif  // SANKHYA_ENABLE_CUDA

}  // namespace
}  // namespace sankhya
