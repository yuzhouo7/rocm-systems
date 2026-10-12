/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
Testcase Scenarios :
Unit_hipOccupancy_Positive_MonotoneInVgprCount - Validate that the occupancy reported by
hipOccupancyMaxActiveBlocksPerMultiprocessor never grows when a kernel uses more VGPRs. Registers
are allocated in granules, so several VGPR counts share an occupancy, but spending more registers
can never buy more blocks.
Unit_hipOccupancy_Positive_EqualVgprCountEqualOccupancy - Validate that two kernels which report
the same VGPR count, the same block size and no LDS are given the same occupancy.

Note: the VGPR allocation granule is not exposed through hipDeviceProp_t, so these tests do not
predict an occupancy for a given register count. They check the ordering between kernels, which
holds on every architecture and does not depend on the granule, the wavefront size, or whether the
multiprocessor is a CU or a WGP. A granule rounded with a power of two bitmask breaks that ordering
on the targets whose granule is not a power of two (24 in wave32 on the gfx11/gfx12 parts with 1536
VGPRs per SIMD): there a kernel using 120 VGPRs was charged 136 registers and a kernel using 122
was charged 128, so the larger kernel was reported at the higher occupancy.

hipModuleOccupancyMaxActiveBlocksPerMultiprocessor shares the same implementation, and so the same
rounding, and is covered through it.
*/

#include "occupancy_common.hh"

#include <algorithm>
#include <vector>

namespace {

constexpr int kBlockSize = 128;

// Clobbering vN forces the register allocator to make v0..vN live, so the kernel reports N + 1
// VGPRs whatever the optimization level the suite is built at. Computed register pressure is not
// usable here: it collapses to a single count once the locals are spilled to scratch at -O0.
#define OCCUPANCY_VGPR_KERNEL(N)                                                                   \
  __global__ void vgprKernel##N(float* out) {                                                      \
    asm volatile("" ::: "v" #N);                                                                   \
    *out = 1.0f;                                                                                   \
  }

// Lands on both sides of the 24 register granule of the 1536 VGPR targets, and covers the counts
// the gfx1151 report was raised with.
#define OCCUPANCY_VGPR_KERNEL_LIST(X)                                                              \
  X(7) X(15) X(23) X(39) X(63) X(81) X(97) X(113) X(119) X(121) X(129) X(137) X(161) X(191)

OCCUPANCY_VGPR_KERNEL_LIST(OCCUPANCY_VGPR_KERNEL)

struct OccupancySample {
  int numRegs;
  int numBlocks;
};

OccupancySample Sample(const void* func) {
  hipFuncAttributes attributes;
  HIP_CHECK(hipFuncGetAttributes(&attributes, func));

  int numBlocks = 0;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, func, kBlockSize, 0));

  INFO("numRegs " << attributes.numRegs << ", numBlocks " << numBlocks);
  REQUIRE(attributes.numRegs > 0);
  REQUIRE(attributes.sharedSizeBytes == 0);  // keeps LDS out of the comparison
  REQUIRE(numBlocks > 0);

  return {attributes.numRegs, numBlocks};
}

std::vector<OccupancySample> CollectSamples() {
#define OCCUPANCY_VGPR_SAMPLE(N) Sample(reinterpret_cast<const void*>(vgprKernel##N)),
  std::vector<OccupancySample> samples{OCCUPANCY_VGPR_KERNEL_LIST(OCCUPANCY_VGPR_SAMPLE)};
#undef OCCUPANCY_VGPR_SAMPLE

  std::sort(samples.begin(), samples.end(),
            [](const OccupancySample& a, const OccupancySample& b) {
              return a.numRegs < b.numRegs;
            });
  return samples;
}

}  // namespace

HIP_TEST_CASE(Unit_hipOccupancy_Positive_MonotoneInVgprCount) {
  const std::vector<OccupancySample> samples = CollectSamples();

  for (size_t i = 1; i < samples.size(); ++i) {
    const OccupancySample& lower = samples[i - 1];
    const OccupancySample& higher = samples[i];

    INFO("numRegs " << lower.numRegs << " -> " << higher.numRegs << " reported numBlocks "
                    << lower.numBlocks << " -> " << higher.numBlocks);
    REQUIRE(higher.numBlocks <= lower.numBlocks);
  }
}

HIP_TEST_CASE(Unit_hipOccupancy_Positive_EqualVgprCountEqualOccupancy) {
  const std::vector<OccupancySample> samples = CollectSamples();

  // Same registers, same block size and no LDS leaves nothing for the occupancy to differ on.
  for (size_t i = 1; i < samples.size(); ++i) {
    if (samples[i - 1].numRegs != samples[i].numRegs) {
      continue;
    }

    INFO("numRegs " << samples[i].numRegs << " reported numBlocks " << samples[i - 1].numBlocks
                    << " and " << samples[i].numBlocks);
    REQUIRE(samples[i - 1].numBlocks == samples[i].numBlocks);
  }
}
