/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Neighbor search on GPU
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <set>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cooperative_groups.h>
#include <cub/warp/warp_merge_sort.cuh>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include "cstone/compressneighbors.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/reducearray.cuh"
#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/traversal/ijloop/ijloop.hpp"
#include "cstone/tree/octree.hpp"

namespace cstone::ijloop
{

namespace gpu_supercluster_nb_list_neighborhood_detail
{

constexpr __forceinline__ bool includeNbSymmetric(unsigned i, unsigned j, unsigned first, unsigned last)
{
    constexpr unsigned blockSize = 8;
    const bool s                 = (i / blockSize) % 2 == (j / blockSize) % 2;
    return (j < first) | (j >= last) | (i == j) | (i < j ? s : !s);
}

template<class T>
constexpr __forceinline__ void atomicAddScalarOrVec(T* ptr, T value)
{
    atomicAdd(ptr, value);
}

template<class T, std::size_t N>
constexpr __forceinline__ void atomicAddScalarOrVec(util::array<T, N>* ptr, util::array<T, N> const& value)
{
#pragma unroll
    for (std::size_t i = 0; i < N; ++i)
        atomicAddScalarOrVec(&((*ptr)[i]), value[i]);
}

template<class Config>
using nbStoragePerICluster =
    std::integral_constant<std::size_t,
                           (Config::compress ? Config::ncMax / Config::expectedCompressionRate : Config::ncMax)>;

template<class Config, class Tc>
__global__ void gpuClusterNbListComputeBboxes(LocalIndex totalNumParticles,
                                              const Tc* const __restrict__ x,
                                              const Tc* const __restrict__ y,
                                              const Tc* const __restrict__ z,
                                              util::tuple<Vec3<Tc>, Vec3<Tc>>* const __restrict__ bboxes)
{
    static_assert(GpuConfig::warpSize % Config::jSize == 0);

    const auto block = cooperative_groups::this_thread_block();
    const auto warp  = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

    const unsigned i = block.thread_index().x + block.group_dim().x * block.group_index().x;

    const Tc xi = x[std::min(i, totalNumParticles - 1)];
    const Tc yi = y[std::min(i, totalNumParticles - 1)];
    const Tc zi = z[std::min(i, totalNumParticles - 1)];

    const unsigned jClusters = iceil(totalNumParticles, Config::jSize);
    const unsigned bboxIdx   = i / Config::jSize;

    if constexpr (Config::jSize >= 3)
    {
        util::array<Tc, 3> bboxMin{xi, yi, zi};
        util::array<Tc, 3> bboxMax{xi, yi, zi};

        const Tc vMin = reduceArray<Config::jSize, false>(bboxMin, [](auto a, auto b) { return std::min(a, b); });
        const Tc vMax = reduceArray<Config::jSize, false>(bboxMax, [](auto a, auto b) { return std::max(a, b); });

        const Tc center = (vMax + vMin) * Tc(0.5);
        const Tc size   = (vMax - vMin) * Tc(0.5);

        const unsigned idx = warp.thread_rank() % Config::jSize;
        if (idx < 3 & bboxIdx < jClusters)
        {
            auto* box     = &bboxes[bboxIdx];
            Tc* centerPtr = (Tc*)box + idx;
            Tc* sizePtr   = centerPtr + 3;
            *centerPtr    = center;
            *sizePtr      = size;
        }
    }
    else
    {
        Vec3<Tc> bboxMin{xi, yi, zi};
        Vec3<Tc> bboxMax{xi, yi, zi};

#pragma unroll
        for (unsigned offset = Config::jSize / 2; offset >= 1; offset /= 2)
        {
            bboxMin = {std::min(warp.shfl_down(bboxMin[0], offset), bboxMin[0]),
                       std::min(warp.shfl_down(bboxMin[1], offset), bboxMin[1]),
                       std::min(warp.shfl_down(bboxMin[2], offset), bboxMin[2])};
            bboxMax = {std::max(warp.shfl_down(bboxMax[0], offset), bboxMax[0]),
                       std::max(warp.shfl_down(bboxMax[1], offset), bboxMax[1]),
                       std::max(warp.shfl_down(bboxMax[2], offset), bboxMax[2])};
        }

        Vec3<Tc> center = (bboxMax + bboxMin) * Tc(0.5);
        Vec3<Tc> size   = (bboxMax - bboxMin) * Tc(0.5);

        if (i % Config::jSize == 0 && bboxIdx < jClusters) bboxes[bboxIdx] = {center, size};
    }
}

template<class Config, unsigned NumWarpsPerBlock>
__device__ __forceinline__ void deduplicateAndStoreNeighbors(unsigned* iClusterNidx,
                                                             const unsigned iClusterNc,
                                                             unsigned* targetIClusterNidx,
                                                             unsigned* targetIClusterNc)
{
    const auto block = cooperative_groups::this_thread_block();
    const auto warp  = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

    constexpr unsigned itemsPerWarp = (Config::ncMax + Config::ncMaxExtra) / GpuConfig::warpSize;
    unsigned items[itemsPerWarp];
#pragma unroll
    for (unsigned i = 0; i < itemsPerWarp; ++i)
    {
        const unsigned nb = warp.thread_rank() * itemsPerWarp + i;
        items[i]          = nb < iClusterNc ? iClusterNidx[nb] : unsigned(-1);
    }
    using WarpSort = cub::WarpMergeSort<unsigned, itemsPerWarp, GpuConfig::warpSize>;
    __shared__ typename WarpSort::TempStorage sortTmp[NumWarpsPerBlock];
    WarpSort(sortTmp[warp.meta_group_rank()]).Sort(items, std::less<unsigned>());

    unsigned prev = warp.shfl_up(items[itemsPerWarp - 1], 1);
    if (warp.thread_rank() == 0) prev = unsigned(-1);
    unsigned unique = 0;
#pragma unroll
    for (unsigned i = 0; i < itemsPerWarp; ++i)
    {
        const unsigned item = items[i];
        if (item != prev & item != unsigned(-1))
        {
            // the following loop implements basically items[unique] = item;
            // but enables scalar replacement of items[]
#pragma unroll
            for (unsigned j = 0; j < itemsPerWarp; ++j)
            {
                if (j == unique) items[unique] = item;
            }
            ++unique;
            prev = item;
        }
    }

    const unsigned totalUnique = inclusiveScanInt(unique);
    assert(totalUnique < Config::ncMax);
    const unsigned startIndex = totalUnique - unique;

    if constexpr (Config::compress)
    {
        // the following loop with if-condition is equivalent to
        // for (unsigned i = 0; i < unique; ++i)
        // but enables scalar replacement of items[]
#pragma unroll
        for (unsigned i = 0; i < itemsPerWarp; ++i)
        {
            if (i < unique)
            {
                const unsigned nb = startIndex + i;
                iClusterNidx[nb]  = items[i];
            }
        }
        const unsigned uniqueNeighbors = warp.shfl(totalUnique, GpuConfig::warpSize - 1);
        assert(uniqueNeighbors < Config::ncMax);
        warpCompressNeighbors(iClusterNidx, (char*)targetIClusterNidx, uniqueNeighbors);
    }
    else
    {
#pragma unroll
        for (unsigned i = 0; i < itemsPerWarp; ++i)
        {
            if (i < unique)
            {
                const unsigned nb      = startIndex + i;
                targetIClusterNidx[nb] = items[i];
            }
        }

        if (warp.thread_rank() == GpuConfig::warpSize - 1) *targetIClusterNc = startIndex + unique;
    }
}

template<class Config, unsigned NumWarpsPerBlock, bool UsePbc, class Tc, class Th, class KeyType>
__global__
    __maxnreg__(72) void gpuClusterNbListBuild(const OctreeNsView<Tc, KeyType> __grid_constant__ tree,
                                               const Box<Tc> __grid_constant__ box,
                                               const LocalIndex firstIParticle,
                                               const LocalIndex lastIParticle,
                                               const Tc* const __restrict__ x,
                                               const Tc* const __restrict__ y,
                                               const Tc* const __restrict__ z,
                                               const Th* const __restrict__ h,
                                               const util::tuple<Vec3<Tc>, Vec3<Tc>>* const __restrict__ jClusterBboxes,
                                               unsigned* const __restrict__ clusterNeighbors,
                                               unsigned* const __restrict__ clusterNeighborsCount,
                                               int* const __restrict__ globalPool,
                                               const Th maxH)
{
    static_assert(Config::ncMax % GpuConfig::warpSize == 0);
    static_assert((Config::ncMax + Config::ncMaxExtra) % GpuConfig::warpSize == 0);

    const auto grid  = cooperative_groups::this_grid();
    const auto block = cooperative_groups::this_thread_block();
    assert(block.dim_threads().x == Config::iSize);
    assert(block.dim_threads().y == GpuConfig::warpSize / Config::iSize);
    assert(block.dim_threads().z == NumWarpsPerBlock);
    static_assert(NumWarpsPerBlock > 0 && Config::iSize * Config::jSize <= GpuConfig::warpSize);
    static_assert(GpuConfig::warpSize % (Config::iSize * Config::jSize) == 0);
    const auto warp = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

    volatile __shared__ int sharedPool[GpuConfig::warpSize * NumWarpsPerBlock];
    __shared__ unsigned nidx[NumWarpsPerBlock][GpuConfig::warpSize / Config::iSize][Config::ncMax + Config::ncMaxExtra];

    const unsigned numTargets =
        iceil(lastIParticle - firstIParticle / Config::iSize * Config::iSize, GpuConfig::warpSize);
    const unsigned firstICluster = firstIParticle / Config::iSize;
    const unsigned lastICluster  = iceil(lastIParticle, Config::iSize);
    const unsigned numJClusters  = iceil(lastIParticle, Config::jSize);

    const TreeNodeIndex* __restrict__ childOffsets   = tree.childOffsets;
    const TreeNodeIndex* __restrict__ internalToLeaf = tree.internalToLeaf;
    const LocalIndex* __restrict__ layout            = tree.layout;
    const Vec3<Tc>* __restrict__ centers             = tree.centers;
    const Vec3<Tc>* __restrict__ sizes               = tree.sizes;

    while (true)
    {
        unsigned target;
        if (warp.thread_rank() == 0) target = atomicAdd(&targetCounterGlob, 1);
        target = warp.shfl(target, 0);

        if (target >= numTargets) return;

        const unsigned iCluster =
            firstICluster + target * (GpuConfig::warpSize / Config::iSize) + block.thread_index().y;

        const unsigned i = std::min(
            std::max(target * GpuConfig::warpSize + warp.thread_rank() + firstIParticle / Config::iSize * Config::iSize,
                     firstIParticle),
            lastIParticle - 1);
        const Vec3<Tc> iPos = {x[i], y[i], z[i]};
        const Th hi         = h[i];

        const Th hBound = Config::symmetric ? maxH : hi;
        Vec3<Tc> bbMin  = {iPos[0] - 2 * hBound, iPos[1] - 2 * hBound, iPos[2] - 2 * hBound};
        Vec3<Tc> bbMax  = {iPos[0] + 2 * hBound, iPos[1] + 2 * hBound, iPos[2] + 2 * hBound};
        Vec3<Tc> iClusterCenter, iClusterSize;
        if constexpr (Config::iSize == 1)
        {
            iClusterCenter = (bbMax + bbMin) * Tc(0.5);
            iClusterSize   = (bbMax - bbMin) * Tc(0.5);
        }
#pragma unroll
        for (unsigned n = 1, s = 0; n < GpuConfig::warpSize; n *= 2, ++s)
        {
            bbMin[0] = std::min(warp.shfl_xor(bbMin[0], 1 << s), bbMin[0]);
            bbMin[1] = std::min(warp.shfl_xor(bbMin[1], 1 << s), bbMin[1]);
            bbMin[2] = std::min(warp.shfl_xor(bbMin[2], 1 << s), bbMin[2]);
            bbMax[0] = std::max(warp.shfl_xor(bbMax[0], 1 << s), bbMax[0]);
            bbMax[1] = std::max(warp.shfl_xor(bbMax[1], 1 << s), bbMax[1]);
            bbMax[2] = std::max(warp.shfl_xor(bbMax[2], 1 << s), bbMax[2]);
            if (n == Config::iSize / 2)
            {
                iClusterCenter = (bbMax + bbMin) * Tc(0.5);
                iClusterSize   = (bbMax - bbMin) * Tc(0.5);
            }
        }

        const Vec3<Tc> targetCenter = (bbMax + bbMin) * Tc(0.5);
        const Vec3<Tc> targetSize   = (bbMax - bbMin) * Tc(0.5);

        const auto distSq = [&](const Vec3<Tc>& jPos)
        { return distanceSq<UsePbc>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box); };

        unsigned nc = 0;

        const auto checkNeighborhood = [&](const unsigned jCluster, const unsigned numLanesValid)
        {
            assert(numLanesValid > 0);

            const unsigned prevJCluster = warp.shfl_up(jCluster, 1);
            const bool validJCluster    = warp.thread_rank() < numLanesValid & jCluster < numJClusters &
                                       (warp.thread_rank() == 0 | prevJCluster != jCluster);

            const auto [jClusterCenter, jClusterSize] =
                validJCluster
                    ? jClusterBboxes[jCluster]
                    : util::tuple<Vec3<Tc>, Vec3<Tc>>({std::numeric_limits<Tc>::max(), std::numeric_limits<Tc>::max(),
                                                       std::numeric_limits<Tc>::max()},
                                                      {Tc(0), Tc(0), Tc(0)});

            for (unsigned c = 0; c < GpuConfig::warpSize / Config::iSize; ++c)
            {
                const auto iClusterC = firstICluster + target * (GpuConfig::warpSize / Config::iSize) + c;
                if (iClusterC < firstICluster | iClusterC >= lastICluster) break;
                const auto iClusterCenterC = warp.shfl(iClusterCenter, c * Config::iSize);
                const auto iClusterSizeC   = warp.shfl(iClusterSize, c * Config::iSize);
                const unsigned ncC         = warp.shfl(nc, c * Config::iSize);
                const bool isNeighbor =
                    (validJCluster & iClusterC * Config::iSize / Config::jSize != jCluster &
                     jCluster * Config::jSize / Config::iSize != iClusterC &
                     (!Config::symmetric ||
                      includeNbSymmetric(iClusterC, jCluster * Config::jSize / Config::iSize, firstICluster))) &&
                    norm2(minDistance(iClusterCenterC, iClusterSizeC, jClusterCenter, jClusterSize, box)) == 0;

                const unsigned nbIndex = exclusiveScanBool(isNeighbor);
                if (isNeighbor & ncC + nbIndex < Config::ncMax + Config::ncMaxExtra)
                    nidx[block.thread_index().z][c][ncC + nbIndex] = jCluster;
                const unsigned newNbs = warp.shfl(nbIndex + isNeighbor, GpuConfig::warpSize - 1);
                if (block.thread_index().y == c) nc += newNbs;
            }
        };

        int jClusterQueue; // warp queue for source jCluster indices
        volatile int* tempQueue = sharedPool + GpuConfig::warpSize * warp.meta_group_rank();
        int* cellQueue =
            globalPool + TravConfig::memPerWarp * (grid.block_rank() * warp.meta_group_size() + warp.meta_group_rank());

        // populate initial cell queue
        if (warp.thread_rank() == 0) cellQueue[0] = 1;
        warp.sync();

        // these variables are always identical on all warp lanes
        int numSources        = 1; // current stack size
        int newSources        = 0; // stack size for next level
        int oldSources        = 0; // cell indices done
        int sourceOffset      = 0; // current level stack pointer, once this reaches numSources, the level is done
        int jClusterFillLevel = 0; // fill level of the source jCluster warp queue

        while (numSources > 0) // While there are source cells to traverse
        {
            int sourceIdx   = sourceOffset + warp.thread_rank();
            int sourceQueue = 0;
            if (warp.thread_rank() < GpuConfig::warpSize / 8)
                sourceQueue = cellQueue[ringAddr(oldSources + sourceIdx)]; // Global source cell index in queue
            sourceQueue = spreadSeg8(sourceQueue);
            sourceIdx   = warp.shfl(sourceIdx, warp.thread_rank() / 8);

            const Vec3<Tc> curSrcCenter = centers[sourceQueue];      // Current source cell center
            const Vec3<Tc> curSrcSize   = sizes[sourceQueue];        // Current source cell center
            const int childBegin        = childOffsets[sourceQueue]; // First child cell
            const bool isNode           = childBegin;
            const bool isClose          = cellOverlap<UsePbc>(curSrcCenter, curSrcSize, targetCenter, targetSize, box);
            const bool isSource         = sourceIdx < numSources; // Source index is within bounds
            const bool isDirect         = isClose && !isNode && isSource;
            const int leafIdx           = isDirect ? internalToLeaf[sourceQueue] : 0; // the cstone leaf index

            // Split
            const bool isSplit     = isNode && isClose && isSource; // Source cell must be split
            const int numChildLane = exclusiveScanBool(isSplit);    // Exclusive scan of numChild
            const int numChildWarp = reduceBool(isSplit);           // Total numChild of current warp
            sourceOffset +=
                imin(GpuConfig::warpSize / 8, numSources - sourceOffset);       // advance current level stack pointer
            int childIdx = oldSources + numSources + newSources + numChildLane; // Child index of current lane
            if (isSplit) cellQueue[ringAddr(childIdx)] = childBegin;            // Queue child cells for next level
            newSources += numChildWarp; // Increment source cell count for next loop

            // check for cellQueue overflow
            const unsigned stackUsed = newSources + numSources - sourceOffset; // current cellQueue size
            if (stackUsed > TravConfig::memPerWarp) return;                    // Exit if cellQueue overflows

            // Direct
            const int firstJCluster = layout[leafIdx] / Config::jSize;
            const int numJClusters  = (iceil(layout[leafIdx + 1], Config::jSize) - firstJCluster) &
                                     -int(isDirect); // Number of jClusters in cell
            bool directTodo            = numJClusters;
            const int numJClustersScan = inclusiveScanInt(numJClusters);  // Inclusive scan of numJClusters
            int numJClustersLane       = numJClustersScan - numJClusters; // Exclusive scan of numJClusters
            int numJClustersWarp =
                shflSync(numJClustersScan, GpuConfig::warpSize - 1); // Total numJClusters of current warp
            int prevJClusterIdx = 0;
            while (numJClustersWarp > 0) // While there are jClusters to process from current source cell set
            {
                tempQueue[warp.thread_rank()] =
                    1; // Default scan input is 1, such that consecutive lanes load consecutive bodies
                if (directTodo && (numJClustersLane < GpuConfig::warpSize))
                {
                    directTodo                  = false;              // Set cell as processed
                    tempQueue[numJClustersLane] = -1 - firstJCluster; // Put first source cell body index into the queue
                }
                const int jClusterIdx = inclusiveSegscanInt(tempQueue[warp.thread_rank()], prevJClusterIdx);
                // broadcast last processed jClusterIdx from the last lane to restart the scan in the next iteration
                prevJClusterIdx = shflSync(jClusterIdx, GpuConfig::warpSize - 1);

                if (numJClustersWarp >= GpuConfig::warpSize) // Process jClusters from current set of source cells
                {
                    checkNeighborhood(jClusterIdx, GpuConfig::warpSize);
                    numJClustersWarp -= GpuConfig::warpSize;
                    numJClustersLane -= GpuConfig::warpSize;
                }
                else // Fewer than warpSize bodies remaining from current source cell set
                {
                    // push the remaining bodies into jClusterQueue
                    int topUp     = shflUpSync(jClusterIdx, jClusterFillLevel);
                    jClusterQueue = (warp.thread_rank() < jClusterFillLevel) ? jClusterQueue : topUp;

                    jClusterFillLevel += numJClustersWarp;
                    if (jClusterFillLevel >= GpuConfig::warpSize) // If this causes jClusterQueue to spill
                    {
                        checkNeighborhood(jClusterQueue, GpuConfig::warpSize);
                        jClusterFillLevel -= GpuConfig::warpSize;
                        // jClusterQueue is now empty; put body indices that spilled into the queue
                        jClusterQueue = shflDownSync(jClusterIdx, numJClustersWarp - jClusterFillLevel);
                    }
                    numJClustersWarp = 0; // No more bodies to process from current source cells
                }
            }

            //  If the current level is done
            if (sourceOffset >= numSources)
            {
                oldSources += numSources;      // Update finished source size
                numSources   = newSources;     // Update current source size
                sourceOffset = newSources = 0; // Initialize next source size and offset
            }
        }

        if (jClusterFillLevel > 0) // If there are leftover direct bodies
            checkNeighborhood(jClusterQueue, jClusterFillLevel);

        for (unsigned c = 0; c < GpuConfig::warpSize / Config::iSize; ++c)
        {
            unsigned ncc      = warp.shfl(nc, c * Config::iSize);
            const unsigned ic = warp.shfl(iCluster, c * Config::iSize);
            if (ic < firstICluster | ic >= lastICluster) continue;

            const unsigned i    = std::min(ic * Config::iSize + block.thread_index().x, lastIParticle - 1);
            const Vec3<Tc> iPos = {x[i], y[i], z[i]};
            const Th hi         = h[i];

            const auto posDiff = [&](const Vec3<Tc>& jPos)
            {
                Vec3<Tc> ijPosDiff = {iPos[0] - jPos[0], iPos[1] - jPos[1], iPos[2] - jPos[2]};
                if constexpr (UsePbc)
                {
                    ijPosDiff[0] -=
                        (box.boundaryX() == BoundaryType::periodic) * box.lx() * std::rint(ijPosDiff[0] * box.ilx());
                    ijPosDiff[1] -=
                        (box.boundaryY() == BoundaryType::periodic) * box.ly() * std::rint(ijPosDiff[1] * box.ily());
                    ijPosDiff[2] -=
                        (box.boundaryZ() == BoundaryType::periodic) * box.lz() * std::rint(ijPosDiff[2] * box.ilz());
                }
                return ijPosDiff;
            };

            constexpr unsigned threadsPerInteraction = Config::iSize * Config::jSize;
            constexpr unsigned jBlocksPerWarp        = GpuConfig::warpSize / threadsPerInteraction;
            const GpuConfig::ThreadMask threadMask   = threadsPerInteraction == GpuConfig::warpSize
                                                           ? ~GpuConfig::ThreadMask(0)
                                                           : (GpuConfig::ThreadMask(1) << threadsPerInteraction) - 1;
            const GpuConfig::ThreadMask jBlockMask =
                threadMask << (threadsPerInteraction * (warp.thread_rank() / threadsPerInteraction));
            unsigned prunedNcc = 0;
            for (unsigned n = 0; n < imin(ncc, Config::ncMax + Config::ncMaxExtra); n += jBlocksPerWarp)
            {
                const unsigned nb = n + block.thread_index().y / Config::jSize;
                const unsigned jCluster =
                    nb < imin(ncc, Config::ncMax + Config::ncMaxExtra) ? nidx[block.thread_index().z][c][nb] : ~0u;
                const unsigned j         = jCluster * Config::jSize + block.thread_index().y % Config::jSize;
                const Vec3<Tc> jPos      = j < lastIParticle
                                               ? Vec3<Tc>{x[j], y[j], z[j]}
                                               : Vec3<Tc>{std::numeric_limits<Tc>::max(), std::numeric_limits<Tc>::max(),
                                                          std::numeric_limits<Tc>::max()};
                const Th hj              = j < lastIParticle ? h[j] : 0;
                const Vec3<Tc> ijPosDiff = posDiff(jPos);
                const Th d2              = norm2(ijPosDiff);
                const Th iRadiusSq       = Th(4) * hi * hi;
                const Th jRadiusSq       = Th(4) * hj * hj;
                const bool keep       = warp.ballot(d2 < iRadiusSq | (Config::symmetric & d2 < jRadiusSq)) & jBlockMask;
                const unsigned offset = exclusiveScanBool(keep & (warp.thread_rank() % threadsPerInteraction == 0));

                if ((warp.thread_rank() % threadsPerInteraction == 0) & keep)
                    nidx[block.thread_index().z][c][prunedNcc + offset] = jCluster;

                prunedNcc += warp.shfl(offset + keep, (GpuConfig::warpSize - 1) / (Config::iSize * Config::jSize) *
                                                          (Config::iSize * Config::jSize));
            }

            ncc = prunedNcc;

            deduplicateAndStoreNeighbors<Config, NumWarpsPerBlock>(
                nidx[block.thread_index().z][c], ncc,
                &clusterNeighbors[(ic - firstICluster) * nbStoragePerICluster<Config>::value],
                &clusterNeighborsCount[ic - firstICluster]);
        }
    }
}

template<class T0, class... T>
__device__ inline constexpr T0 dynamicTupleGet(std::tuple<T0, T...> const& tuple, int index)
{
    T0 res;
    int i = 0;
    util::for_each_tuple(
        [&](auto const& src)
        {
            if (i++ == index) res = src;
        },
        tuple);
    return res;
}

template<class Config, class T0, class... T>
__device__ __forceinline__ void
storeTupleISum(std::tuple<T0, T...> tuple, std::tuple<T0*, T*...> const& ptrs, const unsigned index, const bool store)
{
    const auto block = cooperative_groups::this_thread_block();
    assert(block.dim_threads().x == Config::iThreads);
    const auto warp = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

    if constexpr (std::conjunction_v<std::is_same<T0, T>...> && sizeof...(T) < GpuConfig::warpSize / Config::iThreads)
    {
        const T0 res = reduceTuple<GpuConfig::warpSize / Config::iThreads, true>(tuple, std::plus<T0>());
        if ((block.thread_index().y % (GpuConfig::warpSize / Config::iThreads) <= sizeof...(T)) & store)
        {
            T0* ptr = dynamicTupleGet(ptrs, block.thread_index().y % (GpuConfig::warpSize / Config::iThreads));
            if constexpr (Config::symmetric | (Config::numWarpsPerInteraction > 1))
                atomicAddScalarOrVec(&ptr[index], res);
            else
                ptr[index] = res;
        }
    }
    else
    {
#pragma unroll
        for (unsigned offset = GpuConfig::warpSize / 2; offset >= Config::iThreads; offset /= 2)
            util::for_each_tuple([&](auto& t) { t += warp.shfl_down(t, offset); }, tuple);

        if ((block.thread_index().y % (GpuConfig::warpSize / Config::iThreads) == 0) & store)
            util::for_each_tuple(
                [index](auto* ptr, auto const& t)
                {
                    if constexpr (Config::symmetric | (Config::numWarpsPerInteraction > 1))
                    {
                        atomicAddScalarOrVec(&ptr[index], t);
                    }
                    else { ptr[index] = t; }
                },
                ptrs, tuple);
    }
}

template<class Config, class T0, class... T>
constexpr __device__ void
storeTupleJSum(std::tuple<T0, T...> tuple, std::tuple<T0*, T*...> const& ptrs, const unsigned index, const bool store)
{
    const auto block = cooperative_groups::this_thread_block();
    assert(block.dim_threads().x == Config::iThreads);
    const auto warp = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

    if constexpr (std::conjunction_v<std::is_same<T0, T>...> && sizeof...(T) < Config::iThreads)
    {
        const T0 res = reduceTuple<Config::iThreads, false>(tuple, std::plus<T0>());
        if ((block.thread_index().x <= sizeof...(T)) & store)
        {
            T0* ptr = dynamicTupleGet(ptrs, block.thread_index().x);
            atomicAddScalarOrVec(&ptr[index], res);
        }
    }
    else
    {
#pragma unroll
        for (unsigned offset = Config::iThreads / 2; offset >= 1; offset /= 2)
            util::for_each_tuple([&](auto& t) { t += warp.shfl_down(t, offset); }, tuple);

        if ((block.thread_index().x == 0) & store)
            util::for_each_tuple([index](auto* ptr, auto const& t) { atomicAddScalarOrVec(&ptr[index], t); }, ptrs,
                                 tuple);
    }
}

struct SuperclusterInfo
{
    unsigned index, neighborsCount, dataIndex;

    constexpr bool operator<(const SuperclusterInfo& other) const { return neighborsCount >= other.neighborsCount; }
};

template<class Config,
         unsigned NumSuperclustersPerBlock,
         bool UsePbc,
         Symmetry Sym,
         class Tc,
         class Th,
         class In,
         class Out,
         class Interaction>
__global__
__launch_bounds__(Config::iThreads* Config::jSize* NumSuperclustersPerBlock) void gpuClusterNbListNeighborhoodKernel(
    const Box<Tc> __grid_constant__ box,
    const LocalIndex totalParticles,
    const LocalIndex firstIParticle,
    const LocalIndex lastIParticle,
    const Tc* __restrict__ x,
    const Tc* __restrict__ y,
    const Tc* __restrict__ z,
    const Th* __restrict__ h,
    const In __grid_constant__ input,
    const Out __grid_constant__ output,
    const Interaction interaction,
    const std::uint32_t* __restrict__ neighborData,
    const SuperclusterInfo* __restrict__ superclusterInfo)
{
    static_assert(Config::ncMax % GpuConfig::warpSize == 0);
    static_assert(NumSuperclustersPerBlock > 0);
    static_assert(Config::iThreads * Config::jSize >= GpuConfig::warpSize);
    static_assert(Config::iThreads * Config::jSize % GpuConfig::warpSize == 0);

    const auto block = cooperative_groups::this_thread_block();
    assert(block.dim_threads().x == Config::iThreads);
    assert(block.dim_threads().y == Config::jSize);
    assert(block.dim_threads().z == NumWarpsPerBlock);

    const unsigned firstISupercluster = firstIParticle / Config::iSuperclusterSize;
    const unsigned lastISupercluster  = iceil(lastIParticle, Config::iSuperclusterSize);
    const unsigned numISuperclusters  = lastISupercluster - firstISupercluster;
    const unsigned iSuperclusterIndex = block.group_index().x * NumSuperclustersPerBlock + block.thread_index().z;
    if (iSuperclusterIndex >= numISuperclusters) return;

    auto [iSupercluster, iSuperclusterNeighborsCount, iSuperclusterDataIndex] = superclusterInfo[iSuperclusterIndex];

    using particleData_t = decltype(loadParticleData(x, y, z, h, input, 0));

    // TODO: bank-conflict friendly SoA layout
    __shared__ particleData_t
        iSuperclusterDataBuffer[NumSuperclustersPerBlock][Config::iClustersPerSupercluster * Config::iSize];
    particleData_t* iSuperclusterData = iSuperclusterDataBuffer[block.thread_index().z];
    {
        const unsigned base = iSupercluster * Config::iSuperclusterSize;
        for (unsigned offset = block.thread_index().y * Config::iThreads + block.thread_index().x;
             offset < Config::iClustersPerSupercluster * Config::iSize; offset += Config::iThreads * Config::jSize)
        {
            const unsigned i = base + offset;
            iSuperclusterData[offset] =
                i < totalParticles ? loadParticleData(x, y, z, h, input, i) : dummyParticleData(x, y, z, h, input, i);
        }
    }

    // TODO: proper size
    __shared__ unsigned nbDataBuffer[NumSuperclustersPerBlock][Config::ncMax * 2];
    unsigned* const nbData = nbDataBuffer[block.thread_index().z];

    const unsigned maskSize =
        (iSuperclusterNeighborsCount * Config::iClustersPerSupercluster * Config::numWarpsPerInteraction + 31) / 32;
    const unsigned nbDataSize = iSuperclusterNeighborsCount + maskSize;

    if constexpr (Config::compress)
    {
        // TODO
        static_assert(!Config::compress, "TODO!");
    }
    else
    {
        for (unsigned n = block.thread_index().y * Config::iThreads + block.thread_index().x; n < nbDataSize;
             n += Config::iThreads * Config::jSize)
            nbData[n] = neighborData[iSuperclusterDataIndex + n];
    }

    block.sync();

    constexpr unsigned iClustersPerWarp = Config::iThreads / Config::iSize;
    using result_t                      = decltype(interaction(particleData_t(), particleData_t(), Vec3<Tc>(), Tc(0)));
    std::array<result_t, Config::iClustersPerSupercluster / iClustersPerWarp> iResults = {};
    const unsigned warpIndex      = block.thread_index().y / (Config::jSize / Config::numWarpsPerInteraction);
    const unsigned iClusterOffset = iClustersPerWarp == 1 ? 0 : block.thread_index().x / Config::iSize;

    for (unsigned nb = 0; nb < iSuperclusterNeighborsCount; ++nb)
    {
        const unsigned maskStartIndex = nb * (Config::iClustersPerSupercluster * Config::numWarpsPerInteraction) +
                                        (warpIndex * Config::iClustersPerSupercluster);
        const unsigned warpMask =
            (nbData[maskStartIndex / 32] >> (maskStartIndex % 32)) & ((1 << Config::iClustersPerSupercluster) - 1);

        if (warpMask)
        {
            const unsigned jCluster = nb < iSuperclusterNeighborsCount ? nbData[nb + maskSize] : ~0u;
            const unsigned j        = jCluster * Config::jSize + block.thread_index().y;
            const auto jData        = (nb < iSuperclusterNeighborsCount & j < totalParticles)
                                          ? loadParticleData(x, y, z, h, input, j)
                                          : dummyParticleData(x, y, z, h, input, j);
            result_t jResult        = {};

            for (unsigned c = 0; c < Config::iClustersPerSupercluster; c += iClustersPerWarp)
            {
                const unsigned ci = c + iClusterOffset;
                const bool mask   = (warpMask >> ci) & 1;
                if (mask)
                {
                    const auto iCluster = iSupercluster * Config::iClustersPerSupercluster + ci;
                    const auto i        = iCluster * Config::iSize + block.thread_index().x % Config::iSize;
                    if (!Config::symmetric | (iCluster != jCluster * Config::jSize / Config::iSize) | (i <= j))
                    {
                        const auto iData =
                            iSuperclusterData[ci * Config::iSize + block.thread_index().x % Config::iSize];
                        const auto [ijPosDiff, distSq] = posDiffAndDistSq(UsePbc, box, iData, jData);
                        auto ijInteraction             = interaction(iData, jData, ijPosDiff, distSq);
                        if (distSq < radiusSq(iData)) updateResult(iResults[c / iClustersPerWarp], ijInteraction);
                        if (Config::symmetric & (distSq < radiusSq(jData)) & (i != j))
                        {
                            if constexpr (std::is_same_v<Sym, symmetry::Asymmetric>)
                                ijInteraction = interaction(jData, iData, -ijPosDiff, distSq);
                            updateResult(jResult, ijInteraction);
                        }
                    }
                }
            }

            if constexpr (Config::symmetric)
            {
                if constexpr (std::is_same_v<Sym, symmetry::Odd>)
                    util::for_each_tuple([](auto& v) { v = -v; }, jResult);

                storeTupleJSum<Config>(jResult, output, j, j >= firstIParticle & j < lastIParticle);
            }
        }
    }

    for (unsigned c = 0; c < Config::iClustersPerSupercluster; c += iClustersPerWarp)
    {
        const unsigned ci = c + iClusterOffset;
        const auto i =
            iSupercluster * Config::iSuperclusterSize + ci * Config::iSize + block.thread_index().x % Config::iSize;
        storeTupleISum<Config>(iResults[c / iClustersPerWarp], output, i, i >= firstIParticle & i < lastIParticle);
    }
}

template<class Config, class Tc, class Th>
struct GpuSuperclusterNbListNeighborhoodImpl
{
    Box<Tc> box;
    LocalIndex totalParticles, firstIParticle, lastIParticle;
    const Tc *x, *y, *z;
    const Th* h;
    thrust::universal_vector<std::uint32_t> neighborData;
    thrust::universal_vector<SuperclusterInfo> superclusterInfo;

    template<class... In, class... Out, class Interaction, Symmetry Sym>
    void
    ijLoop(std::tuple<In*...> const& input, std::tuple<Out*...> const& output, Interaction&& interaction, Sym) const
    {
        const LocalIndex numParticles = lastIParticle - firstIParticle;
        if (Config::symmetric | (Config::numWarpsPerInteraction > 1))
        {
            util::for_each_tuple(
                [&](auto* ptr)
                { checkGpuErrors(cudaMemsetAsync(ptr + firstIParticle, 0, sizeof(decltype(*ptr)) * numParticles)); },
                output);
        }

        constexpr unsigned iSuperclusterSize = Config::iSize * Config::iClustersPerSupercluster;
        const LocalIndex firstISupercluster  = firstIParticle / iSuperclusterSize;
        const LocalIndex lastISupercluster   = iceil(lastIParticle, iSuperclusterSize);
        const LocalIndex numISuperclusters   = lastISupercluster - firstISupercluster;

        constexpr unsigned numSuperclustersPerBlock = 64 / (Config::iThreads * Config::jSize);
        const dim3 blockSize                        = {Config::iThreads, Config::jSize, numSuperclustersPerBlock};
        const unsigned numBlocks                    = iceil(numISuperclusters, numSuperclustersPerBlock);
        gpuClusterNbListNeighborhoodKernel<Config, numSuperclustersPerBlock, true, Sym><<<numBlocks, blockSize>>>(
            box, totalParticles, firstIParticle, lastIParticle, x, y, z, h, makeConstRestrict(input), output,
            std::forward<Interaction>(interaction), rawPtr(neighborData), rawPtr(superclusterInfo));
        checkGpuErrors(cudaGetLastError());
    }

    Statistics stats() const
    {
        return {.numParticles = lastIParticle - firstIParticle,
                .numBytes     = neighborData.size() * sizeof(typename decltype(neighborData)::value_type) +
                            superclusterInfo.size() * sizeof(typename decltype(superclusterInfo)::value_type)};
    }
};

template<unsigned NcMax                   = 256,
         unsigned ISize                   = 4,
         unsigned JSize                   = 4,
         unsigned ExpectedCompressionRate = 0,
         bool Symmetric                   = true>
struct GpuSuperclusterNbListNeighborhoodConfig
{
    static constexpr unsigned ncMax                   = NcMax;
    static constexpr unsigned iSize                   = ISize;
    static constexpr unsigned jSize                   = JSize;
    static constexpr unsigned expectedCompressionRate = ExpectedCompressionRate;
    static constexpr bool compress                    = ExpectedCompressionRate > 1;
    static constexpr bool symmetric                   = Symmetric;
    static constexpr unsigned ncMaxExtra              = NcMax;

    static constexpr unsigned iClustersPerSupercluster = std::max(jSize, GpuConfig::warpSize / iSize);
    static constexpr unsigned iThreads                 = std::max(iSize, GpuConfig::warpSize / jSize);
    static constexpr unsigned iSuperclusterSize        = iSize * iClustersPerSupercluster;
    static constexpr unsigned numWarpsPerInteraction = (iSize * jSize + GpuConfig::warpSize - 1) / GpuConfig::warpSize;

    template<unsigned NewNcMax>
    using withNcMax =
        GpuSuperclusterNbListNeighborhoodConfig<NewNcMax, ISize, JSize, ExpectedCompressionRate, Symmetric>;

    template<unsigned NewISize, unsigned newJSize>
    using withClusterSize =
        GpuSuperclusterNbListNeighborhoodConfig<NcMax, NewISize, newJSize, ExpectedCompressionRate, Symmetric>;
    template<unsigned NewExpectedCompressionRate>
    using withCompression =
        GpuSuperclusterNbListNeighborhoodConfig<NcMax, ISize, JSize, NewExpectedCompressionRate, Symmetric>;
    using withoutCompression = withCompression<0>;
    using withSymmetry = GpuSuperclusterNbListNeighborhoodConfig<NcMax, ISize, JSize, ExpectedCompressionRate, true>;
    using withoutSymmetry =
        GpuSuperclusterNbListNeighborhoodConfig<NcMax, ISize, JSize, ExpectedCompressionRate, false>;
};

} // namespace gpu_supercluster_nb_list_neighborhood_detail

template<class Config = gpu_supercluster_nb_list_neighborhood_detail::GpuSuperclusterNbListNeighborhoodConfig<>>
struct GpuSuperclusterNbListNeighborhood
{
    template<unsigned NcMax>
    using withNcMax = GpuSuperclusterNbListNeighborhood<typename Config::withNcMax<NcMax>>;
    template<unsigned ISize, unsigned JSize>
    using withClusterSize = GpuSuperclusterNbListNeighborhood<typename Config::withClusterSize<ISize, JSize>>;
    template<unsigned ExpectedCompressionRate>
    using withCompression =
        GpuSuperclusterNbListNeighborhood<typename Config::withCompression<ExpectedCompressionRate>>;
    using withoutCompression = GpuSuperclusterNbListNeighborhood<typename Config::withoutCompression>;
    using withSymmetry       = GpuSuperclusterNbListNeighborhood<typename Config::withSymmetry>;
    using withoutSymmetry    = GpuSuperclusterNbListNeighborhood<typename Config::withoutSymmetry>;

    template<class Tc, class KeyType, class Th>
    gpu_supercluster_nb_list_neighborhood_detail::GpuSuperclusterNbListNeighborhoodImpl<Config, Tc, Th>
    build(const OctreeNsView<Tc, KeyType>& tree,
          const Box<Tc>& box,
          const LocalIndex totalParticles,
          const LocalIndex firstIParticle,
          const LocalIndex lastIParticle,
          const Tc* x,
          const Tc* y,
          const Tc* z,
          const Th* h) const
    {
        using namespace gpu_supercluster_nb_list_neighborhood_detail;

        const LocalIndex firstICluster      = firstIParticle / Config::iSize;
        const LocalIndex lastICluster       = iceil(lastIParticle, Config::iSize);
        const LocalIndex firstISupercluster = firstIParticle / Config::iSuperclusterSize;
        const LocalIndex lastISupercluster  = iceil(lastIParticle, Config::iSuperclusterSize);
        const LocalIndex numISuperclusters  = lastISupercluster - firstISupercluster;

        GpuSuperclusterNbListNeighborhoodImpl<Config, Tc, Th> nbList{
            box,
            totalParticles,
            firstIParticle,
            lastIParticle,
            x,
            y,
            z,
            h,
            thrust::universal_vector<std::uint32_t>(),
            thrust::universal_vector<SuperclusterInfo>(numISuperclusters)};

#pragma omp parallel
        {
            const unsigned ngmax = Config::ncMax * Config::jSize;
            std::vector<LocalIndex> neighborBuffer(ngmax);
#pragma omp for
            for (unsigned iSupercluster = firstISupercluster; iSupercluster < lastISupercluster; ++iSupercluster)
            {
                std::array<std::set<LocalIndex>, Config::iClustersPerSupercluster> clusterNeighbors;
                std::set<LocalIndex> superclusterNeighbors;
                for (unsigned ci = 0; ci < Config::iClustersPerSupercluster; ++ci)
                {
                    for (unsigned pi = 0; pi < Config::iSize; ++pi)
                    {
                        const unsigned i = iSupercluster * Config::iSuperclusterSize + ci * Config::iSize + pi;
                        if ((i >= firstIParticle / Config::iSize * Config::iSize) & (i < lastIParticle))
                        {
                            unsigned nbs = findNeighbors(i, x, y, z, h, tree, box, ngmax, neighborBuffer.data());
                            if (nbs >= ngmax)
                            {
                                printf("Too many neighbors!\n");
                                nbs = ngmax;
                            }
                            if (nbs < ngmax) neighborBuffer[nbs++] = i;
                            for (unsigned nb = 0; nb < nbs; ++nb)
                            {
                                const unsigned iCluster = i / Config::iSize;
                                const unsigned jSubCluster =
                                    neighborBuffer[nb] / (Config::jSize / Config::numWarpsPerInteraction);
                                const unsigned jCluster      = jSubCluster / Config::numWarpsPerInteraction;
                                const unsigned jSupercluster = neighborBuffer[nb] / Config::iSuperclusterSize;
                                if (!Config::symmetric |
                                    includeNbSymmetric(iCluster, jCluster * Config::jSize / Config::iSize,
                                                       firstICluster, lastICluster))
                                    clusterNeighbors[ci].insert(jSubCluster);
                            }
                        }
                    }
                    for (const auto jSubCluster : clusterNeighbors[ci])
                        superclusterNeighbors.insert(jSubCluster / Config::numWarpsPerInteraction);
                }

                const unsigned maskSize =
                    (superclusterNeighbors.size() * Config::iClustersPerSupercluster * Config::numWarpsPerInteraction +
                     31) /
                    32;
                std::vector<std::uint32_t> data(maskSize);
                for (unsigned nb = 0; const auto jCluster : superclusterNeighbors)
                {
                    for (unsigned ci = 0; ci < Config::iClustersPerSupercluster; ++ci)
                    {
                        for (unsigned warp = 0; warp < Config::numWarpsPerInteraction; ++warp)
                        {
                            if (clusterNeighbors[ci].contains(jCluster * Config::numWarpsPerInteraction + warp))
                            {
                                const unsigned maskBitindex =
                                    nb * (Config::iClustersPerSupercluster * Config::numWarpsPerInteraction) +
                                    warp * Config::iClustersPerSupercluster + ci;
                                data[maskBitindex / 32] |= std::uint32_t(1) << (maskBitindex % 32);
                            }
                        }
                    }
                    ++nb;
                }
                data.insert(data.end(), superclusterNeighbors.begin(), superclusterNeighbors.end());
                unsigned start;
#pragma omp critical
                {
                    start = nbList.neighborData.size();
                    nbList.neighborData.insert(nbList.neighborData.end(), data.begin(), data.end());
                    // printf("SC %u:\n", iSupercluster);
                    // for (auto x : superclusterNeighbors)
                    // printf("%u ", x);
                    // printf("\n");
                }
                nbList.superclusterInfo[iSupercluster - firstISupercluster] = {
                    iSupercluster, unsigned(superclusterNeighbors.size()), start};
            }
        }
        thrust::stable_sort(thrust::device, nbList.superclusterInfo.begin(), nbList.superclusterInfo.end());
        /*for (auto x : nbList.neighborData)
            printf("%x ", x);
        printf("\n");
        for (auto x : nbList.superclusterInfo)
            printf("(%u %u) ", std::get<0>(x), std::get<1>(x));
        printf("\n");*/

        return nbList;
    }
};

} // namespace cstone::ijloop
