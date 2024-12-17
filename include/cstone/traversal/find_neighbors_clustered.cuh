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
 * @brief Neighbor search on GPU with particle cluster-cluster interactions
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <tuple>

#include <cuda/barrier>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cub/warp/warp_merge_sort.cuh>

#include "cstone/compressneighbors.hpp"
#include "cstone/compressneighbors.cuh"
#include "cstone/primitives/warpscan.cuh"
#include "cstone/reducearray.cuh"
#include "cstone/traversal/find_neighbors.cuh"

namespace cstone
{

struct ClusterConfig
{
    static constexpr unsigned iSize                   = 4;
    static constexpr unsigned jSize                   = 4;
    static constexpr unsigned expectedCompressionRate = 10;
};

namespace detail
{

template<std::size_t I, class... Ts>
__device__ __forceinline__ auto& tupleGetOrScalar(std::tuple<Ts...>& t)
{
    return std::get<I>(t);
}

template<std::size_t I, class... Ts>
__device__ __forceinline__ auto const& tupleGetOrScalar(std::tuple<Ts...> const& t)
{
    return std::get<I>(t);
}

template<std::size_t I, class T, std::size_t N>
__device__ __forceinline__ T& tupleGetOrScalar(util::array<T, N>& a)
{
    return a[I];
}

template<std::size_t I, class T>
__device__ __forceinline__ T const& tupleGetOrScalar(T const& t)
{
    return t;
}

template<std::size_t I, class F, class... Tuples>
__device__ __forceinline__ void tupleForeachElement(F&& f, Tuples&&... tuples)
{
    f(tupleGetOrScalar<I>(std::forward<Tuples>(tuples))...);
}

template<std::size_t... Is, class F, class... Tuples>
__device__ __forceinline__ void tupleForeachImpl(std::index_sequence<Is...>, F&& f, Tuples&&... tuples)
{
    (..., (tupleForeachElement<Is>(std::forward<F>(f), std::forward<Tuples>(tuples)...)));
}

template<class F, class Tuple, class... Tuples>
__device__ __forceinline__ void tupleForeach(F&& f, Tuple&& tuple, Tuples&&... tuples)
{
    tupleForeachImpl(std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>>>(), std::forward<F>(f),
                     std::forward<Tuple>(tuple), std::forward<Tuples>(tuples)...);
}

template<class T0, class... T>
__device__ __forceinline__ T0 dynamicTupleGet(std::tuple<T0, T...> const& tuple, std::size_t index)
{
    T0 res;
    std::size_t i = 0;
    tupleForeach(
        [&](auto const& src)
        {
            if (i++ == index) res = src;
        },
        tuple);
    return res;
}

template<int Symmetric, class T0, class... T>
__device__ __forceinline__ void
storeTupleJSum(std::tuple<T0, T...>& tuple, std::tuple<T0*, T*...> const& ptrs, bool store)
{
    const auto block = cooperative_groups::this_thread_block();
    assert(block.dim_threads().x == ClusterConfig::iSize);
    const auto warp = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

    if constexpr (std::conjunction_v<std::is_same<T0, T>...> && sizeof...(T) < ClusterConfig::iSize)
    {
        const T0 res = reduceTuple<ClusterConfig::iSize, false>(tuple, std::plus<T0>());
        if ((block.thread_index().x <= sizeof...(T)) & store)
        {
            T0* ptr = dynamicTupleGet(ptrs, block.thread_index().x);
            atomicAdd(ptr, Symmetric * res);
        }
    }
    else
    {
#pragma unroll
        for (unsigned offset = ClusterConfig::iSize / 2; offset >= 1; offset /= 2)
            tupleForeach([&](auto& t) { t += warp.shfl_down(t, offset); }, tuple);

        if ((block.thread_index().x == 0) & store)
            detail::tupleForeach([](auto* ptr, auto const& t) { atomicAdd(ptr, Symmetric * t); }, ptrs, tuple);
    }
}

template<bool Symmetric, class T0, class... T>
__device__ __forceinline__ void
storeTupleISum(std::tuple<T0, T...>& tuple, std::tuple<T0*, T*...> const& ptrs, bool store)
{
    const auto block = cooperative_groups::this_thread_block();
    assert(block.dim_threads().x == ClusterConfig::iSize);
    const auto warp = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

    if constexpr (std::conjunction_v<std::is_same<T0, T>...> && sizeof...(T) < ClusterConfig::jSize)
    {
        const T0 res = reduceTuple<GpuConfig::warpSize / ClusterConfig::iSize, true>(tuple, std::plus<T0>());
        if ((block.thread_index().y <= sizeof...(T)) & store)
        {
            T0* ptr = dynamicTupleGet(ptrs, block.thread_index().y);
            if constexpr (Symmetric)
                atomicAdd(ptr, res);
            else
                *ptr = res;
        }
    }
    else
    {
#pragma unroll
        for (unsigned offset = GpuConfig::warpSize / 2; offset >= ClusterConfig::iSize; offset /= 2)
            detail::tupleForeach([&](auto& t) { t += warp.shfl_down(t, offset); }, tuple);

        if ((block.thread_index().y == 0) & store)
            detail::tupleForeach(
                [](auto* ptr, auto const& t)
                {
                    if constexpr (Symmetric) { atomicAdd(ptr, t); }
                    else { *ptr = t; }
                },
                ptrs, tuple);
    }
}

template<unsigned warpsPerBlock, unsigned NcMax, bool Compress>
__device__ __forceinline__ void deduplicateAndStoreNeighbors(unsigned* iClusterNidx,
                                                             const unsigned iClusterNc,
                                                             unsigned* targetIClusterNidx,
                                                             unsigned* targetIClusterNc)
{
    namespace cg     = cooperative_groups;
    const auto block = cg::this_thread_block();
    const auto warp  = cg::tiled_partition<GpuConfig::warpSize>(block);

    constexpr unsigned itemsPerWarp = NcMax / GpuConfig::warpSize;
    unsigned items[itemsPerWarp];
#pragma unroll
    for (unsigned i = 0; i < itemsPerWarp; ++i)
    {
        const unsigned nb = warp.thread_rank() * itemsPerWarp + i;
        items[i]          = nb < iClusterNc ? iClusterNidx[nb] : unsigned(-1);
    }
    using WarpSort = cub::WarpMergeSort<unsigned, itemsPerWarp, GpuConfig::warpSize>;
    __shared__ typename WarpSort::TempStorage sortTmp[warpsPerBlock];
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
    assert(totalUnique < NcMax);
    const unsigned startIndex = totalUnique - unique;

    if constexpr (Compress)
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
        assert(uniqueNeighbors < NcMax);
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

constexpr __forceinline__ bool includeNbSymmetric(unsigned i, unsigned j)
{
    constexpr unsigned block_size = 32;
    const bool s                  = (i / block_size) % 2 == (j / block_size) % 2;
    return i < j ? s : !s;
}

} // namespace detail

template<class Tc>
__global__ void computeClusterBoundingBoxes(cstone::LocalIndex firstBody,
                                            cstone::LocalIndex lastBody,
                                            const Tc* const __restrict__ x,
                                            const Tc* const __restrict__ y,
                                            const Tc* const __restrict__ z,
                                            util::tuple<Vec3<Tc>, Vec3<Tc>>* const __restrict__ bboxes)
{
    static_assert(GpuConfig::warpSize % ClusterConfig::jSize == 0);

    namespace cg     = cooperative_groups;
    const auto block = cg::this_thread_block();
    const auto warp  = cg::tiled_partition<GpuConfig::warpSize>(block);

    const unsigned i = block.thread_index().x + block.group_dim().x * block.group_index().x;

    const Tc xi = x[std::min(i, lastBody - 1)];
    const Tc yi = y[std::min(i, lastBody - 1)];
    const Tc zi = z[std::min(i, lastBody - 1)];

    if constexpr (ClusterConfig::jSize <= 3)
    {
        util::array<Tc, 3> bboxMin{xi, yi, zi};
        util::array<Tc, 3> bboxMax{xi, yi, zi};

        const Tc vMin =
            reduceArray<ClusterConfig::jSize, false>(bboxMin, [](auto a, auto b) { return std::min(a, b); });
        const Tc vMax =
            reduceArray<ClusterConfig::jSize, false>(bboxMax, [](auto a, auto b) { return std::max(a, b); });

        const Tc center = (vMax + vMin) * Tc(0.5);
        const Tc size   = (vMax - vMin) * Tc(0.5);

        const unsigned idx = warp.thread_rank() % ClusterConfig::jSize;
        if (idx < 3)
        {
            auto* box     = &bboxes[i / ClusterConfig::jSize];
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
        for (unsigned offset = ClusterConfig::jSize / 2; offset >= 1; offset /= 2)
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

        if (i % ClusterConfig::jSize == 0 && i < lastBody) bboxes[i / ClusterConfig::jSize] = {center, size};
    }
}

template<unsigned warpsPerBlock,
         bool UsePbc               = true,
         bool BypassL1CacheOnLoads = true,
         unsigned NcMax            = 256,
         bool Compress             = false,
         bool Symmetric            = false,
         class Tc,
         class Th,
         class KeyType>
__global__
    __maxnreg__(72) void findClusterNeighbors(const cstone::LocalIndex firstBody,
                                              const cstone::LocalIndex lastBody,
                                              const Tc* const __restrict__ x,
                                              const Tc* const __restrict__ y,
                                              const Tc* const __restrict__ z,
                                              const Th* const __restrict__ h,
                                              const util::tuple<Vec3<Tc>, Vec3<Tc>>* const __restrict__ jClusterBboxes,
                                              OctreeNsView<Tc, KeyType> tree,
                                              const __grid_constant__ Box<Tc> box,
                                              unsigned* const __restrict__ ncClustered,
                                              unsigned* const __restrict__ nidxClustered,
                                              int* const globalPool)
{
    static_assert(NcMax % GpuConfig::warpSize == 0, "NcMax must be divisible by warp size");
    namespace cg = cooperative_groups;

    const auto grid  = cg::this_grid();
    const auto block = cg::this_thread_block();
    assert(block.dim_threads().x == ClusterConfig::iSize);
    assert(block.dim_threads().y == GpuConfig::warpSize / ClusterConfig::iSize);
    assert(block.dim_threads().z == warpsPerBlock);
    static_assert(warpsPerBlock > 0 && ClusterConfig::iSize * ClusterConfig::jSize <= GpuConfig::warpSize);
    static_assert(GpuConfig::warpSize % (ClusterConfig::iSize * ClusterConfig::jSize) == 0);
    const auto warp = cg::tiled_partition<GpuConfig::warpSize>(block);

    volatile __shared__ int sharedPool[GpuConfig::warpSize * warpsPerBlock];
    __shared__ unsigned nidx[warpsPerBlock][GpuConfig::warpSize / ClusterConfig::iSize][NcMax];

    assert(firstBody == 0); // TODO: other cases
    const unsigned numTargets   = iceil(lastBody, GpuConfig::warpSize);
    const unsigned numIClusters = iceil(lastBody, ClusterConfig::iSize);
    const unsigned numJClusters = iceil(lastBody, ClusterConfig::jSize);

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

        const unsigned iCluster = target * (GpuConfig::warpSize / ClusterConfig::iSize) + block.thread_index().y;

        const unsigned i    = imin(target * GpuConfig::warpSize + warp.thread_rank(), lastBody - 1);
        const Vec3<Tc> iPos = {x[i], y[i], z[i]};
        const Th hi         = h[i];

        Vec3<Tc> bbMin = {iPos[0] - 2 * hi, iPos[1] - 2 * hi, iPos[2] - 2 * hi};
        Vec3<Tc> bbMax = {iPos[0] + 2 * hi, iPos[1] + 2 * hi, iPos[2] + 2 * hi};
        Vec3<Tc> iClusterCenter, iClusterSize;
        if constexpr (ClusterConfig::iSize == 1)
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
            if (n == ClusterConfig::iSize / 2)
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

            for (unsigned c = 0; c < GpuConfig::warpSize / ClusterConfig::iSize; ++c)
            {
                const auto iClusterC = target * (GpuConfig::warpSize / ClusterConfig::iSize) + c;
                if (iClusterC >= numIClusters) break;
                const auto iClusterCenterC = warp.shfl(iClusterCenter, c * ClusterConfig::iSize);
                const auto iClusterSizeC   = warp.shfl(iClusterSize, c * ClusterConfig::iSize);
                const unsigned ncC         = warp.shfl(nc, c * ClusterConfig::iSize);
                const bool isNeighbor =
                    (validJCluster & iClusterC * ClusterConfig::iSize / ClusterConfig::jSize != jCluster &
                     jCluster * ClusterConfig::jSize / ClusterConfig::iSize != iClusterC &
                     (!Symmetric ||
                      detail::includeNbSymmetric(iClusterC, jCluster * ClusterConfig::jSize / ClusterConfig::iSize))) &&
                    norm2(minDistance(iClusterCenterC, iClusterSizeC, jClusterCenter, jClusterSize, box)) == 0;

                const unsigned nbIndex = exclusiveScanBool(isNeighbor);
                if (isNeighbor & ncC + nbIndex < NcMax) nidx[block.thread_index().z][c][ncC + nbIndex] = jCluster;
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
            const int firstJCluster = layout[leafIdx] / ClusterConfig::jSize;
            const int numJClusters  = (iceil(layout[leafIdx + 1], ClusterConfig::jSize) - firstJCluster) &
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

        for (unsigned c = 0; c < GpuConfig::warpSize / ClusterConfig::iSize; ++c)
        {
            unsigned ncc      = warp.shfl(nc, c * ClusterConfig::iSize);
            const unsigned ic = warp.shfl(iCluster, c * ClusterConfig::iSize);
            if (ic >= numIClusters) continue;

            const unsigned i    = imin(ic * ClusterConfig::iSize + block.thread_index().x, lastBody - 1);
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

            constexpr unsigned threadsPerInteraction = ClusterConfig::iSize * ClusterConfig::jSize;
            constexpr unsigned jBlocksPerWarp        = GpuConfig::warpSize / threadsPerInteraction;
            const GpuConfig::ThreadMask threadMask   = threadsPerInteraction == GpuConfig::warpSize
                                                           ? ~GpuConfig::ThreadMask(0)
                                                           : (GpuConfig::ThreadMask(1) << threadsPerInteraction) - 1;
            const GpuConfig::ThreadMask jBlockMask =
                threadMask << (threadsPerInteraction * (warp.thread_rank() / threadsPerInteraction));
            unsigned prunedNcc = 0;
            for (unsigned n = 0; n < imin(ncc, NcMax); n += jBlocksPerWarp)
            {
                const unsigned nb       = n + block.thread_index().y / ClusterConfig::jSize;
                const unsigned jCluster = nb < imin(ncc, NcMax) ? nidx[block.thread_index().z][c][nb] : ~0u;
                const unsigned j    = jCluster * ClusterConfig::jSize + block.thread_index().y % ClusterConfig::jSize;
                const Vec3<Tc> jPos = j < lastBody
                                          ? Vec3<Tc>{x[j], y[j], z[j]}
                                          : Vec3<Tc>{std::numeric_limits<Tc>::max(), std::numeric_limits<Tc>::max(),
                                                     std::numeric_limits<Tc>::max()};
                const Vec3<Tc> ijPosDiff = posDiff(jPos);
                const Th d2              = norm2(ijPosDiff);
                const Th hi2             = Th(2) * hi;
                const bool keep          = warp.ballot(d2 < hi2 * hi2) & jBlockMask;
                const unsigned offset    = exclusiveScanBool(keep & (warp.thread_rank() % threadsPerInteraction == 0));

                if ((warp.thread_rank() % threadsPerInteraction == 0) & keep)
                    nidx[block.thread_index().z][c][prunedNcc + offset] = jCluster;

                prunedNcc +=
                    warp.shfl(offset + keep, (GpuConfig::warpSize - 1) / (ClusterConfig::iSize * ClusterConfig::jSize) *
                                                 (ClusterConfig::iSize * ClusterConfig::jSize));
            }

            ncc = prunedNcc;

            constexpr unsigned long nbStoragePerICluster =
                Compress ? NcMax / ClusterConfig::expectedCompressionRate : NcMax;
            detail::deduplicateAndStoreNeighbors<warpsPerBlock, NcMax, Compress>(
                nidx[block.thread_index().z][c], ncc, &nidxClustered[ic * nbStoragePerICluster], &ncClustered[ic]);
        }
    }
}

template<int warpsPerBlock,
         bool UsePbc               = true,
         bool BypassL1CacheOnLoads = true,
         unsigned NcMax            = 256,
         bool Compress             = false,
         int Symmetric             = 0,
         class Tc,
         class Th,
         class Contribution,
         class... Tr>
__global__ __launch_bounds__(GpuConfig::warpSize* warpsPerBlock) void findNeighborsClustered(
    const cstone::LocalIndex firstBody,
    const cstone::LocalIndex lastBody,
    const Tc* const __restrict__ x,
    const Tc* const __restrict__ y,
    const Tc* const __restrict__ z,
    const Th* const __restrict__ h,
    const __grid_constant__ Box<Tc> box,
    const unsigned* const __restrict__ ncClustered,
    const unsigned* const __restrict__ nidxClustered,
    const Contribution contribution,
    Tr* const __restrict__... results)
{
    static_assert(NcMax % GpuConfig::warpSize == 0, "NcMax must be divisible by warp size");
    static_assert(Symmetric == 0 || Symmetric == 1 || Symmetric == -1, "Symmetric must be 0, 1, or -1");
    namespace cg = cooperative_groups;

    const auto block = cg::this_thread_block();
    assert(block.dim_threads().x == ClusterConfig::iSize);
    assert(block.dim_threads().y == GpuConfig::warpSize / ClusterConfig::iSize);
    static_assert(warpsPerBlock > 0 && ClusterConfig::iSize * ClusterConfig::jSize <= GpuConfig::warpSize);
    static_assert(GpuConfig::warpSize % (ClusterConfig::iSize * ClusterConfig::jSize) == 0);
    assert(block.dim_threads().z == warpsPerBlock);
    const auto warp = cg::tiled_partition<GpuConfig::warpSize>(block);

    const unsigned iCluster     = block.group_index().x * warpsPerBlock + block.thread_index().z;
    const unsigned numIClusters = iceil(lastBody, ClusterConfig::iSize);

    if (iCluster >= numIClusters) return;

    const unsigned i = imin(iCluster * ClusterConfig::iSize + block.thread_index().x, lastBody - 1);

    __shared__ unsigned nidxBuffer[warpsPerBlock][NcMax];
    unsigned* const nidx               = nidxBuffer[block.thread_index().z];
    constexpr unsigned compressedNcMax = Compress ? NcMax / ClusterConfig::expectedCompressionRate : NcMax;

    unsigned iClusterNeighborsCount;

    if constexpr (Compress)
    {
        warpDecompressNeighbors((const char*)&nidxClustered[iCluster * compressedNcMax], nidx, iClusterNeighborsCount);
    }
    else
    {
        iClusterNeighborsCount = imin(ncClustered[iCluster], NcMax);
#pragma unroll
        for (unsigned nb = warp.thread_rank(); nb < iClusterNeighborsCount; nb += GpuConfig::warpSize)
            nidx[nb] = nidxClustered[iCluster * compressedNcMax + nb];
    }

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

    std::tuple<Tr...> sums;
    detail::tupleForeach([](auto& sum) { sum = 0; }, sums);
    const auto computeClusterInteraction = [&](const unsigned jCluster, const bool self)
    {
        const unsigned j = jCluster * ClusterConfig::jSize + block.thread_index().y % ClusterConfig::jSize;
        std::tuple<Tr...> contrib;
        detail::tupleForeach([](auto& contrib) { contrib = 0; }, contrib);
        if (i < lastBody & j < lastBody & (!Symmetric | !self | (i <= j)))
        {
            const Vec3<Tc> jPos{x[j], y[j], z[j]};
            const Vec3<Tc> ijPosDiff = posDiff(jPos);
            const Th d2              = norm2(ijPosDiff);
            const Th hi2             = Th(2) * hi;
            if (d2 < hi2 * hi2)
                detail::tupleForeach([](auto& lhs, const auto& rhs) { lhs = rhs; }, contrib,
                                     contribution(i, iPos, hi, j, jPos, ijPosDiff, d2));
        }
        detail::tupleForeach([](auto& sum, auto const& contrib) { sum += contrib; }, sums, contrib);
        if constexpr (Symmetric)
        {
            if (i == j) detail::tupleForeach([&](auto& c) { c = 0; }, contrib);

            detail::storeTupleJSum<Symmetric>(contrib, std::make_tuple(&results[j]...), j < lastBody);
        }
    };

    constexpr unsigned jClustersPerWarp = GpuConfig::warpSize / ClusterConfig::iSize / ClusterConfig::jSize;
    constexpr unsigned overlappingJClusters =
        ClusterConfig::iSize <= ClusterConfig::jSize ? 1 : ClusterConfig::iSize / ClusterConfig::jSize;
#pragma unroll
    for (unsigned overlapping = 0; overlapping < overlappingJClusters; overlapping += jClustersPerWarp)
    {
        const unsigned o = overlapping + block.thread_index().y / ClusterConfig::jSize;
        const unsigned jCluster =
            o < overlappingJClusters ? iCluster * ClusterConfig::iSize / ClusterConfig::jSize + o : ~0u;
        computeClusterInteraction(jCluster, true);
    }
    for (unsigned jc = 0; jc < iClusterNeighborsCount; jc += jClustersPerWarp)
    {
        const unsigned jcc      = jc + block.thread_index().y / ClusterConfig::jSize;
        const unsigned jCluster = jcc < iClusterNeighborsCount ? nidx[jcc] : ~0u;
        computeClusterInteraction(jCluster, false);
    }

    detail::storeTupleISum<Symmetric>(sums, std::make_tuple(&results[i]...), i < lastBody);
}

} // namespace cstone
