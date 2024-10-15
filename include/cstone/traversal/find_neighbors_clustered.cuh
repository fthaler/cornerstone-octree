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
#include "cstone/traversal/find_neighbors.cuh"

#define CSTONE_USE_CUDA_PIPELINE 1

namespace cstone
{

struct ClusterConfig
{
    static constexpr unsigned iSize                   = 8;
    static constexpr unsigned jSize                   = 4;
    static constexpr unsigned expectedCompressionRate = 10;
};

namespace detail
{

template<std::size_t I, class... Ts>
__device__ inline auto& tuple_get_or_scalar(std::tuple<Ts...>& t)
{
    return std::get<I>(t);
}

template<std::size_t I, class... Ts>
__device__ inline auto const& tuple_get_or_scalar(std::tuple<Ts...> const& t)
{
    return std::get<I>(t);
}

template<std::size_t I, class T>
__device__ inline T const& tuple_get_or_scalar(T const& t)
{
    return t;
}

template<std::size_t I, class F, class... Tuples>
__device__ inline void tuple_foreach_element(F&& f, Tuples&&... tuples)
{
    f(tuple_get_or_scalar<I>(std::forward<Tuples>(tuples))...);
}

template<std::size_t... Is, class F, class... Tuples>
__device__ inline void tuple_foreach_impl(std::index_sequence<Is...>, F&& f, Tuples&&... tuples)
{
    (..., (tuple_foreach_element<Is>(std::forward<F>(f), std::forward<Tuples>(tuples)...)));
}

template<class F, class Tuple, class... Tuples>
__device__ inline void tuple_foreach(F&& f, Tuple&& tuple, Tuples&&... tuples)
{
    tuple_foreach_impl(std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>>>(), std::forward<F>(f),
                       std::forward<Tuple>(tuple), std::forward<Tuples>(tuples)...);
}

template<unsigned warpsPerBlock, unsigned NcMax, bool Compress>
__device__ inline void deduplicateAndStoreNeighbors(unsigned* iClusterNidx,
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
#pragma unroll
        for (unsigned i = 0; i < itemsPerWarp; ++i)
            items[i] = iClusterNidx[itemsPerWarp * warp.thread_rank() + i];

        constexpr unsigned long maxCompressedNeighborsSize = NcMax / ClusterConfig::expectedCompressionRate;
        warpCompressNeighbors<warpsPerBlock, itemsPerWarp>(items, (char*)targetIClusterNidx, uniqueNeighbors);
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

__device__ inline bool includeNbSymmetric(unsigned i, unsigned j)
{
    const bool s = i % 2 == j % 2;
    return i < j ? s : !s;
}

} // namespace detail

template<unsigned warpsPerBlock,
         bool UsePbc               = true,
         bool BypassL1CacheOnLoads = true,
         unsigned NcMax            = 256,
         bool Compress             = false,
         bool Symmetric            = false,
         class Tc,
         class Th,
         class KeyType>
__global__ __launch_bounds__(GpuConfig::warpSize* warpsPerBlock,
                             8) void findClusterNeighbors(cstone::LocalIndex firstBody,
                                                          cstone::LocalIndex lastBody,
                                                          const Tc* __restrict__ x,
                                                          const Tc* __restrict__ y,
                                                          const Tc* __restrict__ z,
                                                          const Th* __restrict__ h,
                                                          OctreeNsView<Tc, KeyType> tree,
                                                          const Box<Tc> box,
                                                          unsigned* __restrict__ ncClustered,
                                                          unsigned* __restrict__ nidxClustered,
                                                          int* globalPool)
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

#if CSTONE_USE_CUDA_PIPELINE
    alignas(16) __shared__ Tc xjSharedBuffer[warpsPerBlock][ClusterConfig::jSize];
    alignas(16) __shared__ Tc yjSharedBuffer[warpsPerBlock][ClusterConfig::jSize];
    alignas(16) __shared__ Tc zjSharedBuffer[warpsPerBlock][ClusterConfig::jSize];
    Tc* const xjShared = xjSharedBuffer[block.thread_index().z];
    Tc* const yjShared = yjSharedBuffer[block.thread_index().z];
    Tc* const zjShared = zjSharedBuffer[block.thread_index().z];
#endif

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

        Vec3<Tc> bbMin = {warpMin(iPos[0] - 2 * hi), warpMin(iPos[1] - 2 * hi), warpMin(iPos[2] - 2 * hi)};
        Vec3<Tc> bbMax = {warpMax(iPos[0] + 2 * hi), warpMax(iPos[1] + 2 * hi), warpMax(iPos[2] + 2 * hi)};

        const Vec3<Tc> targetCenter = (bbMax + bbMin) * Tc(0.5);
        const Vec3<Tc> targetSize   = (bbMax - bbMin) * Tc(0.5);

        const auto distSq = [&](const Vec3<Tc>& jPos)
        { return distanceSq<UsePbc>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box); };

        unsigned nc = 0;

        const auto checkNeighborhood = [&](const unsigned laneJCluster, const unsigned numLanesValid)
        {
            assert(numLanesValid > 0);
#if CSTONE_USE_CUDA_PIPELINE
            const auto thread = cg::this_thread();
            auto jPipeline    = cuda::make_pipeline();

            unsigned nextJCluster = warp.shfl(laneJCluster, 0);
            unsigned jCluster;

            const auto preloadNextJCluster = [&]
            {
                jPipeline.producer_acquire();
                if constexpr (BypassL1CacheOnLoads && sizeof(Tc) * ClusterConfig::jSize % 16 == 0)
                {
                    constexpr int numTcPer16Bytes = 16 / sizeof(Tc);
                    if (warp.thread_rank() < ClusterConfig::jSize / numTcPer16Bytes)
                    {
                        const unsigned nextJ = imin(
                            nextJCluster * ClusterConfig::jSize + warp.thread_rank() * numTcPer16Bytes, lastBody - 1);
                        cuda::memcpy_async(thread, &xjShared[warp.thread_rank() * numTcPer16Bytes], &x[nextJ],
                                           cuda::aligned_size_t<16>(16), jPipeline);
                        cuda::memcpy_async(thread, &yjShared[warp.thread_rank() * numTcPer16Bytes], &y[nextJ],
                                           cuda::aligned_size_t<16>(16), jPipeline);
                        cuda::memcpy_async(thread, &zjShared[warp.thread_rank() * numTcPer16Bytes], &z[nextJ],
                                           cuda::aligned_size_t<16>(16), jPipeline);
                    }
                }
                else
                {
                    if (warp.thread_rank() < ClusterConfig::jSize)
                    {
                        const unsigned nextJ =
                            imin(nextJCluster * ClusterConfig::jSize + warp.thread_rank(), lastBody - 1);
                        cuda::memcpy_async(thread, &xjShared[warp.thread_rank()], &x[nextJ],
                                           cuda::aligned_size_t<sizeof(Tc)>(sizeof(Tc)), jPipeline);
                        cuda::memcpy_async(thread, &yjShared[warp.thread_rank()], &y[nextJ],
                                           cuda::aligned_size_t<sizeof(Tc)>(sizeof(Tc)), jPipeline);
                        cuda::memcpy_async(thread, &zjShared[warp.thread_rank()], &z[nextJ],
                                           cuda::aligned_size_t<sizeof(Tc)>(sizeof(Tc)), jPipeline);
                    }
                }
                jPipeline.producer_commit();
            };
            preloadNextJCluster();
#endif

            for (unsigned n = 0; n < numLanesValid; ++n)
            {
#if CSTONE_USE_CUDA_PIPELINE
                jCluster = nextJCluster;

                Vec3<Tc> jPos[ClusterConfig::jSize];

                jPipeline.consumer_wait();
                warp.sync();
#pragma unroll
                for (unsigned jc = 0; jc < ClusterConfig::jSize; ++jc)
                    jPos[jc] = {xjShared[jc], yjShared[jc], zjShared[jc]};

                nextJCluster = warp.shfl(laneJCluster, imin(n + 1, numLanesValid - 1));
                jPipeline.consumer_release();

                if (nextJCluster != jCluster) preloadNextJCluster();
#else
                const unsigned jCluster = warp.shfl(laneJCluster, n);
#endif

                if (jCluster >= numJClusters) continue;

                bool isNeighbor = false;
                if (iCluster < numIClusters & iCluster * ClusterConfig::iSize / ClusterConfig::jSize != jCluster &
                    jCluster * ClusterConfig::jSize / ClusterConfig::iSize != iCluster &
                    (!Symmetric |
                     detail::includeNbSymmetric(iCluster, jCluster * ClusterConfig::jSize / ClusterConfig::iSize)))
                {
#pragma unroll
                    for (unsigned jc = 0; jc < ClusterConfig::jSize; ++jc)
                    {
#if CSTONE_USE_CUDA_PIPELINE
                        const Th d2 = distSq(jPos[jc]);
#else
                        const unsigned j    = imin(jCluster * ClusterConfig::jSize + jc, lastBody - 1);
                        const Vec3<Tc> jPos = {x[j], y[j], z[j]};
                        const Th d2         = distSq(jPos);
#endif
                        isNeighbor |= d2 < 4 * hi * hi;
                    }
                }

                using mask_t        = decltype(warp.ballot(false));
                mask_t iClusterMask = ((mask_t(1) << ClusterConfig::iSize) - 1)
                                      << (ClusterConfig::iSize * block.thread_index().y);
                bool newNeighbor = warp.ballot(isNeighbor) & iClusterMask;
                if (newNeighbor)
                {
                    if (nc < NcMax && block.thread_index().x == 0)
                        nidx[block.thread_index().z][block.thread_index().y][nc] = jCluster;
                    ++nc;
                }
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
            const unsigned ncc = warp.shfl(nc, c * ClusterConfig::iSize);
            const unsigned ic  = warp.shfl(iCluster, c * ClusterConfig::iSize);
            if (ic >= numIClusters) continue;

            constexpr unsigned long nbStoragePerICluster =
                Compress ? NcMax / ClusterConfig::expectedCompressionRate : NcMax;
            detail::deduplicateAndStoreNeighbors<warpsPerBlock, NcMax, Compress>(
                nidx[block.thread_index().z][c], ncc, &nidxClustered[ic * nbStoragePerICluster], &ncClustered[ic]);
        }
    }
}

template<int warpsPerBlock,
         bool BypassL1CacheOnLoads = true,
         unsigned NcMax            = 256,
         bool Compress             = false,
         int Symmetric             = 0,
         class Tc,
         class Th,
         class Contribution,
         class... Tr>
__global__ __launch_bounds__(GpuConfig::warpSize* warpsPerBlock,
                             8) void findNeighborsClustered(cstone::LocalIndex firstBody,
                                                            cstone::LocalIndex lastBody,
                                                            const Tc* __restrict__ x,
                                                            const Tc* __restrict__ y,
                                                            const Tc* __restrict__ z,
                                                            const Th* __restrict__ h,
                                                            const Box<Tc> box,
                                                            const unsigned* __restrict__ ncClustered,
                                                            const unsigned* __restrict__ nidxClustered,
                                                            Contribution contribution,
                                                            Tr* __restrict__... results)
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
    const auto warp   = cg::tiled_partition<GpuConfig::warpSize>(block);
    const auto thread = cg::this_thread();

    constexpr auto pbc = BoundaryType::periodic;
    const bool anyPbc  = box.boundaryX() == pbc | box.boundaryY() == pbc | box.boundaryZ() == pbc;

    assert(firstBody == 0); // TODO: other cases
    const unsigned numIClusters = iceil(lastBody - firstBody, ClusterConfig::iSize);

    unsigned* nidx = nullptr;
    if constexpr (Compress)
    {
        __shared__ unsigned nidxBuffer[warpsPerBlock][NcMax];
        nidx = nidxBuffer[block.thread_index().z];
    }

#if CSTONE_USE_CUDA_PIPELINE
    alignas(16) __shared__ Tc xiSharedBuffer[warpsPerBlock][ClusterConfig::iSize];
    alignas(16) __shared__ Tc yiSharedBuffer[warpsPerBlock][ClusterConfig::iSize];
    alignas(16) __shared__ Tc ziSharedBuffer[warpsPerBlock][ClusterConfig::iSize];
    alignas(16) __shared__ Th hiSharedBuffer[warpsPerBlock][ClusterConfig::iSize];
    Tc* const xiShared = xiSharedBuffer[block.thread_index().z];
    Tc* const yiShared = yiSharedBuffer[block.thread_index().z];
    Tc* const ziShared = ziSharedBuffer[block.thread_index().z];
    Th* const hiShared = hiSharedBuffer[block.thread_index().z];

    auto iPipeline = cuda::make_pipeline();

    unsigned iCluster = 0, nextICluster = 0;

    const auto preloadNextICluster = [&]
    {
        iPipeline.producer_acquire();
        if constexpr (BypassL1CacheOnLoads && sizeof(Tc) * ClusterConfig::iSize % 16 == 0 &&
                      sizeof(Th) * ClusterConfig::jSize % 16 == 0)
        {
            constexpr int numTcPer16Bytes = 16 / sizeof(Tc);
            constexpr int numThPer16Bytes = 16 / sizeof(Th);
            if (warp.thread_rank() < ClusterConfig::iSize / numTcPer16Bytes)
            {
                const unsigned nextI = nextICluster * ClusterConfig::iSize + warp.thread_rank() * numTcPer16Bytes;
                cuda::memcpy_async(thread, &xiShared[warp.thread_rank() * numTcPer16Bytes], &x[nextI],
                                   cuda::aligned_size_t<16>(16), iPipeline);
                cuda::memcpy_async(thread, &yiShared[warp.thread_rank() * numTcPer16Bytes], &y[nextI],
                                   cuda::aligned_size_t<16>(16), iPipeline);
                cuda::memcpy_async(thread, &ziShared[warp.thread_rank() * numTcPer16Bytes], &z[nextI],
                                   cuda::aligned_size_t<16>(16), iPipeline);
            }
            if (warp.thread_rank() < ClusterConfig::iSize / numThPer16Bytes)
            {
                const unsigned nextI = nextICluster * ClusterConfig::iSize + warp.thread_rank() * numThPer16Bytes;
                cuda::memcpy_async(thread, &hiShared[warp.thread_rank() * numThPer16Bytes], &h[nextI],
                                   cuda::aligned_size_t<16>(16), iPipeline);
            }
        }
        else
        {
            if (block.thread_index().y == 0)
            {
                const unsigned nextI = nextICluster * ClusterConfig::iSize + block.thread_index().x;
                cuda::memcpy_async(thread, &xiShared[block.thread_index().x], &x[nextI],
                                   cuda::aligned_size_t<sizeof(Tc)>(sizeof(Tc)), iPipeline);
                cuda::memcpy_async(thread, &yiShared[block.thread_index().x], &y[nextI],
                                   cuda::aligned_size_t<sizeof(Tc)>(sizeof(Tc)), iPipeline);
                cuda::memcpy_async(thread, &ziShared[block.thread_index().x], &z[nextI],
                                   cuda::aligned_size_t<sizeof(Tc)>(sizeof(Tc)), iPipeline);
                cuda::memcpy_async(thread, &hiShared[block.thread_index().x], &h[nextI],
                                   cuda::aligned_size_t<sizeof(Th)>(sizeof(Th)), iPipeline);
            }
        }
        iPipeline.producer_commit();
    };

    if (warp.thread_rank() == 0) nextICluster = atomicAdd(&targetCounterGlob, 1);
    nextICluster = warp.shfl(nextICluster, 0);
    if (nextICluster < numIClusters) preloadNextICluster();
#endif

    while (true)
    {
#if CSTONE_USE_CUDA_PIPELINE
        iCluster = nextICluster;
#else
        unsigned iCluster;
        if (warp.thread_rank() == 0) iCluster = atomicAdd(&targetCounterGlob, 1);
        iCluster = warp.shfl(iCluster, 0);
#endif

        if (iCluster >= numIClusters) return;

        const unsigned i = iCluster * ClusterConfig::iSize + block.thread_index().x;

#if CSTONE_USE_CUDA_PIPELINE
        iPipeline.consumer_wait();
        warp.sync();
        const Vec3<Tc> iPos = {xiShared[block.thread_index().x], yiShared[block.thread_index().x],
                               ziShared[block.thread_index().x]};
        const Th hi         = hiShared[block.thread_index().x];

        if (warp.thread_rank() == 0) nextICluster = atomicAdd(&targetCounterGlob, 1);
        nextICluster = warp.shfl(nextICluster, 0);
        iPipeline.consumer_release();
        if (nextICluster < numIClusters) preloadNextICluster();
#else
        const Vec3<Tc> iPos = {x[i], y[i], z[i]};
        const Th hi         = h[i];
#endif

        // const bool usePbc   = warp.any(anyPbc & !insideBox(iPos, {2 * hi, 2 * hi, 2 * hi}, box));

        const auto posDiff = [&](const Vec3<Tc>& jPos)
        {
            Vec3<Tc> ijPosDiff = {iPos[0] - jPos[0], iPos[1] - jPos[1], iPos[2] - jPos[2]};
            if (anyPbc)
            {
                ijPosDiff[0] -= (box.boundaryX() == pbc) * box.lx() * std::rint(ijPosDiff[0] * box.ilx());
                ijPosDiff[1] -= (box.boundaryY() == pbc) * box.ly() * std::rint(ijPosDiff[1] * box.ily());
                ijPosDiff[2] -= (box.boundaryZ() == pbc) * box.lz() * std::rint(ijPosDiff[2] * box.ilz());
            }
            return ijPosDiff;
        };

        std::tuple<Tr...> sums;
        detail::tuple_foreach([](auto& sum) { sum = 0; }, sums);
        const auto computeClusterInteraction = [&](const unsigned jCluster, const bool self)
        {
            const unsigned j = jCluster * ClusterConfig::jSize + block.thread_index().y % ClusterConfig::jSize;
            std::tuple<Tr...> contrib;
            detail::tuple_foreach([](auto& contrib) { contrib = 0; }, contrib);
            if (i < lastBody & j < lastBody & (!Symmetric | !self | (i <= j)))
            {
                const Vec3<Tc> jPos{x[j], y[j], z[j]};
                const Vec3<Tc> ijPosDiff = posDiff(jPos);
                const Th d2              = norm2(ijPosDiff);
                if (d2 < 4 * hi * hi)
                    detail::tuple_foreach([](auto& lhs, const auto& rhs) { lhs = rhs; }, contrib,
                                          contribution(i, iPos, hi, j, jPos, ijPosDiff, d2));
            }
            detail::tuple_foreach([](auto& sum, auto const& contrib) { sum += contrib; }, sums, contrib);
            if constexpr (Symmetric)
            {
                if (i == j) detail::tuple_foreach([&](auto& contrib) { contrib = 0; }, contrib);
#pragma unroll
                for (unsigned offset = ClusterConfig::iSize / 2; offset >= 1; offset /= 2)
                    detail::tuple_foreach([&](auto& contrib) { contrib += warp.shfl_down(contrib, offset); }, contrib);
                if (block.thread_index().x == 0 & j < lastBody)
                    detail::tuple_foreach(
                        [j](auto& res, auto const& sum)
                        {
                            if (sum != 0) atomicAdd(&res[j], Symmetric * sum);
                        },
                        std::make_tuple(results...), contrib);
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

        if constexpr (Compress)
        {
            constexpr unsigned long maxCompressedNeighborsSize = NcMax / ClusterConfig::expectedCompressionRate;
            unsigned iClusterNeighborsCount;
            warpDecompressNeighbors<warpsPerBlock, NcMax / GpuConfig::warpSize>(
                (const char*)&nidxClustered[iCluster * maxCompressedNeighborsSize], nidx, iClusterNeighborsCount);
            for (unsigned jc = 0; jc < imin(iClusterNeighborsCount, NcMax); jc += jClustersPerWarp)
            {
                const unsigned jcc      = jc + block.thread_index().y / ClusterConfig::jSize;
                const unsigned jCluster = jcc < iClusterNeighborsCount ? nidx[jcc] : ~0u;
                computeClusterInteraction(jCluster, false);
            }
        }
        else
        {
            const unsigned iClusterNeighborsCount = imin(ncClustered[iCluster], NcMax);
#pragma unroll ClusterConfig::jSize
            for (unsigned jc = 0; jc < imin(iClusterNeighborsCount, NcMax); jc += jClustersPerWarp)
            {
                const unsigned jcc = jc + block.thread_index().y / ClusterConfig::jSize;
                const unsigned jCluster =
                    jcc < iClusterNeighborsCount ? nidxClustered[(unsigned long)iCluster * NcMax + jcc] : ~0u;
                computeClusterInteraction(jCluster, false);
            }
        }

#pragma unroll
        for (unsigned offset = GpuConfig::warpSize / 2; offset >= ClusterConfig::iSize; offset /= 2)
            detail::tuple_foreach([&](auto& sum) { sum += warp.shfl_down(sum, offset); }, sums);

        if (block.thread_index().y == 0 & i < lastBody)
            detail::tuple_foreach(
                [i](auto& res, auto const& sum)
                {
                    if constexpr (Symmetric)
                    {
                        if (sum != 0) atomicAdd(&res[i], sum);
                    }
                    else { res[i] = sum; }
                },
                std::make_tuple(results...), sums);
    }
}

} // namespace cstone

#undef CSTONE_USE_CUDA_PIPELINE
