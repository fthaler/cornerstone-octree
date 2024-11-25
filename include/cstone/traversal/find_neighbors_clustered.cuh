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
__device__ inline auto& tupleGetOrScalar(std::tuple<Ts...>& t)
{
    return std::get<I>(t);
}

template<std::size_t I, class... Ts>
__device__ inline auto const& tupleGetOrScalar(std::tuple<Ts...> const& t)
{
    return std::get<I>(t);
}

template<std::size_t I, class T>
__device__ inline T const& tupleGetOrScalar(T const& t)
{
    return t;
}

template<std::size_t I, class F, class... Tuples>
__device__ inline void tupleForeachElement(F&& f, Tuples&&... tuples)
{
    f(tupleGetOrScalar<I>(std::forward<Tuples>(tuples))...);
}

template<std::size_t... Is, class F, class... Tuples>
__device__ inline void tupleForeachImpl(std::index_sequence<Is...>, F&& f, Tuples&&... tuples)
{
    (..., (tupleForeachElement<Is>(std::forward<F>(f), std::forward<Tuples>(tuples)...)));
}

template<class F, class Tuple, class... Tuples>
__device__ inline void tupleForeach(F&& f, Tuple&& tuple, Tuples&&... tuples)
{
    tupleForeachImpl(std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>>>(), std::forward<F>(f),
                     std::forward<Tuple>(tuple), std::forward<Tuples>(tuples)...);
}

template<int Symmetric, class... T>
__device__ inline void storeTupleJSum(std::tuple<T...>& tuple, std::tuple<T*...> const& ptrs, bool store)
{
    const auto block = cooperative_groups::this_thread_block();
    assert(block.dim_threads().x == ClusterConfig::iSize);
    const auto warp = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

#pragma unroll
    for (unsigned offset = ClusterConfig::iSize / 2; offset >= 1; offset /= 2)
        tupleForeach([&](auto& t) { t += warp.shfl_down(t, offset); }, tuple);

    if ((block.thread_index().x == 0) & store)
        detail::tupleForeach(
            [](auto* ptr, auto const& t)
            {
                if (t != 0) atomicAdd(ptr, Symmetric * t);
            },
            ptrs, tuple);
}

template<int Symmetric, class T>
__device__ inline void storeTupleJSum(std::tuple<T, T, T>& tuple, std::tuple<T*, T*, T*> const& ptrs, bool store)
{
    const auto block = cooperative_groups::this_thread_block();
    assert(block.dim_threads().x == ClusterConfig::iSize);
    const auto warp = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

    if constexpr (ClusterConfig::iSize == 8)
    {
        std::get<0>(tuple) += warp.shfl_down(std::get<0>(tuple), 1);
        std::get<1>(tuple) += warp.shfl_up(std::get<1>(tuple), 1);
        std::get<2>(tuple) += warp.shfl_down(std::get<2>(tuple), 1);

        if (block.thread_index().x & 1) std::get<0>(tuple) = std::get<1>(tuple);

        std::get<0>(tuple) += warp.shfl_down(std::get<0>(tuple), 2);
        std::get<2>(tuple) += warp.shfl_up(std::get<2>(tuple), 2);

        if (block.thread_index().x & 2) std::get<0>(tuple) = std::get<2>(tuple);

        std::get<0>(tuple) += warp.shfl_down(std::get<0>(tuple), 4);

        if ((block.thread_index().x < 3) & store)
        {
            T* ptr = block.thread_index().x == 0   ? std::get<0>(ptrs)
                     : block.thread_index().x == 1 ? std::get<1>(ptrs)
                                                   : std::get<2>(ptrs);
            if (std::get<0>(tuple) != 0) atomicAdd(ptr, Symmetric * std::get<0>(tuple));
        }
    }
    else if constexpr (ClusterConfig::iSize == 4)
    {
        std::get<0>(tuple) += warp.shfl_down(std::get<0>(tuple), 1);
        std::get<1>(tuple) += warp.shfl_up(std::get<1>(tuple), 1);
        std::get<2>(tuple) += warp.shfl_down(std::get<2>(tuple), 1);

        if (block.thread_index().x & 1) std::get<0>(tuple) = std::get<1>(tuple);

        std::get<0>(tuple) += warp.shfl_down(std::get<0>(tuple), 2);
        std::get<2>(tuple) += warp.shfl_up(std::get<2>(tuple), 2);

        if (block.thread_index().x & 2) std::get<0>(tuple) = std::get<2>(tuple);

        if ((block.thread_index().x < 3) & store)
        {
            T* ptr = block.thread_index().x == 0   ? std::get<0>(ptrs)
                     : block.thread_index().x == 1 ? std::get<1>(ptrs)
                                                   : std::get<2>(ptrs);
            if (std::get<0>(tuple) != 0) atomicAdd(ptr, Symmetric * std::get<0>(tuple));
        }
    }
    else { storeTupleJSum<Symmetric, T, T, T>(tuple, ptrs, store); }
}

template<bool Symmetric, class... T>
__device__ inline void storeTupleISum(std::tuple<T...>& tuple, std::tuple<T*...> const& ptrs, bool store)
{
    const auto block = cooperative_groups::this_thread_block();
    assert(block.dim_threads().x == ClusterConfig::iSize);
    const auto warp = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

#pragma unroll
    for (unsigned offset = GpuConfig::warpSize / 2; offset >= ClusterConfig::iSize; offset /= 2)
        detail::tupleForeach([&](auto& t) { t += warp.shfl_down(t, offset); }, tuple);

    if ((block.thread_index().y == 0) & store)
        detail::tupleForeach(
            [](auto* ptr, auto const& t)
            {
                if constexpr (Symmetric)
                {
                    if (t != 0) atomicAdd(ptr, t);
                }
                else { *ptr = t; }
            },
            ptrs, tuple);
}

template<bool Symmetric, class T>
__device__ inline void storeTupleISum(std::tuple<T, T, T>& tuple, std::tuple<T*, T*, T*> const& ptrs, bool store)
{
    const auto block = cooperative_groups::this_thread_block();
    assert(block.dim_threads().x == ClusterConfig::iSize);
    const auto warp = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

    if constexpr (GpuConfig::warpSize / ClusterConfig::iSize == 4)
    {
        std::get<0>(tuple) += warp.shfl_down(std::get<0>(tuple), ClusterConfig::iSize);
        std::get<1>(tuple) += warp.shfl_up(std::get<1>(tuple), ClusterConfig::iSize);
        std::get<2>(tuple) += warp.shfl_down(std::get<2>(tuple), ClusterConfig::iSize);

        if (block.thread_index().y & 1) std::get<0>(tuple) = std::get<1>(tuple);

        std::get<0>(tuple) += warp.shfl_down(std::get<0>(tuple), 2 * ClusterConfig::iSize);
        std::get<2>(tuple) += warp.shfl_up(std::get<2>(tuple), 2 * ClusterConfig::iSize);

        if (block.thread_index().y & 2) std::get<0>(tuple) = std::get<2>(tuple);

        if ((block.thread_index().y < 3) & store)
        {
            T* ptr = block.thread_index().y == 0   ? std::get<0>(ptrs)
                     : block.thread_index().y == 1 ? std::get<1>(ptrs)
                                                   : std::get<2>(ptrs);
            if constexpr (Symmetric)
            {
                if (std::get<0>(tuple) != 0) atomicAdd(ptr, Symmetric * std::get<0>(tuple));
            }
            else { *ptr = std::get<0>(tuple); }
        }
    }
    else if (GpuConfig::warpSize / ClusterConfig::iSize == 8)
    {
        std::get<0>(tuple) += warp.shfl_down(std::get<0>(tuple), ClusterConfig::iSize);
        std::get<1>(tuple) += warp.shfl_up(std::get<1>(tuple), ClusterConfig::iSize);
        std::get<2>(tuple) += warp.shfl_down(std::get<2>(tuple), ClusterConfig::iSize);

        if (block.thread_index().y & 1) std::get<0>(tuple) = std::get<1>(tuple);

        std::get<0>(tuple) += warp.shfl_down(std::get<0>(tuple), 2 * ClusterConfig::iSize);
        std::get<2>(tuple) += warp.shfl_up(std::get<2>(tuple), 2 * ClusterConfig::iSize);

        if (block.thread_index().y & 2) std::get<0>(tuple) = std::get<2>(tuple);

        std::get<0>(tuple) += warp.shfl_down(std::get<0>(tuple), 4 * ClusterConfig::iSize);

        if ((block.thread_index().y < 3) & store)
        {
            T* ptr = block.thread_index().y == 0   ? std::get<0>(ptrs)
                     : block.thread_index().y == 1 ? std::get<1>(ptrs)
                                                   : std::get<2>(ptrs);
            if constexpr (Symmetric)
            {
                if (std::get<0>(tuple) != 0) atomicAdd(ptr, Symmetric * std::get<0>(tuple));
            }
            else { *ptr = std::get<0>(tuple); }
        }
    }
    else { storeTupleISum<Symmetric, T, T, T>(tuple, ptrs, store); }
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

constexpr inline bool includeNbSymmetric(unsigned i, unsigned j)
{
    const bool s = i % 2 == j % 2;
    return i < j ? s : !s;
}

template<unsigned warpsPerBlock, bool BypassL1CacheOnLoads, class T, unsigned N>
struct alignas(16) Preloader
{
    T buffer[warpsPerBlock][N];

    inline __device__ void
    startLoad(const T* __restrict__ ptr, cuda::pipeline<cuda::thread_scope_thread>& pipeline, unsigned n = N)
    {
        namespace cg  = cooperative_groups;
        auto block    = cg::this_thread_block();
        auto warp     = cg::tiled_partition<GpuConfig::warpSize>(block);
        auto thread   = cg::this_thread();
        T* warpBuffer = buffer[warp.meta_group_rank()];
        if constexpr (BypassL1CacheOnLoads && (sizeof(T) * N) % 16 == 0)
        {
            constexpr unsigned numTPer16Bytes = 16 / sizeof(T);
#pragma unroll
            for (unsigned i = warp.thread_rank(); i < N / numTPer16Bytes; i += GpuConfig::warpSize)
            {
                if (i < (n + numTPer16Bytes - 1) / numTPer16Bytes)
                {
                    cuda::memcpy_async(thread, &warpBuffer[i * numTPer16Bytes], &ptr[i * numTPer16Bytes],
                                       cuda::aligned_size_t<16>(16), pipeline);
                }
            }
        }
        else
        {
#pragma unroll
            for (unsigned i = warp.thread_rank(); i < N; i += GpuConfig::warpSize)
            {
                if (i < n)
                {
                    cuda::memcpy_async(thread, &warpBuffer[i], &ptr[i], cuda::aligned_size_t<sizeof(T)>(sizeof(T)),
                                       pipeline);
                }
            }
        }
    }

    inline __device__ T& operator[](unsigned index)
    {
        namespace cg = cooperative_groups;
        auto block   = cg::this_thread_block();
        auto warp    = cg::tiled_partition<GpuConfig::warpSize>(block);
        return buffer[warp.meta_group_rank()][index];
    }
};

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
                                                          const __grid_constant__ Box<Tc> box,
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
    __shared__ detail::Preloader<warpsPerBlock, BypassL1CacheOnLoads, Tc, ClusterConfig::jSize> xjPreloader,
        yjPreloader, zjPreloader;
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
                if (nextJCluster < numJClusters)
                {
                    xjPreloader.startLoad(&x[nextJCluster * ClusterConfig::jSize], jPipeline);
                    yjPreloader.startLoad(&y[nextJCluster * ClusterConfig::jSize], jPipeline);
                    zjPreloader.startLoad(&z[nextJCluster * ClusterConfig::jSize], jPipeline);
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
                    jPos[jc] = {xjPreloader[jc], yjPreloader[jc], zjPreloader[jc]};

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
                        const unsigned j = imin(jCluster * ClusterConfig::jSize + jc, lastBody - 1);
                        const Vec3<Tc> jPos = {x[j], y[j], z[j]};
                        const Th d2 = distSq(jPos);
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
    cstone::LocalIndex firstBody,
    cstone::LocalIndex lastBody,
    const Tc* __restrict__ x,
    const Tc* __restrict__ y,
    const Tc* __restrict__ z,
    const Th* __restrict__ h,
    const __grid_constant__ Box<Tc> box,
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

    const unsigned iCluster = block.group_index().x * warpsPerBlock + block.thread_index().z;

    if (iCluster > iceil(lastBody, ClusterConfig::iSize)) return;

    const unsigned i = iCluster * ClusterConfig::iSize + block.thread_index().x;

    __shared__ unsigned nidxBuffer[warpsPerBlock][NcMax];
    unsigned* const nidx               = nidxBuffer[block.thread_index().z];
    constexpr unsigned compressedNcMax = Compress ? NcMax / ClusterConfig::expectedCompressionRate : NcMax;

    unsigned iClusterNeighborsCount = Compress ? compressedNcMax : imin(ncClustered[iCluster], NcMax);

#pragma unroll
    for (unsigned nb = warp.thread_rank(); nb < iClusterNeighborsCount; nb += GpuConfig::warpSize)
        nidx[nb] = nidxClustered[iCluster * compressedNcMax + nb];

    if constexpr (Compress)
    {
        warp.sync();
        warpDecompressNeighbors<warpsPerBlock, NcMax / GpuConfig::warpSize>((char*)nidx, nidx, iClusterNeighborsCount);
    }

    const Vec3<Tc> iPos = {x[i], y[i], z[i]};
    const Th hi         = h[i];

    const auto posDiff = [&](const Vec3<Tc>& jPos)
    {
        Vec3<Tc> ijPosDiff = {iPos[0] - jPos[0], iPos[1] - jPos[1], iPos[2] - jPos[2]};
        if constexpr (UsePbc)
        {
            ijPosDiff[0] -= (box.boundaryX() == pbc) * box.lx() * std::rint(ijPosDiff[0] * box.ilx());
            ijPosDiff[1] -= (box.boundaryY() == pbc) * box.ly() * std::rint(ijPosDiff[1] * box.ily());
            ijPosDiff[2] -= (box.boundaryZ() == pbc) * box.lz() * std::rint(ijPosDiff[2] * box.ilz());
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

#undef CSTONE_USE_CUDA_PIPELINE
