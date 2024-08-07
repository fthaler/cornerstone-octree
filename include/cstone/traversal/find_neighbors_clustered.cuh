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

#include <cuda/barrier>
#include <cuda/pipeline>
#include <cooperative_groups.h>

#include "cstone/compressneighbors.hpp"
#include "cstone/primitives/warpscan.cuh"
#include "cstone/traversal/find_neighbors.cuh"

#define CSTONE_USE_CUDA_PIPELINE 1

namespace cstone
{

struct ClusterConfig
{
    static constexpr unsigned iSize = 8;
    static constexpr unsigned jSize = 4;
};

__host__ __device__ inline constexpr unsigned clusterNeighborIndex(unsigned cluster, unsigned neighbor, unsigned ncmax)
{
    // constexpr unsigned blockSize = TravConfig::targetSize / ClusterConfig::iSize;
    constexpr unsigned blockSize = 1; // better for findNeighborsClustered3
    return (cluster / blockSize) * blockSize * ncmax + (cluster % blockSize) + neighbor * blockSize;
}

template<class Tc, class Th, class KeyType>
__global__ __launch_bounds__(TravConfig::numThreads) void findClusterNeighbors(cstone::LocalIndex firstBody,
                                                                               cstone::LocalIndex lastBody,
                                                                               const Tc* __restrict__ x,
                                                                               const Tc* __restrict__ y,
                                                                               const Tc* __restrict__ z,
                                                                               const Th* __restrict__ h,
                                                                               OctreeNsView<Tc, KeyType> tree,
                                                                               const Box<Tc> box,
                                                                               unsigned* __restrict__ ncClustered,
                                                                               unsigned* __restrict__ nidxClustered,
                                                                               unsigned ncmax,
                                                                               int* globalPool)
{
    const unsigned laneIdx    = threadIdx.x & (GpuConfig::warpSize - 1);
    const unsigned numTargets = (lastBody - firstBody - 1) / TravConfig::targetSize + 1;
    int targetIdx             = 0;

    constexpr unsigned iClustersPerWarp = GpuConfig::warpSize / ClusterConfig::iSize;
    constexpr unsigned targetsPerBlock  = TravConfig::numThreads / TravConfig::targetSize;

    __shared__ unsigned ncData[targetsPerBlock][TravConfig::nwt][iClustersPerWarp];
    __shared__ unsigned nidxData[targetsPerBlock][iClustersPerWarp][512 /* TODO: ncmax */][TravConfig::nwt];
    assert(ncmax == 512);

    const unsigned targetIdxLocal = threadIdx.x / TravConfig::targetSize;
    auto nc                       = [&](unsigned iClusterWarp, unsigned warpTarget) -> unsigned&
    { return ncData[targetIdxLocal][warpTarget][iClusterWarp]; };
    auto nidx = [&](unsigned iClusterWarp, unsigned warpTarget, unsigned nb) -> unsigned&
    { return nidxData[targetIdxLocal][iClusterWarp][nb][warpTarget]; };

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&targetCounterGlob, 1); }
        targetIdx = shflSync(targetIdx, 0);

        if (targetIdx >= numTargets) return;

        const cstone::LocalIndex bodyBegin = firstBody + targetIdx * TravConfig::targetSize;
        const cstone::LocalIndex bodyEnd   = imin(bodyBegin + TravConfig::targetSize, lastBody);
        const unsigned iClusterWarp        = laneIdx / ClusterConfig::iSize;
        const unsigned i                   = imin(bodyBegin + laneIdx, bodyEnd - 1);

        if (laneIdx < iClustersPerWarp)
        {
#pragma unroll
            for (int warpTarget = 0; warpTarget < TravConfig::nwt; ++warpTarget)
                nc(laneIdx, warpTarget) = 0;
        }

        __syncwarp();

        auto handleInteraction = [&](int warpTarget, cstone::LocalIndex j)
        {
            const unsigned jCluster = j / ClusterConfig::jSize;
            if (i / ClusterConfig::jSize == jCluster || j / ClusterConfig::iSize == i / ClusterConfig::iSize) return;
            const unsigned iClusterMask = ((1 << ClusterConfig::iSize) - 1)
                                          << (laneIdx / ClusterConfig::iSize * ClusterConfig::iSize);
            const unsigned leader = __ffs(__activemask() & iClusterMask) - 1;

            if (leader != laneIdx) return;

            const unsigned ncc = imin(nc(iClusterWarp, warpTarget), ncmax);
            for (unsigned nb = 0; nb < ncc; ++nb)
            {
                if (nidx(iClusterWarp, warpTarget, ncc - 1 - nb) == jCluster) return;
            }
            const unsigned idx = nc(iClusterWarp, warpTarget)++;
            if (idx < ncmax) nidx(iClusterWarp, warpTarget, idx) = jCluster;
        };

        traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, handleInteraction, globalPool);

        __syncwarp();

#pragma unroll
        for (unsigned warpTarget = 0; warpTarget < TravConfig::nwt; ++warpTarget)
        {
            const unsigned nbs = nc(laneIdx % iClustersPerWarp, warpTarget);
            if (laneIdx < TravConfig::targetSize / ClusterConfig::iSize)
                ncClustered[(bodyBegin + warpTarget * GpuConfig::warpSize) / ClusterConfig::iSize + laneIdx] = nbs;

            const unsigned iCluster =
                (bodyBegin + warpTarget * GpuConfig::warpSize) / ClusterConfig::iSize + laneIdx % iClustersPerWarp;
            const unsigned iClusterWarp = laneIdx % iClustersPerWarp;
            for (unsigned nb = laneIdx / iClustersPerWarp; nb < imin(nbs, ncmax); nb += iClustersPerWarp)
            {
                nidxClustered[clusterNeighborIndex(iCluster, nb, ncmax)] = nidx(iClusterWarp, warpTarget, nb);
            }
        }
    }
}

template<class Tc, class Th, class KeyType>
__global__ __launch_bounds__(TravConfig::numThreads) void findClusterNeighbors2(cstone::LocalIndex firstBody,
                                                                                cstone::LocalIndex lastBody,
                                                                                const Tc* __restrict__ x,
                                                                                const Tc* __restrict__ y,
                                                                                const Tc* __restrict__ z,
                                                                                const Th* __restrict__ h,
                                                                                OctreeNsView<Tc, KeyType> tree,
                                                                                const Box<Tc> box,
                                                                                unsigned* __restrict__ ncClustered,
                                                                                unsigned* __restrict__ nidxClustered,
                                                                                unsigned ncmax,
                                                                                int* globalPool)
{
    const unsigned laneIdx    = threadIdx.x & (GpuConfig::warpSize - 1);
    const unsigned numTargets = (lastBody - firstBody - 1) / TravConfig::targetSize + 1;
    int targetIdx             = 0;

    constexpr unsigned iClustersPerWarp = GpuConfig::warpSize / ClusterConfig::iSize;

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&targetCounterGlob, 1); }
        targetIdx = shflSync(targetIdx, 0);

        if (targetIdx >= numTargets) return;

        const cstone::LocalIndex bodyBegin = firstBody + targetIdx * TravConfig::targetSize;
        const cstone::LocalIndex bodyEnd   = imin(bodyBegin + TravConfig::targetSize, lastBody);
        const unsigned iClusterWarp        = laneIdx / ClusterConfig::iSize;
        const unsigned i                   = imin(bodyBegin + laneIdx, bodyEnd - 1);

        auto nc = [&](unsigned iClusterWarp, unsigned warpTarget) -> unsigned&
        { return ncClustered[(bodyBegin + warpTarget * GpuConfig::warpSize) / ClusterConfig::iSize + iClusterWarp]; };

        auto nidx = [&](unsigned iClusterWarp, unsigned warpTarget, unsigned nb) -> unsigned&
        {
            unsigned iCluster = (bodyBegin + warpTarget * GpuConfig::warpSize) / ClusterConfig::iSize + iClusterWarp;
            return nidxClustered[clusterNeighborIndex(iCluster, nb, ncmax)];
        };

        if (laneIdx < iClustersPerWarp)
        {
            for (int warpTarget = 0; warpTarget < TravConfig::nwt; ++warpTarget)
                nc(laneIdx, warpTarget) = 0;
        }

        __syncwarp();

        auto handleInteraction = [&](int warpTarget, cstone::LocalIndex j)
        {
            if (i / ClusterConfig::iSize == j / ClusterConfig::iSize ||
                i / ClusterConfig::jSize == j / ClusterConfig::jSize)
                return;

            unsigned jCluster = j / ClusterConfig::jSize;
            unsigned mask =
                (((1 << ClusterConfig::iSize) - 1) << (laneIdx / ClusterConfig::iSize * ClusterConfig::iSize)) &
                __activemask();
            unsigned leader = __ffs(mask) - 1;

            if (laneIdx != leader) return;

            const unsigned ncc = imin(nc(iClusterWarp, warpTarget), ncmax);

            unsigned nb = 0;
            if constexpr (ClusterConfig::jSize == 1)
            {
                // no deduplication required for ClusterConfig::jSize == 1
                nb = ncc;
            }
            else
            {
                // with ClusterConfig::jSize != we have to deduplicate
                if (ncc > 0 && nidx(iClusterWarp, warpTarget, ncc - 1) == jCluster) return;

                unsigned last  = ncc;
                unsigned count = last - nb;
                while (count > 0)
                {
                    unsigned step   = count / 2;
                    unsigned center = nb + step;

                    if (!(jCluster < nidx(iClusterWarp, warpTarget, center)))
                    {
                        nb = center + 1;
                        count -= step + 1;
                    }
                    else { count = step; }
                }

                if (nb > 0 && nidx(iClusterWarp, warpTarget, nb - 1) == jCluster) return;

                for (unsigned nbi = imin(ncc, ncmax - 1); nbi > nb; --nbi)
                    nidx(iClusterWarp, warpTarget, nbi) = nidx(iClusterWarp, warpTarget, nbi - 1);
            }

            ++nc(iClusterWarp, warpTarget);
            if (nb < ncmax) nidx(iClusterWarp, warpTarget, nb) = jCluster;
        };

        traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, handleInteraction, globalPool);
    }
}

template<class Tc, class Th, class KeyType>
__global__ __launch_bounds__(TravConfig::numThreads,
                             16) void findClusterNeighbors3(cstone::LocalIndex firstBody,
                                                            cstone::LocalIndex lastBody,
                                                            const Tc* __restrict__ x,
                                                            const Tc* __restrict__ y,
                                                            const Tc* __restrict__ z,
                                                            const Th* __restrict__ h,
                                                            OctreeNsView<Tc, KeyType> tree,
                                                            const Box<Tc> box,
                                                            unsigned* __restrict__ ncClustered,
                                                            unsigned* __restrict__ nidxClustered,
                                                            unsigned ncmax,
                                                            int* globalPool)
{
    const unsigned laneIdx    = threadIdx.x & (GpuConfig::warpSize - 1);
    const unsigned warpIdx    = threadIdx.x >> GpuConfig::warpSizeLog2;
    const unsigned numTargets = (lastBody - firstBody - 1) / TravConfig::targetSize + 1;
    int targetIdx             = 0;

    constexpr unsigned iClustersPerWarp = GpuConfig::warpSize / ClusterConfig::iSize;

    __shared__ unsigned tmp[TravConfig::numThreads / GpuConfig::warpSize][512 /* TODO: ncmax */];
    assert(ncmax == 512);

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&targetCounterGlob, 1); }
        targetIdx = shflSync(targetIdx, 0);

        if (targetIdx >= numTargets) return;

        const cstone::LocalIndex bodyBegin = firstBody + targetIdx * TravConfig::targetSize;
        const cstone::LocalIndex bodyEnd   = imin(bodyBegin + TravConfig::targetSize, lastBody);
        const unsigned iClusterWarp        = laneIdx / ClusterConfig::iSize;
        const unsigned i                   = imin(bodyBegin + laneIdx, bodyEnd - 1);

        auto nc = [&](unsigned iClusterWarp, unsigned warpTarget) -> unsigned&
        { return ncClustered[(bodyBegin + warpTarget * GpuConfig::warpSize) / ClusterConfig::iSize + iClusterWarp]; };

        auto nidx = [&](unsigned iClusterWarp, unsigned warpTarget, unsigned nb) -> unsigned&
        {
            unsigned iCluster = (bodyBegin + warpTarget * GpuConfig::warpSize) / ClusterConfig::iSize + iClusterWarp;
            return nidxClustered[clusterNeighborIndex(iCluster, nb, ncmax)];
        };

        if (laneIdx < iClustersPerWarp)
        {
            for (int warpTarget = 0; warpTarget < TravConfig::nwt; ++warpTarget)
                nc(laneIdx, warpTarget) = 0;
        }

        __syncwarp();

        auto handleInteraction = [&](int warpTarget, cstone::LocalIndex j)
        {
            if (i / ClusterConfig::iSize == j / ClusterConfig::iSize ||
                i / ClusterConfig::jSize == j / ClusterConfig::jSize || bodyBegin + laneIdx >= bodyEnd)
                return;

            unsigned mask =
                (((1 << ClusterConfig::iSize) - 1) << (laneIdx / ClusterConfig::iSize * ClusterConfig::iSize)) &
                __activemask();
            unsigned leader = __ffs(mask) - 1;

            if (laneIdx != leader) return;

            unsigned idx = nc(iClusterWarp, warpTarget)++;
            if (idx < ncmax) nidx(iClusterWarp, warpTarget, idx) = j / ClusterConfig::jSize;
        };

        traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, handleInteraction, globalPool);

        __syncwarp();

        for (unsigned warpTarget = 0; warpTarget < TravConfig::nwt; ++warpTarget)
        {
            for (unsigned iClusterWarp = 0; iClusterWarp < iClustersPerWarp; ++iClusterWarp)
            {
                unsigned iCluster =
                    (bodyBegin + warpTarget * GpuConfig::warpSize) / ClusterConfig::iSize + iClusterWarp;
                if (iCluster > (lastBody - 1) / ClusterConfig::iSize) continue;

                unsigned nbs;
                if (laneIdx == 0) nbs = imin(nc(iClusterWarp, warpTarget), ncmax);
                nbs            = shflSync(nbs, 0);
                unsigned nbsp2 = 1;
                while (nbsp2 < nbs)
                    nbsp2 *= 2;
                assert(nbsp2 <= ncmax);
                unsigned nbsWarp = iceil(nbs, GpuConfig::warpSize) * GpuConfig::warpSize;
                for (unsigned nb = laneIdx; nb < nbsp2; nb += GpuConfig::warpSize)
                    tmp[warpIdx][nb] = nb < nbs ? nidx(iClusterWarp, warpTarget, nb) : unsigned(-1);

                __syncwarp();

                for (unsigned k = 2; k <= nbsp2; k *= 2)
                {
                    for (unsigned j = k / 2; j > 0; j /= 2)
                    {
                        for (unsigned i = laneIdx; i < nbsp2; i += GpuConfig::warpSize)
                        {
                            unsigned ij = i ^ j;
                            if (ij > i && ((i & k) == 0 && tmp[warpIdx][i] > tmp[warpIdx][ij] ||
                                           (i & k) != 0 && tmp[warpIdx][i] < tmp[warpIdx][ij]))
                            {
                                unsigned t       = tmp[warpIdx][i];
                                tmp[warpIdx][i]  = tmp[warpIdx][ij];
                                tmp[warpIdx][ij] = t;
                            }
                        }
                        __syncwarp();
                    }
                }

                unsigned start = 0, previous = ~0u;
                for (unsigned nb = laneIdx; nb < nbsWarp; nb += GpuConfig::warpSize)
                {
                    unsigned current = tmp[warpIdx][imin(nb, ncmax - 1)];
                    unsigned left    = shflUpSync(current, 1);
                    if (laneIdx == 0) left = previous;
                    unsigned keep  = nb < nbs && current != left;
                    unsigned index = inclusiveSegscan(keep, laneIdx) - keep + start;
                    if (keep) nidx(iClusterWarp, warpTarget, index) = current;
                    start    = shflSync(index + keep, GpuConfig::warpSize - 1);
                    previous = shflSync(current, GpuConfig::warpSize - 1);
                }
                if (laneIdx == 0) nc(iClusterWarp, warpTarget) = start;
            }
        }
    }
}

template<bool UsePbc = true, class Tc, class Th, class KeyType>
__global__ void findClusterNeighbors4(cstone::LocalIndex firstBody,
                                      cstone::LocalIndex lastBody,
                                      const Tc* __restrict__ x,
                                      const Tc* __restrict__ y,
                                      const Tc* __restrict__ z,
                                      const Th* __restrict__ h,
                                      OctreeNsView<Tc, KeyType> tree,
                                      const Box<Tc> box,
                                      unsigned* __restrict__ ncClustered,
                                      unsigned* __restrict__ nidxClustered,
                                      unsigned ncmax,
                                      int* globalPool)
{
    namespace cg = cooperative_groups;

    const auto grid  = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto warp  = cg::tiled_partition<GpuConfig::warpSize>(block);

    constexpr unsigned targetSize = ClusterConfig::iSize * GpuConfig::warpSize;
    const unsigned numTargets     = iceil(lastBody, targetSize);
    const unsigned numIClusters   = iceil(lastBody, ClusterConfig::iSize);
    const unsigned numJClusters   = iceil(lastBody, ClusterConfig::jSize);

    volatile __shared__ int sharedPool[TravConfig::numThreads];

    __shared__ Tc xiShared[ClusterConfig::iSize][TravConfig::numThreads];
    __shared__ Tc yiShared[ClusterConfig::iSize][TravConfig::numThreads];
    __shared__ Tc ziShared[ClusterConfig::iSize][TravConfig::numThreads];
    __shared__ Th hiShared[ClusterConfig::iSize][TravConfig::numThreads];

    const TreeNodeIndex* __restrict__ childOffsets   = tree.childOffsets;
    const TreeNodeIndex* __restrict__ internalToLeaf = tree.internalToLeaf;
    const LocalIndex* __restrict__ layout            = tree.layout;
    const Vec3<Tc>* __restrict__ centers             = tree.centers;
    const Vec3<Tc>* __restrict__ sizes               = tree.sizes;

    const auto distSq = [&](const Vec3<Tc>& iPos, const Vec3<Tc>& jPos)
    { return distanceSq<UsePbc>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box); };

    while (true)
    {
        unsigned target;
        if (warp.thread_rank() == 0) target = atomicAdd(&targetCounterGlob, 1);
        target = warp.shfl(target, 0);

        if (target >= numTargets) return;

        const unsigned iCluster = target * GpuConfig::warpSize + warp.thread_rank();
#pragma unroll
        for (unsigned ic = 0; ic < ClusterConfig::iSize; ++ic)
        {
            const unsigned i                  = imin(iCluster * ClusterConfig::iSize + ic, lastBody - 1);
            xiShared[ic][block.thread_rank()] = x[i];
            yiShared[ic][block.thread_rank()] = y[i];
            ziShared[ic][block.thread_rank()] = z[i];
            hiShared[ic][block.thread_rank()] = h[i];
        }
        if (iCluster < numIClusters) ncClustered[iCluster] = 0;
        warp.sync();

        Vec3<Tc> bbMin = {std::numeric_limits<Tc>::max(), std::numeric_limits<Tc>::max(),
                          std::numeric_limits<Tc>::max()};
        Vec3<Tc> bbMax = {std::numeric_limits<Tc>::min(), std::numeric_limits<Tc>::min(),
                          std::numeric_limits<Tc>::min()};
#pragma unroll
        for (unsigned i = 0; i < ClusterConfig::iSize; ++i)
        {
            const Tc xi = xiShared[i][block.thread_rank()];
            const Tc yi = yiShared[i][block.thread_rank()];
            const Tc zi = ziShared[i][block.thread_rank()];
            const Tc hi = hiShared[i][block.thread_rank()];
            bbMin = {std::min(bbMin[0], xi - 2 * hi), std::min(bbMin[1], yi - 2 * hi), std::min(bbMin[2], zi - 2 * hi)};
            bbMax = {std::max(bbMax[0], xi + 2 * hi), std::max(bbMax[1], yi + 2 * hi), std::max(bbMax[2], zi + 2 * hi)};
        }
        const Vec3<Tc> iClusterCenter = (bbMax + bbMin) * Tc(0.5);
        const Vec3<Tc> iClusterSize   = (bbMax - bbMin) * Tc(0.5);

        bbMin = {warpMin(bbMin[0]), warpMin(bbMin[1]), warpMin(bbMin[2])};
        bbMax = {warpMax(bbMax[0]), warpMax(bbMax[1]), warpMax(bbMax[2])};

        const Vec3<Tc> targetCenter = (bbMax + bbMin) * Tc(0.5);
        const Vec3<Tc> targetSize   = (bbMax - bbMin) * Tc(0.5);

        auto checkNeighborhood = [&](const unsigned laneJCluster, const unsigned numLanesValid)
        {
            for (unsigned n = 0; n < numLanesValid; ++n)
            {
                const unsigned jCluster = warp.shfl(laneJCluster, n);
                if (iCluster < numIClusters & jCluster < numJClusters &
                    iCluster * ClusterConfig::iSize / ClusterConfig::jSize != jCluster &
                    jCluster * ClusterConfig::jSize / ClusterConfig::iSize != iCluster)
                {
#pragma unroll
                    for (unsigned cj = 0; cj < ClusterConfig::jSize; ++cj)
                    {
                        const unsigned j    = imin(jCluster * ClusterConfig::jSize + cj, lastBody - 1);
                        const Vec3<Tc> jPos = {x[j], y[j], z[j]};
#pragma unroll
                        for (unsigned ci = 0; ci < ClusterConfig::iSize; ++ci)
                        {
                            const unsigned i = imin(iCluster * ClusterConfig::iSize + ci, lastBody - 1);
                            // const Vec3<Tc> iPos = {x[i], y[i], z[i]};
                            // const Th hi = h[i];
                            const Vec3<Tc> iPos = {xiShared[ci][block.thread_rank()], yiShared[ci][block.thread_rank()],
                                                   ziShared[ci][block.thread_rank()]};
                            const Th hi         = hiShared[ci][block.thread_rank()];
                            const Th d2         = distSq(iPos, jPos);
                            if (d2 < 4 * hi * hi)
                            {
                                const unsigned nc = imin(ncClustered[iCluster], ncmax);
                                bool alreadyIn    = false;
#pragma unroll 4
                                for (unsigned nb = 0; nb < nc; ++nb)
                                {
                                    if (nidxClustered[clusterNeighborIndex(iCluster, nb, ncmax)] == jCluster)
                                        alreadyIn = true;
                                }
                                if (!alreadyIn)
                                {
                                    ++ncClustered[iCluster];
                                    if (nc < ncmax) nidxClustered[clusterNeighborIndex(iCluster, nc, ncmax)] = jCluster;
                                }
                                goto breakout;
                            }
                        }
                    }
                breakout:;
                }
            }
        };

        unsigned maxStack = 0;

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
            maxStack                 = max(stackUsed, maxStack);
            if (stackUsed > TravConfig::memPerWarp) return; // Exit if cellQueue overflows

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
                        // Load source body coordinates
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
        {
            const bool laneHasJCluster = warp.thread_rank() < jClusterFillLevel;
            checkNeighborhood(jClusterQueue, jClusterFillLevel);
        }
        warp.sync();
    }
}

template<unsigned warpsPerBlock, bool UsePbc = true, class Tc, class Th, class KeyType>
__global__ __launch_bounds__(ClusterConfig::iSize* ClusterConfig::jSize* warpsPerBlock) void findClusterNeighbors5(
    cstone::LocalIndex firstBody,
    cstone::LocalIndex lastBody,
    const Tc* __restrict__ x,
    const Tc* __restrict__ y,
    const Tc* __restrict__ z,
    const Th* __restrict__ h,
    OctreeNsView<Tc, KeyType> tree,
    const Box<Tc> box,
    unsigned* __restrict__ ncClustered,
    unsigned* __restrict__ nidxClustered,
    unsigned ncmax,
    int* globalPool)
{
    namespace cg = cooperative_groups;

    const auto grid  = cg::this_grid();
    const auto block = cg::this_thread_block();
    assert(block.dim_threads().x == ClusterConfig::iSize);
    assert(block.dim_threads().y == ClusterConfig::jSize);
    assert(block.dim_threads().z == warpsPerBlock);
    static_assert(warpsPerBlock > 0 && ClusterConfig::iSize * ClusterConfig::jSize == GpuConfig::warpSize);
    const auto warp = cg::tiled_partition<GpuConfig::warpSize>(block);

    volatile __shared__ int sharedPool[GpuConfig::warpSize * warpsPerBlock];

    assert(firstBody == 0); // TODO: other cases
    const unsigned numIClusters = iceil(lastBody, ClusterConfig::iSize);
    const unsigned numJClusters = iceil(lastBody, ClusterConfig::jSize);

    const TreeNodeIndex* __restrict__ childOffsets   = tree.childOffsets;
    const TreeNodeIndex* __restrict__ internalToLeaf = tree.internalToLeaf;
    const LocalIndex* __restrict__ layout            = tree.layout;
    const Vec3<Tc>* __restrict__ centers             = tree.centers;
    const Vec3<Tc>* __restrict__ sizes               = tree.sizes;

    while (true)
    {
        unsigned iCluster;
        if (warp.thread_rank() == 0) iCluster = atomicAdd(&targetCounterGlob, 1);
        iCluster = warp.shfl(iCluster, 0);

        if (iCluster >= numIClusters) return;

        const unsigned i    = imin(iCluster * ClusterConfig::iSize + block.thread_index().x, lastBody - 1);
        const Vec3<Tc> iPos = {x[i], y[i], z[i]};
        const Th hi         = h[i];

        Vec3<Tc> bbMin = {iPos[0] - 2 * hi, iPos[1] - 2 * hi, iPos[2] - 2 * hi};
        Vec3<Tc> bbMax = {iPos[0] + 2 * hi, iPos[1] + 2 * hi, iPos[2] + 2 * hi};
#pragma unroll
        for (unsigned offset = ClusterConfig::iSize / 2; offset >= 1; offset /= 2)
        {
            bbMin = {std::min(bbMin[0], warp.shfl_down(bbMin[0], offset)),
                     std::min(bbMin[1], warp.shfl_down(bbMin[1], offset)),
                     std::min(bbMin[2], warp.shfl_down(bbMin[2], offset))};
            bbMax = {std::max(bbMax[0], warp.shfl_down(bbMax[0], offset)),
                     std::max(bbMax[1], warp.shfl_down(bbMax[1], offset)),
                     std::max(bbMax[2], warp.shfl_down(bbMax[2], offset))};
        }
        bbMin = warp.shfl(bbMin, 0);
        bbMax = warp.shfl(bbMax, 0);

        const Vec3<Tc> iClusterCenter = (bbMax + bbMin) * Tc(0.5);
        const Vec3<Tc> iClusterSize   = (bbMax - bbMin) * Tc(0.5);

        const auto distSq = [&](const Vec3<Tc>& jPos)
        { return distanceSq<UsePbc>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box); };

        unsigned nc = 0;

        auto checkNeighborhood = [&](const unsigned laneJCluster, const unsigned numLanesValid)
        {
#pragma unroll 4
            for (unsigned n = 0; n < numLanesValid; ++n)
            {
                const unsigned jCluster = warp.shfl(laneJCluster, n);
                if (iCluster < numIClusters & jCluster < numJClusters &
                    iCluster * ClusterConfig::iSize / ClusterConfig::jSize != jCluster &
                    jCluster * ClusterConfig::jSize / ClusterConfig::iSize != iCluster)
                {
                    const unsigned j    = imin(jCluster * ClusterConfig::jSize + block.thread_index().y, lastBody - 1);
                    const Vec3<Tc> jPos = {x[j], y[j], z[j]};
                    const Th d2         = distSq(jPos);
                    if (warp.any(d2 < 4 * hi * hi))
                    {
                        bool alreadyIn     = false;
                        unsigned ncLimited = imin(nc, ncmax);
                        for (unsigned nb = 0; nb < ncLimited; nb += GpuConfig::warpSize)
                        {
                            alreadyIn = warp.any(nidxClustered[clusterNeighborIndex(
                                                     iCluster, imin(nb + warp.thread_rank(), ncLimited - 1), ncmax)] ==
                                                 jCluster);
                            if (alreadyIn) break;
                        }
                        if (!alreadyIn)
                        {
                            if (nc < ncmax && warp.thread_rank() == 0)
                                nidxClustered[clusterNeighborIndex(iCluster, nc, ncmax)] = jCluster;
                            ++nc;
                        }
                    }
                }
            }
        };

        unsigned maxStack = 0;

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
            const bool isClose  = cellOverlap<UsePbc>(curSrcCenter, curSrcSize, iClusterCenter, iClusterSize, box);
            const bool isSource = sourceIdx < numSources; // Source index is within bounds
            const bool isDirect = isClose && !isNode && isSource;
            const int leafIdx   = isDirect ? internalToLeaf[sourceQueue] : 0; // the cstone leaf index

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
            maxStack                 = max(stackUsed, maxStack);
            if (stackUsed > TravConfig::memPerWarp) return; // Exit if cellQueue overflows

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

        if (warp.thread_rank() == 0) ncClustered[iCluster] = nc;
    }
}

template<unsigned warpsPerBlock, bool UsePbc = true, class Tc, class Th, class KeyType>
__global__
__launch_bounds__(GpuConfig::warpSize* warpsPerBlock) void findClusterNeighbors6(cstone::LocalIndex firstBody,
                                                                                 cstone::LocalIndex lastBody,
                                                                                 const Tc* __restrict__ x,
                                                                                 const Tc* __restrict__ y,
                                                                                 const Tc* __restrict__ z,
                                                                                 const Th* __restrict__ h,
                                                                                 OctreeNsView<Tc, KeyType> tree,
                                                                                 const Box<Tc> box,
                                                                                 unsigned* __restrict__ ncClustered,
                                                                                 unsigned* __restrict__ nidxClustered,
                                                                                 unsigned ncmax,
                                                                                 int* globalPool)
{
    namespace cg = cooperative_groups;

    const auto grid  = cg::this_grid();
    const auto block = cg::this_thread_block();
    assert(block.dim_threads().x == ClusterConfig::iSize);
    assert(block.dim_threads().y == GpuConfig::warpSize / ClusterConfig::iSize);
    assert(block.dim_threads().z == warpsPerBlock);
    static_assert(warpsPerBlock > 0 && ClusterConfig::iSize * ClusterConfig::jSize == GpuConfig::warpSize);
    const auto warp = cg::tiled_partition<GpuConfig::warpSize>(block);

    volatile __shared__ int sharedPool[GpuConfig::warpSize * warpsPerBlock];

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

        const unsigned i    = imin(iCluster * ClusterConfig::iSize + block.thread_index().x, lastBody - 1);
        const Vec3<Tc> iPos = {x[i], y[i], z[i]};
        const Th hi         = h[i];

        Vec3<Tc> bbMin = {warpMin(iPos[0] - 2 * hi), warpMin(iPos[1] - 2 * hi), warpMin(iPos[2] - 2 * hi)};
        Vec3<Tc> bbMax = {warpMax(iPos[0] + 2 * hi), warpMax(iPos[1] + 2 * hi), warpMax(iPos[2] + 2 * hi)};

        const Vec3<Tc> targetCenter = (bbMax + bbMin) * Tc(0.5);
        const Vec3<Tc> targetSize   = (bbMax - bbMin) * Tc(0.5);

        const auto distSq = [&](const Vec3<Tc>& jPos)
        { return distanceSq<UsePbc>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box); };

        unsigned nc = 0;

        auto checkNeighborhood = [&](const unsigned laneJCluster, const unsigned numLanesValid)
        {
#pragma unroll 2
            for (unsigned n = 0; n < numLanesValid; ++n)
            {
                const unsigned jCluster = warp.shfl(laneJCluster, n);
                if (jCluster >= numJClusters) continue;

                bool isNeighbor = false;
                if (iCluster < numIClusters & iCluster * ClusterConfig::iSize / ClusterConfig::jSize != jCluster &
                    jCluster * ClusterConfig::jSize / ClusterConfig::iSize != iCluster)
                {
#pragma unroll
                    for (unsigned jc = 0; jc < ClusterConfig::jSize; ++jc)
                    {
                        const unsigned j    = imin(jCluster * ClusterConfig::jSize + jc, lastBody - 1);
                        const Vec3<Tc> jPos = {x[j], y[j], z[j]};
                        const Th d2         = distSq(jPos);
                        isNeighbor |= d2 < 4 * hi * hi;
                    }
                }

                using mask_t        = decltype(warp.ballot(false));
                mask_t iClusterMask = ((mask_t(1) << ClusterConfig::iSize) - 1)
                                      << (ClusterConfig::iSize * block.thread_index().y);
                bool newNeighbor = warp.ballot(isNeighbor) & iClusterMask;
                if (newNeighbor)
                {
                    unsigned ncLimited = imin(nc, ncmax);
#pragma unroll 4
                    for (unsigned nb = 0; nb < ncLimited; nb += ClusterConfig::iSize)
                    {
                        newNeighbor &=
                            nidxClustered[clusterNeighborIndex(
                                iCluster, imin(nb + block.thread_index().x, ncLimited - 1), ncmax)] != jCluster;
                    }
                }
                if (!(warp.ballot(!newNeighbor) & iClusterMask))
                {
                    if (nc < ncmax && block.thread_index().x == 0)
                        nidxClustered[clusterNeighborIndex(iCluster, nc, ncmax)] = jCluster;
                    ++nc;
                }
            }
        };

        unsigned maxStack = 0;

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
            maxStack                 = max(stackUsed, maxStack);
            if (stackUsed > TravConfig::memPerWarp) return; // Exit if cellQueue overflows

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

        if (block.thread_index().x == 0) ncClustered[iCluster] = nc;
    }
}

__global__ void compressClusterNeighbors(const unsigned iClusters,
                                         const unsigned* __restrict__ ncClustered,
                                         unsigned* __restrict__ nidxClustered,
                                         const unsigned ncmax)
{
    constexpr unsigned safetyMargin = 128;
    unsigned buffer[safetyMargin];
    for (unsigned iCluster = blockIdx.x * blockDim.x + threadIdx.x; iCluster < iClusters;
         iCluster += blockDim.x * gridDim.x)
    {
        const unsigned nc = imin(ncClustered[iCluster], ncmax);
        for (unsigned i = 0; i < imin(safetyMargin, nc); ++i)
            buffer[i] = nidxClustered[clusterNeighborIndex(iCluster, i, ncmax)];

        NeighborListCompressor comp(&nidxClustered[clusterNeighborIndex(iCluster, 0, ncmax)], sizeof(unsigned) * ncmax);

        for (unsigned i = 0; i < imin(safetyMargin, nc); ++i)
            comp.push_back(buffer[i]);
        for (unsigned i = safetyMargin; i < nc; ++i)
            comp.push_back(nidxClustered[clusterNeighborIndex(iCluster, i, ncmax)]);
    }
}

template<class Tc, class Th, class Contribution, class Tr>
__global__
__launch_bounds__(TravConfig::numThreads) void findNeighborsClustered(const LocalIndex firstBody,
                                                                      const LocalIndex lastBody,
                                                                      const Tc* __restrict__ x,
                                                                      const Tc* __restrict__ y,
                                                                      const Tc* __restrict__ z,
                                                                      const Th* __restrict__ h,
                                                                      const Box<Tc> box,
                                                                      const unsigned* __restrict__ ncClustered,
                                                                      const unsigned* __restrict__ nidxClustered,
                                                                      unsigned ncmax,
                                                                      Contribution contribution,
                                                                      Tr* __restrict__ result)
{
    const unsigned laneIdx    = threadIdx.x & (GpuConfig::warpSize - 1);
    const unsigned numTargets = (lastBody - firstBody - 1) / TravConfig::targetSize + 1;
    int targetIdx             = 0;

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&targetCounterGlob, 1); }
        targetIdx = shflSync(targetIdx, 0);

        if (targetIdx >= numTargets) return;

        const cstone::LocalIndex bodyBegin   = firstBody + targetIdx * TravConfig::targetSize;
        const cstone::LocalIndex bodyIdxLane = bodyBegin + laneIdx;

#pragma unroll
        for (int k = 0; k < TravConfig::nwt; ++k)
        {
            Tr sum = 0;

            const unsigned i = bodyIdxLane + k * GpuConfig::warpSize;
            if (i >= lastBody) continue;

            const Vec3<Tc> iPos{x[i], y[i], z[i]};
            const Th hi = h[i];

            const unsigned iCluster = i / ClusterConfig::iSize;

#pragma unroll
            for (unsigned jCluster = iCluster * ClusterConfig::iSize / ClusterConfig::jSize;
                 jCluster <
                 (iCluster * ClusterConfig::iSize +
                  (ClusterConfig::iSize > ClusterConfig::jSize ? ClusterConfig::iSize : ClusterConfig::jSize)) /
                     ClusterConfig::jSize;
                 ++jCluster)
            {
#pragma unroll
                for (unsigned j = jCluster * ClusterConfig::jSize; j < (jCluster + 1) * ClusterConfig::jSize; ++j)
                {
                    if (ClusterConfig::jSize == 1 || j < lastBody)
                    {

                        const Vec3<Tc> jPos = {x[j], y[j], z[j]};
                        const Th dist = distancePBC(box, hi, iPos[0], iPos[1], iPos[2], jPos[0], jPos[1], jPos[2]);
                        if (ClusterConfig::iSize == 1 && ClusterConfig::jSize == 1 || dist < 2 * hi)
                            sum += contribution(i, iPos, hi, j, jPos, dist);
                    }
                }
            }

            const unsigned iClusterNeighborsCount = imin(ncClustered[iCluster], ncmax);
            for (unsigned jc = 0; jc < iClusterNeighborsCount; ++jc)
            {
                const unsigned jCluster = nidxClustered[clusterNeighborIndex(iCluster, jc, ncmax)];
#pragma unroll
                for (unsigned j = jCluster * ClusterConfig::jSize; j < (jCluster + 1) * ClusterConfig::jSize; ++j)
                {
                    if (ClusterConfig::jSize == 1 || j < lastBody)
                    {
                        const Vec3<Tc> jPos = {x[j], y[j], z[j]};
                        const Th dist = distancePBC(box, hi, iPos[0], iPos[1], iPos[2], jPos[0], jPos[1], jPos[2]);
                        if (ClusterConfig::iSize == 1 && ClusterConfig::jSize == 1 || dist < 2 * hi)
                            sum += contribution(i, iPos, hi, j, jPos, dist);
                    }
                }
            }

            result[i] = sum;
        }
    }
}

template<class Tc, class Th, class Contribution, class Tr>
__global__
__launch_bounds__(TravConfig::numThreads) void findNeighborsClustered2(cstone::LocalIndex firstBody,
                                                                       cstone::LocalIndex lastBody,
                                                                       const Tc* __restrict__ x,
                                                                       const Tc* __restrict__ y,
                                                                       const Tc* __restrict__ z,
                                                                       const Th* __restrict__ h,
                                                                       const Box<Tc> box,
                                                                       const unsigned* __restrict__ ncClustered,
                                                                       const unsigned* __restrict__ nidxClustered,
                                                                       unsigned ncmax,
                                                                       Contribution contribution,
                                                                       Tr* __restrict__ result)
{
    const unsigned laneIdx    = threadIdx.x & (GpuConfig::warpSize - 1);
    const unsigned warpIdx    = threadIdx.x >> GpuConfig::warpSizeLog2;
    const unsigned numTargets = (lastBody - firstBody - 1) / TravConfig::targetSize + 1;
    int targetIdx             = 0;

    constexpr auto pbc = BoundaryType::periodic;
    const bool anyPbc  = box.boundaryX() == pbc || box.boundaryY() == pbc || box.boundaryZ() == pbc;

    constexpr unsigned warpsPerBlock = TravConfig::numThreads / GpuConfig::warpSize;
    __shared__ Tc xSuperCluster[warpsPerBlock][GpuConfig::warpSize];
    __shared__ Tc ySuperCluster[warpsPerBlock][GpuConfig::warpSize];
    __shared__ Tc zSuperCluster[warpsPerBlock][GpuConfig::warpSize];
    __shared__ Th hSuperCluster[warpsPerBlock][GpuConfig::warpSize];
    __shared__ Tr resultSuperCluster[warpsPerBlock][GpuConfig::warpSize];

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&targetCounterGlob, 1); }
        targetIdx = shflSync(targetIdx, 0);

        if (targetIdx >= numTargets) return;

        static_assert(TravConfig::targetSize == GpuConfig::warpSize, "Requires targetSize == warpSize");
        static_assert(ClusterConfig::iSize * ClusterConfig::jSize == GpuConfig::warpSize,
                      "Single warp required per cluster-cluster interaction");
        constexpr unsigned clustersPerWarp = GpuConfig::warpSize / ClusterConfig::iSize;

        const unsigned iSuperCluster        = firstBody + GpuConfig::warpSize * targetIdx + laneIdx;
        const unsigned iSuperClusterClamped = imin(iSuperCluster, lastBody - 1);
        xSuperCluster[warpIdx][laneIdx]     = x[iSuperClusterClamped];
        ySuperCluster[warpIdx][laneIdx]     = y[iSuperClusterClamped];
        zSuperCluster[warpIdx][laneIdx]     = z[iSuperClusterClamped];
        hSuperCluster[warpIdx][laneIdx]     = h[iSuperClusterClamped];
        __syncwarp();

#pragma unroll
        for (unsigned c = 0; c < clustersPerWarp; ++c)
        {
            const unsigned i = (iSuperCluster / GpuConfig::warpSize) * GpuConfig::warpSize +
                               laneIdx % ClusterConfig::iSize + c * ClusterConfig::iSize;

            Vec3<Tc> iPos;
            Th hi;
            if (laneIdx < ClusterConfig::iSize)
            {
                const unsigned iSuperClusterLocal = laneIdx + c * ClusterConfig::iSize;
                iPos = {xSuperCluster[warpIdx][iSuperClusterLocal], ySuperCluster[warpIdx][iSuperClusterLocal],
                        zSuperCluster[warpIdx][iSuperClusterLocal]};
                hi   = hSuperCluster[warpIdx][iSuperClusterLocal];
            }
            const unsigned srcLane = laneIdx % ClusterConfig::iSize;
            iPos = {shflSync(iPos[0], srcLane), shflSync(iPos[1], srcLane), shflSync(iPos[2], srcLane)};
            hi   = shflSync(hi, srcLane);

            Tr sum              = 0;
            const auto iCluster = imin(i, lastBody - 1) / ClusterConfig::iSize;

            auto distSq = [&](const Vec3<Tc>& jPos)
            {
                const bool usePbc = anyPbc && !insideBox(iPos, {2 * hi, 2 * hi, 2 * hi}, box);
                return true ? distanceSq<true>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box)
                            : distanceSq<false>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box);
            };
            const auto radiusSq = 4 * hi * hi;

            for (unsigned jCluster = iCluster * ClusterConfig::iSize / ClusterConfig::jSize;
                 jCluster <
                 (iCluster * ClusterConfig::iSize +
                  (ClusterConfig::iSize > ClusterConfig::jSize ? ClusterConfig::iSize : ClusterConfig::jSize)) /
                     ClusterConfig::jSize;
                 ++jCluster)
            {
                const unsigned j = jCluster * ClusterConfig::jSize + laneIdx / ClusterConfig::iSize;
                if (i < lastBody & j < lastBody)
                {
                    const Vec3<Tc> jPos{x[j], y[j], z[j]};
                    const Th d2 = distSq(jPos);
                    if (d2 < radiusSq) sum += contribution(i, iPos, hi, j, jPos, std::sqrt(d2));
                }
            }

            const unsigned iClusterNeighborsCount = imin(ncClustered[iCluster], ncmax);
            for (unsigned jc = 0; jc < iClusterNeighborsCount; ++jc)
            {
                const unsigned jCluster = nidxClustered[clusterNeighborIndex(iCluster, jc, ncmax)];
                const unsigned j        = jCluster * ClusterConfig::jSize + laneIdx / ClusterConfig::iSize;
                if (i < lastBody & j < lastBody & j / ClusterConfig::iSize != iCluster)
                {
                    const Vec3<Tc> jPos{x[j], y[j], z[j]};
                    const Th d2 = distSq(jPos);
                    if (d2 < radiusSq) sum += contribution(i, iPos, hi, j, jPos, std::sqrt(d2));
                }
            }

#pragma unroll
            for (unsigned offset = GpuConfig::warpSize / 2; offset >= ClusterConfig::iSize; offset /= 2)
                sum += shflDownSync(sum, offset);

            sum = shflSync(sum, laneIdx % ClusterConfig::iSize);
            if (laneIdx / ClusterConfig::iSize == c) resultSuperCluster[warpIdx][laneIdx] = sum;
        }

        if (iSuperCluster < lastBody) result[iSuperCluster] = resultSuperCluster[warpIdx][laneIdx];
    }
}

template<class Tc, class Th, class Contribution, class Tr>
__global__
__launch_bounds__(TravConfig::numThreads) void findNeighborsClustered3(cstone::LocalIndex firstBody,
                                                                       cstone::LocalIndex lastBody,
                                                                       const Tc* __restrict__ x,
                                                                       const Tc* __restrict__ y,
                                                                       const Tc* __restrict__ z,
                                                                       const Th* __restrict__ h,
                                                                       const Box<Tc> box,
                                                                       const unsigned* __restrict__ ncClustered,
                                                                       const unsigned* __restrict__ nidxClustered,
                                                                       unsigned ncmax,
                                                                       Contribution contribution,
                                                                       Tr* __restrict__ result)
{
    namespace cg = cooperative_groups;

    auto warp = cg::tiled_partition<GpuConfig::warpSize>(cg::this_thread_block());

    const unsigned numTargets = (lastBody - firstBody - 1) / TravConfig::targetSize + 1;
    int targetIdx             = 0;

    constexpr auto pbc = BoundaryType::periodic;
    const bool anyPbc  = box.boundaryX() == pbc || box.boundaryY() == pbc || box.boundaryZ() == pbc;

    while (true)
    {
        // first thread in warp grabs next target
        if (warp.thread_rank() == 0) { targetIdx = atomicAdd(&targetCounterGlob, 1); }
        targetIdx = shflSync(targetIdx, 0);

        if (targetIdx >= numTargets) return;

        static_assert(TravConfig::targetSize == GpuConfig::warpSize, "Requires targetSize == warpSize");
        static_assert(ClusterConfig::iSize * ClusterConfig::jSize == GpuConfig::warpSize,
                      "Single warp required per cluster-cluster interaction");
        constexpr unsigned clustersPerWarp = GpuConfig::warpSize / ClusterConfig::iSize;
        const unsigned iSuperCluster       = firstBody + GpuConfig::warpSize * targetIdx + warp.thread_rank();

#pragma unroll
        for (unsigned c = 0; c < clustersPerWarp; ++c)
        {
            const unsigned i = (iSuperCluster / GpuConfig::warpSize) * GpuConfig::warpSize +
                               warp.thread_rank() % ClusterConfig::iSize + c * ClusterConfig::iSize;

            const Vec3<Tc> iPos = {x[i], y[i], z[i]};
            const Th hi         = h[i];
            const bool usePbc   = warp.any(anyPbc && !insideBox(iPos, {2 * hi, 2 * hi, 2 * hi}, box));

            Tr sum              = 0;
            const auto iCluster = imin(i, lastBody - 1) / ClusterConfig::iSize;

            auto distSq = [&](const Vec3<Tc>& jPos)
            {
                return usePbc ? distanceSq<true>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box)
                              : distanceSq<false>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box);
            };
            const auto radiusSq = 4 * hi * hi;

#pragma unroll
            for (unsigned jCluster = iCluster * ClusterConfig::iSize / ClusterConfig::jSize;
                 jCluster <
                 (iCluster * ClusterConfig::iSize +
                  (ClusterConfig::iSize > ClusterConfig::jSize ? ClusterConfig::iSize : ClusterConfig::jSize)) /
                     ClusterConfig::jSize;
                 ++jCluster)
            {
                const unsigned j = jCluster * ClusterConfig::jSize + warp.thread_rank() / ClusterConfig::iSize;
                if (i < lastBody & j < lastBody)
                {
                    const Vec3<Tc> jPos{x[j], y[j], z[j]};
                    const Th d2 = distSq(jPos);
                    if (d2 < radiusSq) sum += contribution(i, iPos, hi, j, jPos, std::sqrt(d2));
                }
            }

            const unsigned iClusterNeighborsCount = imin(ncClustered[iCluster], ncmax);
            for (unsigned jc = 0; jc < iClusterNeighborsCount; ++jc)
            {
                const unsigned jCluster = nidxClustered[clusterNeighborIndex(iCluster, jc, ncmax)];
                const unsigned j        = jCluster * ClusterConfig::jSize + warp.thread_rank() / ClusterConfig::iSize;
                if (i < lastBody & j < lastBody & j / ClusterConfig::iSize != iCluster)
                {
                    const Vec3<Tc> jPos{x[j], y[j], z[j]};
                    const Th d2 = distSq(jPos);
                    if (d2 < radiusSq) sum += contribution(i, iPos, hi, j, jPos, std::sqrt(d2));
                }
            }

#pragma unroll
            for (unsigned offset = GpuConfig::warpSize / 2; offset >= ClusterConfig::iSize; offset /= 2)
                sum += warp.shfl_down(sum, offset);

            if (warp.thread_rank() < ClusterConfig::iSize) result[iSuperCluster + c * ClusterConfig::iSize] = sum;
        }
    }
}

template<class Tc, class Th, class Contribution, class Tr>
__global__
__launch_bounds__(ClusterConfig::iSize* ClusterConfig::jSize* GpuConfig::warpSize /
                  ClusterConfig::iSize) void findNeighborsClustered4(cstone::LocalIndex firstBody,
                                                                     cstone::LocalIndex lastBody,
                                                                     const Tc* __restrict__ x,
                                                                     const Tc* __restrict__ y,
                                                                     const Tc* __restrict__ z,
                                                                     const Th* __restrict__ h,
                                                                     const Box<Tc> box,
                                                                     const unsigned* __restrict__ ncClustered,
                                                                     const unsigned* __restrict__ nidxClustered,
                                                                     unsigned ncmax,
                                                                     Contribution contribution,
                                                                     Tr* __restrict__ result)
{
    namespace cg = cooperative_groups;

    const auto block = cg::this_thread_block();
    assert(block.dim_threads().x == ClusterConfig::iSize);
    assert(block.dim_threads().y == ClusterConfig::jSize);
    assert(block.dim_threads().z == GpuConfig::warpSize / ClusterConfig::iSize);
    const auto warp = cg::tiled_partition<GpuConfig::warpSize>(block);

    const unsigned numTargets = iceil(lastBody - firstBody, GpuConfig::warpSize);
    __shared__ int sharedTargetIdx;

    constexpr auto pbc = BoundaryType::periodic;

    auto token = block.barrier_arrive();

    while (true)
    {
        // first thread in block grabs next target
        if (block.thread_rank() == 0) sharedTargetIdx = atomicAdd(&targetCounterGlob, 1);
        block.barrier_wait(std::move(token));
        const unsigned targetIdx = sharedTargetIdx;
        token                    = block.barrier_arrive();

        if (targetIdx >= numTargets) return;

        static_assert(ClusterConfig::iSize * ClusterConfig::jSize == GpuConfig::warpSize,
                      "Single warp required per cluster-cluster interaction");

        const unsigned i =
            targetIdx * GpuConfig::warpSize + block.thread_index().x + block.thread_index().z * ClusterConfig::iSize;

        const Vec3<Tc> iPos = {x[i], y[i], z[i]};
        const Th hi         = h[i];
        // const bool usePbc   = warp.any(anyPbc & !insideBox(iPos, {2 * hi, 2 * hi, 2 * hi}, box));

        const unsigned iCluster = imin(i, lastBody - 1) / ClusterConfig::iSize;

        auto distSq = [&](const Vec3<Tc>& jPos)
        {
            const bool anyPbc = box.boundaryX() == pbc | box.boundaryY() == pbc | box.boundaryZ() == pbc;
            return anyPbc ? distanceSq<true>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box)
                          : distanceSq<false>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box);
        };

        Tr sum = 0;
#pragma unroll
        for (unsigned jCluster = iCluster * ClusterConfig::iSize / ClusterConfig::jSize;
             jCluster < (iCluster * ClusterConfig::iSize +
                         (ClusterConfig::iSize > ClusterConfig::jSize ? ClusterConfig::iSize : ClusterConfig::jSize)) /
                            ClusterConfig::jSize;
             ++jCluster)
        {
            const unsigned j = jCluster * ClusterConfig::jSize + block.thread_index().y;
            if (i < lastBody & j < lastBody)
            {
                const Vec3<Tc> jPos{x[j], y[j], z[j]};
                const Th d2 = distSq(jPos);
                if (d2 < 4 * hi * hi) sum += contribution(i, iPos, hi, j, jPos, std::sqrt(d2));
            }
        }

        const unsigned iClusterNeighborsCount = imin(ncClustered[iCluster], ncmax);
#pragma unroll ClusterConfig::jSize
        for (unsigned jc = 0; jc < iClusterNeighborsCount; ++jc)
        {
            const unsigned jCluster = nidxClustered[clusterNeighborIndex(iCluster, jc, ncmax)];
            const unsigned j        = jCluster * ClusterConfig::jSize + block.thread_index().y;
            if (i < lastBody & j < lastBody & j / ClusterConfig::iSize != iCluster)
            {
                const Vec3<Tc> jPos{x[j], y[j], z[j]};
                const Th d2 = distSq(jPos);
                if (d2 < 4 * hi * hi) sum += contribution(i, iPos, hi, j, jPos, std::sqrt(d2));
            }
        }

#pragma unroll
        for (unsigned offset = GpuConfig::warpSize / 2; offset >= ClusterConfig::iSize; offset /= 2)
            sum += warp.shfl_down(sum, offset);

        if (block.thread_index().y == 0) result[i] = sum;
    }
}

template<class Tc, class Th, class Contribution, class Tr>
__global__
__launch_bounds__(ClusterConfig::iSize* ClusterConfig::jSize* GpuConfig::warpSize /
                  ClusterConfig::iSize) void findNeighborsClustered5(cstone::LocalIndex firstBody,
                                                                     cstone::LocalIndex lastBody,
                                                                     const Tc* __restrict__ x,
                                                                     const Tc* __restrict__ y,
                                                                     const Tc* __restrict__ z,
                                                                     const Th* __restrict__ h,
                                                                     const Box<Tc> box,
                                                                     const unsigned* __restrict__ ncClustered,
                                                                     const unsigned* __restrict__ nidxClustered,
                                                                     unsigned ncmax,
                                                                     Contribution contribution,
                                                                     Tr* __restrict__ result)
{
    namespace cg = cooperative_groups;

    const auto block = cg::this_thread_block();
    assert(block.dim_threads().x == ClusterConfig::iSize);
    assert(block.dim_threads().y == ClusterConfig::jSize);
    assert(block.dim_threads().z == GpuConfig::warpSize / ClusterConfig::iSize);
    const auto warp   = cg::tiled_partition<GpuConfig::warpSize>(block);
    const auto thread = cg::this_thread();

    const unsigned numTargets = iceil(lastBody - firstBody, GpuConfig::warpSize);
    alignas(16) __shared__ Tc xShared[GpuConfig::warpSize];
    alignas(16) __shared__ Tc yShared[GpuConfig::warpSize];
    alignas(16) __shared__ Tc zShared[GpuConfig::warpSize];
    alignas(16) __shared__ Th hShared[GpuConfig::warpSize];
    __shared__ int sharedTargetIdx;
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    if (block.thread_rank() == 0) init(&barrier, block.num_threads());
    block.sync();

    constexpr auto pbc = BoundaryType::periodic;

    const unsigned iShared = block.thread_index().x + block.thread_index().z * ClusterConfig::iSize;

    cuda::barrier<cuda::thread_scope_block>::arrival_token token;

    if (block.thread_rank() == 0)
    {
        sharedTargetIdx              = atomicAdd(&targetCounterGlob, 1);
        const unsigned nextTargetIdx = sharedTargetIdx;
        cuda::device::memcpy_async_tx(xShared, x + nextTargetIdx * GpuConfig::warpSize,
                                      cuda::aligned_size_t<16>(sizeof(Tc) * GpuConfig::warpSize), barrier);
        cuda::device::memcpy_async_tx(yShared, y + nextTargetIdx * GpuConfig::warpSize,
                                      cuda::aligned_size_t<16>(sizeof(Tc) * GpuConfig::warpSize), barrier);
        cuda::device::memcpy_async_tx(zShared, z + nextTargetIdx * GpuConfig::warpSize,
                                      cuda::aligned_size_t<16>(sizeof(Tc) * GpuConfig::warpSize), barrier);
        cuda::device::memcpy_async_tx(hShared, h + nextTargetIdx * GpuConfig::warpSize,
                                      cuda::aligned_size_t<16>(sizeof(Th) * GpuConfig::warpSize), barrier);
        token = cuda::device::barrier_arrive_tx(barrier, 1, (3 * sizeof(Tc) + sizeof(Th)) * GpuConfig::warpSize);
    }
    else { token = barrier.arrive(1); }

    while (true)
    {
        barrier.wait(std::move(token));

        const unsigned targetIdx = sharedTargetIdx;

        if (targetIdx >= numTargets) return;

        const unsigned i =
            targetIdx * GpuConfig::warpSize + block.thread_index().x + block.thread_index().z * ClusterConfig::iSize;

        const Vec3<Tc> iPos = {xShared[iShared], yShared[iShared], zShared[iShared]};
        const Th hi         = hShared[iShared];

        block.sync();

        if (block.thread_rank() == 0)
        {
            sharedTargetIdx              = atomicAdd(&targetCounterGlob, 1);
            const unsigned nextTargetIdx = sharedTargetIdx;
            if (nextTargetIdx < numTargets)
            {
                cuda::device::memcpy_async_tx(xShared, x + nextTargetIdx * GpuConfig::warpSize,
                                              cuda::aligned_size_t<16>(sizeof(Tc) * GpuConfig::warpSize), barrier);
                cuda::device::memcpy_async_tx(yShared, y + nextTargetIdx * GpuConfig::warpSize,
                                              cuda::aligned_size_t<16>(sizeof(Tc) * GpuConfig::warpSize), barrier);
                cuda::device::memcpy_async_tx(zShared, z + nextTargetIdx * GpuConfig::warpSize,
                                              cuda::aligned_size_t<16>(sizeof(Tc) * GpuConfig::warpSize), barrier);
                cuda::device::memcpy_async_tx(hShared, h + nextTargetIdx * GpuConfig::warpSize,
                                              cuda::aligned_size_t<16>(sizeof(Th) * GpuConfig::warpSize), barrier);
                token =
                    cuda::device::barrier_arrive_tx(barrier, 1, (3 * sizeof(Tc) + sizeof(Th)) * GpuConfig::warpSize);
            }
            else { token = barrier.arrive(1); }
        }
        else { token = barrier.arrive(1); }

        static_assert(ClusterConfig::iSize * ClusterConfig::jSize == GpuConfig::warpSize,
                      "Single warp required per cluster-cluster interaction");
        // const bool usePbc   = warp.any(anyPbc & !insideBox(iPos, {2 * hi, 2 * hi, 2 * hi}, box));

        const unsigned iCluster = imin(i, lastBody - 1) / ClusterConfig::iSize;

        auto distSq = [&](const Vec3<Tc>& jPos)
        {
            const bool anyPbc = box.boundaryX() == pbc | box.boundaryY() == pbc | box.boundaryZ() == pbc;
            return anyPbc ? distanceSq<true>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box)
                          : distanceSq<false>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box);
        };

        Tr sum = 0;
#pragma unroll
        for (unsigned jCluster = iCluster * ClusterConfig::iSize / ClusterConfig::jSize;
             jCluster < (iCluster * ClusterConfig::iSize +
                         (ClusterConfig::iSize > ClusterConfig::jSize ? ClusterConfig::iSize : ClusterConfig::jSize)) /
                            ClusterConfig::jSize;
             ++jCluster)
        {
            const unsigned j = jCluster * ClusterConfig::jSize + block.thread_index().y;
            if (i < lastBody & j < lastBody)
            {
                const Vec3<Tc> jPos{x[j], y[j], z[j]};
                const Th d2 = distSq(jPos);
                if (d2 < 4 * hi * hi) sum += contribution(i, iPos, hi, j, jPos, std::sqrt(d2));
            }
        }

        const unsigned iClusterNeighborsCount = imin(ncClustered[iCluster], ncmax);
#pragma unroll ClusterConfig::jSize
        for (unsigned jc = 0; jc < iClusterNeighborsCount; ++jc)
        {
            const unsigned jCluster = nidxClustered[clusterNeighborIndex(iCluster, jc, ncmax)];
            const unsigned j        = jCluster * ClusterConfig::jSize + block.thread_index().y;
            if (i < lastBody & j < lastBody & j / ClusterConfig::iSize != iCluster)
            {
                const Vec3<Tc> jPos{x[j], y[j], z[j]};
                const Th d2 = distSq(jPos);
                if (d2 < 4 * hi * hi) sum += contribution(i, iPos, hi, j, jPos, std::sqrt(d2));
            }
        }

#pragma unroll
        for (unsigned offset = GpuConfig::warpSize / 2; offset >= ClusterConfig::iSize; offset /= 2)
            sum += warp.shfl_down(sum, offset);

        if (block.thread_index().y == 0) result[i] = sum;
    }
}

template<class Tc, class Th, class Contribution, class Tr>
__global__
__launch_bounds__(ClusterConfig::iSize* ClusterConfig::jSize* GpuConfig::warpSize /
                  ClusterConfig::iSize) void findNeighborsClustered6(cstone::LocalIndex firstBody,
                                                                     cstone::LocalIndex lastBody,
                                                                     const Tc* __restrict__ x,
                                                                     const Tc* __restrict__ y,
                                                                     const Tc* __restrict__ z,
                                                                     const Th* __restrict__ h,
                                                                     const Box<Tc> box,
                                                                     const unsigned* __restrict__ ncClustered,
                                                                     const unsigned* __restrict__ nidxClustered,
                                                                     unsigned ncmax,
                                                                     Contribution contribution,
                                                                     Tr* __restrict__ result)
{
    namespace cg = cooperative_groups;

    const auto block = cg::this_thread_block();
    assert(block.dim_threads().x == ClusterConfig::iSize);
    assert(block.dim_threads().y == ClusterConfig::jSize);
    assert(block.dim_threads().z == GpuConfig::warpSize / ClusterConfig::iSize);
    const auto warp   = cg::tiled_partition<GpuConfig::warpSize>(block);
    const auto thread = cg::this_thread();

    const unsigned numTargets = iceil(lastBody - firstBody, GpuConfig::warpSize);
    alignas(16) __shared__ Tc xShared[GpuConfig::warpSize];
    alignas(16) __shared__ Tc yShared[GpuConfig::warpSize];
    alignas(16) __shared__ Tc zShared[GpuConfig::warpSize];
    alignas(16) __shared__ Th hShared[GpuConfig::warpSize];
    __shared__ int sharedTargetIdx;

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 1> pipelineState;
    auto pipeline = cuda::make_pipeline(block, &pipelineState);

    constexpr auto pbc = BoundaryType::periodic;

    const unsigned iShared = block.thread_index().x + block.thread_index().z * ClusterConfig::iSize;

    if (block.thread_rank() == 0) sharedTargetIdx = atomicAdd(&targetCounterGlob, 1);
    block.sync();
    const unsigned nextTargetIdx = sharedTargetIdx;
    pipeline.producer_acquire();
    cuda::memcpy_async(block, xShared, x + nextTargetIdx * GpuConfig::warpSize,
                       cuda::aligned_size_t<16>(sizeof(Tc) * GpuConfig::warpSize), pipeline);
    cuda::memcpy_async(block, yShared, y + nextTargetIdx * GpuConfig::warpSize,
                       cuda::aligned_size_t<16>(sizeof(Tc) * GpuConfig::warpSize), pipeline);
    cuda::memcpy_async(block, zShared, z + nextTargetIdx * GpuConfig::warpSize,
                       cuda::aligned_size_t<16>(sizeof(Tc) * GpuConfig::warpSize), pipeline);
    cuda::memcpy_async(block, hShared, h + nextTargetIdx * GpuConfig::warpSize,
                       cuda::aligned_size_t<16>(sizeof(Th) * GpuConfig::warpSize), pipeline);
    pipeline.producer_commit();

    while (true)
    {
        const unsigned targetIdx = sharedTargetIdx;
        block.sync();

        if (targetIdx >= numTargets) return;

        const unsigned i =
            targetIdx * GpuConfig::warpSize + block.thread_index().x + block.thread_index().z * ClusterConfig::iSize;

        pipeline.consumer_wait();
        const Vec3<Tc> iPos = {xShared[iShared], yShared[iShared], zShared[iShared]};
        const Th hi         = hShared[iShared];
        pipeline.consumer_release();

        if (block.thread_rank() == 0) sharedTargetIdx = atomicAdd(&targetCounterGlob, 1);
        block.sync();
        const unsigned nextTargetIdx = sharedTargetIdx;
        if (nextTargetIdx < numTargets)
        {
            pipeline.producer_acquire();
            cuda::memcpy_async(block, xShared, x + nextTargetIdx * GpuConfig::warpSize,
                               cuda::aligned_size_t<16>(sizeof(Tc) * GpuConfig::warpSize), pipeline);
            cuda::memcpy_async(block, yShared, y + nextTargetIdx * GpuConfig::warpSize,
                               cuda::aligned_size_t<16>(sizeof(Tc) * GpuConfig::warpSize), pipeline);
            cuda::memcpy_async(block, zShared, z + nextTargetIdx * GpuConfig::warpSize,
                               cuda::aligned_size_t<16>(sizeof(Tc) * GpuConfig::warpSize), pipeline);
            cuda::memcpy_async(block, hShared, h + nextTargetIdx * GpuConfig::warpSize,
                               cuda::aligned_size_t<16>(sizeof(Th) * GpuConfig::warpSize), pipeline);
            pipeline.producer_commit();
        }

        static_assert(ClusterConfig::iSize * ClusterConfig::jSize == GpuConfig::warpSize,
                      "Single warp required per cluster-cluster interaction");
        // const bool usePbc   = warp.any(anyPbc & !insideBox(iPos, {2 * hi, 2 * hi, 2 * hi}, box));

        const unsigned iCluster = imin(i, lastBody - 1) / ClusterConfig::iSize;

        auto distSq = [&](const Vec3<Tc>& jPos)
        {
            const bool anyPbc = box.boundaryX() == pbc | box.boundaryY() == pbc | box.boundaryZ() == pbc;
            return anyPbc ? distanceSq<true>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box)
                          : distanceSq<false>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box);
        };

        Tr sum = 0;
#pragma unroll
        for (unsigned jCluster = iCluster * ClusterConfig::iSize / ClusterConfig::jSize;
             jCluster < (iCluster * ClusterConfig::iSize +
                         (ClusterConfig::iSize > ClusterConfig::jSize ? ClusterConfig::iSize : ClusterConfig::jSize)) /
                            ClusterConfig::jSize;
             ++jCluster)
        {
            const unsigned j = jCluster * ClusterConfig::jSize + block.thread_index().y;
            if (i < lastBody & j < lastBody)
            {
                const Vec3<Tc> jPos{x[j], y[j], z[j]};
                const Th d2 = distSq(jPos);
                if (d2 < 4 * hi * hi) sum += contribution(i, iPos, hi, j, jPos, std::sqrt(d2));
            }
        }

        const unsigned iClusterNeighborsCount = imin(ncClustered[iCluster], ncmax);
#pragma unroll ClusterConfig::jSize
        for (unsigned jc = 0; jc < iClusterNeighborsCount; ++jc)
        {
            const unsigned jCluster = nidxClustered[clusterNeighborIndex(iCluster, jc, ncmax)];
            const unsigned j        = jCluster * ClusterConfig::jSize + block.thread_index().y;
            if (i < lastBody & j < lastBody & j / ClusterConfig::iSize != iCluster)
            {
                const Vec3<Tc> jPos{x[j], y[j], z[j]};
                const Th d2 = distSq(jPos);
                if (d2 < 4 * hi * hi) sum += contribution(i, iPos, hi, j, jPos, std::sqrt(d2));
            }
        }

#pragma unroll
        for (unsigned offset = GpuConfig::warpSize / 2; offset >= ClusterConfig::iSize; offset /= 2)
            sum += warp.shfl_down(sum, offset);

        if (block.thread_index().y == 0) result[i] = sum;
    }
}

template<class Tc, class Th, class Contribution, class Tr>
__global__
__launch_bounds__(ClusterConfig::iSize* ClusterConfig::jSize* GpuConfig::warpSize /
                  ClusterConfig::iSize) void findNeighborsClustered7(cstone::LocalIndex firstBody,
                                                                     cstone::LocalIndex lastBody,
                                                                     const Tc* __restrict__ x,
                                                                     const Tc* __restrict__ y,
                                                                     const Tc* __restrict__ z,
                                                                     const Th* __restrict__ h,
                                                                     const Box<Tc> box,
                                                                     const unsigned* __restrict__ ncClustered,
                                                                     const unsigned* __restrict__ nidxClustered,
                                                                     unsigned ncmax,
                                                                     Contribution contribution,
                                                                     Tr* __restrict__ result)
{
    namespace cg = cooperative_groups;

    const auto block = cg::this_thread_block();
    assert(block.dim_threads().x == ClusterConfig::iSize);
    assert(block.dim_threads().y == ClusterConfig::jSize);
    assert(block.dim_threads().z == GpuConfig::warpSize / ClusterConfig::iSize);
    const auto warp   = cg::tiled_partition<GpuConfig::warpSize>(block);
    const auto thread = cg::this_thread();

    const unsigned numTargets = iceil(lastBody - firstBody, GpuConfig::warpSize);
    alignas(16) __shared__ Tc xShared[GpuConfig::warpSize];
    alignas(16) __shared__ Tc yShared[GpuConfig::warpSize];
    alignas(16) __shared__ Tc zShared[GpuConfig::warpSize];
    alignas(16) __shared__ Th hShared[GpuConfig::warpSize];
    __shared__ int sharedTargetIdx;

    auto pipeline = cuda::make_pipeline();

    constexpr auto pbc = BoundaryType::periodic;

    const unsigned iShared = block.thread_index().x + block.thread_index().z * ClusterConfig::iSize;

    if (block.thread_rank() == 0) sharedTargetIdx = atomicAdd(&targetCounterGlob, 1);
    block.sync();
    const unsigned nextTargetIdx = sharedTargetIdx;
    if (warp.meta_group_rank() == 0)
    {
        pipeline.producer_acquire();
        cuda::memcpy_async(thread, xShared + warp.thread_rank(),
                           x + nextTargetIdx * GpuConfig::warpSize + warp.thread_rank(), sizeof(Tc), pipeline);
        cuda::memcpy_async(thread, yShared + warp.thread_rank(),
                           y + nextTargetIdx * GpuConfig::warpSize + warp.thread_rank(), sizeof(Tc), pipeline);
        cuda::memcpy_async(thread, zShared + warp.thread_rank(),
                           z + nextTargetIdx * GpuConfig::warpSize + warp.thread_rank(), sizeof(Tc), pipeline);
        cuda::memcpy_async(thread, hShared + warp.thread_rank(),
                           h + nextTargetIdx * GpuConfig::warpSize + warp.thread_rank(), sizeof(Th), pipeline);
        pipeline.producer_commit();
    }

    while (true)
    {
        const unsigned targetIdx = sharedTargetIdx;
        block.sync();

        if (targetIdx >= numTargets) return;

        const unsigned i =
            targetIdx * GpuConfig::warpSize + block.thread_index().x + block.thread_index().z * ClusterConfig::iSize;

        if (warp.meta_group_rank() == 0) pipeline.consumer_wait();
        block.sync();
        const Vec3<Tc> iPos = {xShared[iShared], yShared[iShared], zShared[iShared]};
        const Th hi         = hShared[iShared];

        if (block.thread_rank() == 0) sharedTargetIdx = atomicAdd(&targetCounterGlob, 1);
        block.sync();
        if (warp.meta_group_rank() == 0) pipeline.consumer_release();
        const unsigned nextTargetIdx = sharedTargetIdx;
        if (nextTargetIdx < numTargets && warp.meta_group_rank() == 0)
        {
            pipeline.producer_acquire();
            cuda::memcpy_async(thread, xShared + warp.thread_rank(),
                               x + nextTargetIdx * GpuConfig::warpSize + warp.thread_rank(), sizeof(Tc), pipeline);
            cuda::memcpy_async(thread, yShared + warp.thread_rank(),
                               y + nextTargetIdx * GpuConfig::warpSize + warp.thread_rank(), sizeof(Tc), pipeline);
            cuda::memcpy_async(thread, zShared + warp.thread_rank(),
                               z + nextTargetIdx * GpuConfig::warpSize + warp.thread_rank(), sizeof(Tc), pipeline);
            cuda::memcpy_async(thread, hShared + warp.thread_rank(),
                               h + nextTargetIdx * GpuConfig::warpSize + warp.thread_rank(), sizeof(Th), pipeline);
            pipeline.producer_commit();
        }

        static_assert(ClusterConfig::iSize * ClusterConfig::jSize == GpuConfig::warpSize,
                      "Single warp required per cluster-cluster interaction");
        // const bool usePbc   = warp.any(anyPbc & !insideBox(iPos, {2 * hi, 2 * hi, 2 * hi}, box));

        const unsigned iCluster = imin(i, lastBody - 1) / ClusterConfig::iSize;

        auto distSq = [&](const Vec3<Tc>& jPos)
        {
            const bool anyPbc = box.boundaryX() == pbc | box.boundaryY() == pbc | box.boundaryZ() == pbc;
            return anyPbc ? distanceSq<true>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box)
                          : distanceSq<false>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box);
        };

        Tr sum = 0;
#pragma unroll
        for (unsigned jCluster = iCluster * ClusterConfig::iSize / ClusterConfig::jSize;
             jCluster < (iCluster * ClusterConfig::iSize +
                         (ClusterConfig::iSize > ClusterConfig::jSize ? ClusterConfig::iSize : ClusterConfig::jSize)) /
                            ClusterConfig::jSize;
             ++jCluster)
        {
            const unsigned j = jCluster * ClusterConfig::jSize + block.thread_index().y;
            if (i < lastBody & j < lastBody)
            {
                const Vec3<Tc> jPos{x[j], y[j], z[j]};
                const Th d2 = distSq(jPos);
                if (d2 < 4 * hi * hi) sum += contribution(i, iPos, hi, j, jPos, std::sqrt(d2));
            }
        }

        const unsigned iClusterNeighborsCount = imin(ncClustered[iCluster], ncmax);
#pragma unroll ClusterConfig::jSize
        for (unsigned jc = 0; jc < iClusterNeighborsCount; ++jc)
        {
            const unsigned jCluster = nidxClustered[clusterNeighborIndex(iCluster, jc, ncmax)];
            const unsigned j        = jCluster * ClusterConfig::jSize + block.thread_index().y;
            if (i < lastBody & j < lastBody & j / ClusterConfig::iSize != iCluster)
            {
                const Vec3<Tc> jPos{x[j], y[j], z[j]};
                const Th d2 = distSq(jPos);
                if (d2 < 4 * hi * hi) sum += contribution(i, iPos, hi, j, jPos, std::sqrt(d2));
            }
        }

#pragma unroll
        for (unsigned offset = GpuConfig::warpSize / 2; offset >= ClusterConfig::iSize; offset /= 2)
            sum += warp.shfl_down(sum, offset);

        if (block.thread_index().y == 0) result[i] = sum;
    }
}

template<int warpsPerBlock, bool bypassL1CacheOnLoads = true, class Tc, class Th, class Contribution, class Tr>
__global__ __launch_bounds__(ClusterConfig::iSize* ClusterConfig::jSize* warpsPerBlock) void findNeighborsClustered8(
    cstone::LocalIndex firstBody,
    cstone::LocalIndex lastBody,
    const Tc* __restrict__ x,
    const Tc* __restrict__ y,
    const Tc* __restrict__ z,
    const Th* __restrict__ h,
    const Box<Tc> box,
    const unsigned* __restrict__ ncClustered,
    const unsigned* __restrict__ nidxClustered,
    unsigned ncmax,
    Contribution contribution,
    Tr* __restrict__ result)
{
    namespace cg = cooperative_groups;

    const auto block = cg::this_thread_block();
    assert(block.dim_threads().x == ClusterConfig::iSize);
    assert(block.dim_threads().y == ClusterConfig::jSize);
    static_assert(warpsPerBlock > 0 && ClusterConfig::iSize * ClusterConfig::jSize == GpuConfig::warpSize);
    assert(block.dim_threads().z == warpsPerBlock);
    const auto warp   = cg::tiled_partition<GpuConfig::warpSize>(block);
    const auto thread = cg::this_thread();

    constexpr auto pbc = BoundaryType::periodic;
    const bool anyPbc  = box.boundaryX() == pbc | box.boundaryY() == pbc | box.boundaryZ() == pbc;

    assert(firstBody == 0); // TODO: other cases
    const unsigned numIClusters = iceil(lastBody - firstBody, ClusterConfig::iSize);

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
        if constexpr (bypassL1CacheOnLoads)
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
    preloadNextICluster();
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
        const Th hi = h[i];
#endif

        // const bool usePbc   = warp.any(anyPbc & !insideBox(iPos, {2 * hi, 2 * hi, 2 * hi}, box));

        const auto distSq = [&](const Vec3<Tc>& jPos)
        {
            return anyPbc ? distanceSq<true>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box)
                          : distanceSq<false>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box);
        };

        Tr sum                               = 0;
        const auto computeClusterInteraction = [&](unsigned jCluster)
        {
            const unsigned j = jCluster * ClusterConfig::jSize + block.thread_index().y;
            if (i < lastBody & j < lastBody)
            {
                const Vec3<Tc> jPos{x[j], y[j], z[j]};
                const Th d2 = distSq(jPos);
                if (d2 < 4 * hi * hi) sum += contribution(i, iPos, hi, j, jPos, std::sqrt(d2));
            }
        };

#pragma unroll
        for (unsigned jCluster = iCluster * ClusterConfig::iSize / ClusterConfig::jSize;
             jCluster < (iCluster * ClusterConfig::iSize +
                         (ClusterConfig::iSize > ClusterConfig::jSize ? ClusterConfig::iSize : ClusterConfig::jSize)) /
                            ClusterConfig::jSize;
             ++jCluster)
            computeClusterInteraction(jCluster);

        const unsigned iClusterNeighborsCount = imin(ncClustered[iCluster], ncmax);
#pragma unroll ClusterConfig::jSize
        for (unsigned jc = 0; jc < iClusterNeighborsCount; ++jc)
        {
            const unsigned jCluster = nidxClustered[clusterNeighborIndex(iCluster, jc, ncmax)];
            computeClusterInteraction(jCluster);
        }

#pragma unroll
        for (unsigned offset = GpuConfig::warpSize / 2; offset >= ClusterConfig::iSize; offset /= 2)
            sum += warp.shfl_down(sum, offset);

        if (block.thread_index().y == 0) result[i] = sum;
    }
}

template<int warpsPerBlock, bool bypassL1CacheOnLoads = true, class Tc, class Th, class Contribution, class Tr>
__global__ /*__launch_bounds__(ClusterConfig::iSize* ClusterConfig::jSize* warpsPerBlock)*/
    __maxnreg__(64) void findNeighborsClustered9(cstone::LocalIndex firstBody,
                                                 cstone::LocalIndex lastBody,
                                                 const Tc* __restrict__ x,
                                                 const Tc* __restrict__ y,
                                                 const Tc* __restrict__ z,
                                                 const Th* __restrict__ h,
                                                 const Box<Tc> box,
                                                 const unsigned* __restrict__ ncClustered,
                                                 const unsigned* __restrict__ nidxClustered,
                                                 unsigned ncmax,
                                                 Contribution contribution,
                                                 Tr* __restrict__ result)
{
    namespace cg = cooperative_groups;

    const auto block = cg::this_thread_block();
    assert(block.dim_threads().x == ClusterConfig::iSize);
    assert(block.dim_threads().y == ClusterConfig::jSize);
    static_assert(warpsPerBlock > 0 && ClusterConfig::iSize * ClusterConfig::jSize == GpuConfig::warpSize);
    assert(block.dim_threads().z == warpsPerBlock);
    const auto warp = cg::tiled_partition<GpuConfig::warpSize>(block);

    alignas(16) __shared__ Tc xjShared[warpsPerBlock][ClusterConfig::jSize];
    alignas(16) __shared__ Tc yjShared[warpsPerBlock][ClusterConfig::jSize];
    alignas(16) __shared__ Tc zjShared[warpsPerBlock][ClusterConfig::jSize];

    auto jPipeline = cuda::make_pipeline();

    constexpr auto pbc = BoundaryType::periodic;
    const bool anyPbc  = box.boundaryX() == pbc | box.boundaryY() == pbc | box.boundaryZ() == pbc;

    const unsigned numIClusters = iceil(lastBody - firstBody, ClusterConfig::iSize);

    while (true)
    {
        unsigned iCluster;
        if (warp.thread_rank() == 0) iCluster = atomicAdd(&targetCounterGlob, 1);
        iCluster = warp.shfl(iCluster, 0);

        if (iCluster >= iceil(lastBody - firstBody, ClusterConfig::iSize)) return;

        const unsigned i    = iCluster * ClusterConfig::iSize + block.thread_index().x;
        const Vec3<Tc> iPos = {x[i], y[i], z[i]};
        const Th hi         = h[i];

        // const bool usePbc = warp.any(anyPbc & !insideBox(iPos, {2 * hi, 2 * hi, 2 * hi}, box));

        const auto distSq = [&](const Vec3<Tc>& jPos)
        {
            return anyPbc ? distanceSq<true>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box)
                          : distanceSq<false>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box);
        };

        const auto preloadJCluster = [&](unsigned nextJCluster)
        {
            jPipeline.producer_acquire();
            const auto thread = cg::this_thread();
            if constexpr (bypassL1CacheOnLoads)
            {
                constexpr int numTcPer16Bytes = 16 / sizeof(Tc);
                if (warp.thread_rank() < ClusterConfig::jSize / numTcPer16Bytes)
                {
                    const unsigned nextJ = nextJCluster * ClusterConfig::jSize + warp.thread_rank() * numTcPer16Bytes;
                    cuda::memcpy_async(thread, &xjShared[block.thread_index().z][warp.thread_rank() * numTcPer16Bytes],
                                       &x[nextJ], cuda::aligned_size_t<16>(16), jPipeline);
                    cuda::memcpy_async(thread, &yjShared[block.thread_index().z][warp.thread_rank() * numTcPer16Bytes],
                                       &y[nextJ], cuda::aligned_size_t<16>(16), jPipeline);
                    cuda::memcpy_async(thread, &zjShared[block.thread_index().z][warp.thread_rank() * numTcPer16Bytes],
                                       &z[nextJ], cuda::aligned_size_t<16>(16), jPipeline);
                }
            }
            else
            {
                const unsigned nextJ = nextJCluster * ClusterConfig::jSize + warp.thread_rank();
                if (warp.thread_rank() < ClusterConfig::jSize && nextJ < lastBody)
                {
                    cuda::memcpy_async(thread, &xjShared[block.thread_index().z][warp.thread_rank()], &x[nextJ],
                                       cuda::aligned_size_t<sizeof(Tc)>(sizeof(Tc)), jPipeline);
                    cuda::memcpy_async(thread, &yjShared[block.thread_index().z][warp.thread_rank()], &y[nextJ],
                                       cuda::aligned_size_t<sizeof(Tc)>(sizeof(Tc)), jPipeline);
                    cuda::memcpy_async(thread, &zjShared[block.thread_index().z][warp.thread_rank()], &z[nextJ],
                                       cuda::aligned_size_t<sizeof(Tc)>(sizeof(Tc)), jPipeline);
                }
            }
            jPipeline.producer_commit();
        };

        Tr sum = 0;

        const auto computeClusterInteraction = [&](unsigned jCluster, unsigned nextJCluster)
        {
            const unsigned j = jCluster * ClusterConfig::jSize + block.thread_index().y;
            jPipeline.consumer_wait();
            warp.sync();
            const Vec3<Tc> jPos{xjShared[block.thread_index().z][block.thread_index().y],
                                yjShared[block.thread_index().z][block.thread_index().y],
                                zjShared[block.thread_index().z][block.thread_index().y]};
            jPipeline.consumer_release();
            if (jCluster != nextJCluster) preloadJCluster(nextJCluster);
            const Th d2 = distSq(jPos);
            if (i < lastBody & j < lastBody & d2 < 4 * hi * hi)
                sum += contribution(i, iPos, hi, j, jPos, std::sqrt(d2));
        };

        constexpr unsigned overlappingJClusters =
            ClusterConfig::iSize > ClusterConfig::jSize ? ClusterConfig::iSize / ClusterConfig::jSize : 1;

        const unsigned iClusterNeighborsCount = imin(ncClustered[iCluster], ncmax);

        const unsigned firstNbJCluster = nidxClustered[clusterNeighborIndex(iCluster, 0, ncmax)];
        unsigned nextJCluster          = iCluster * ClusterConfig::iSize / ClusterConfig::jSize;
        preloadJCluster(nextJCluster);

#pragma unroll
        for (unsigned overlappingJCluster = 0; overlappingJCluster < overlappingJClusters; ++overlappingJCluster)
        {
            const unsigned jCluster = nextJCluster;
            nextJCluster            = overlappingJCluster + 1 < overlappingJClusters ? jCluster + 1
                                      : iClusterNeighborsCount > 0                   ? firstNbJCluster
                                                                                     : jCluster;
            computeClusterInteraction(jCluster, nextJCluster);
        }

#pragma unroll 4
        for (unsigned jc = 0; jc < iClusterNeighborsCount; ++jc)
        {
            const unsigned jCluster = nextJCluster;
            nextJCluster =
                nidxClustered[clusterNeighborIndex(iCluster, imin(jc + 1, iClusterNeighborsCount - 1), ncmax)];
            computeClusterInteraction(jCluster, nextJCluster);
        }

#pragma unroll
        for (unsigned offset = GpuConfig::warpSize / 2; offset >= ClusterConfig::iSize; offset /= 2)
            sum += warp.shfl_down(sum, offset);

        if (block.thread_index().y == 0) result[i] = sum;
    }
}

template<int warpsPerBlock, bool bypassL1CacheOnLoads = true, class Tc, class Th, class Contribution, class Tr>
__global__ __launch_bounds__(ClusterConfig::iSize* ClusterConfig::jSize* warpsPerBlock) void findNeighborsClustered10(
    cstone::LocalIndex firstBody,
    cstone::LocalIndex lastBody,
    const Tc* __restrict__ x,
    const Tc* __restrict__ y,
    const Tc* __restrict__ z,
    const Th* __restrict__ h,
    const Box<Tc> box,
    const unsigned* __restrict__ ncClustered,
    const unsigned* __restrict__ nidxClustered,
    unsigned ncmax,
    Contribution contribution,
    Tr* __restrict__ result)
{
    namespace cg = cooperative_groups;

    const auto block = cg::this_thread_block();
    assert(block.dim_threads().x == ClusterConfig::iSize);
    assert(block.dim_threads().y == ClusterConfig::jSize);
    static_assert(warpsPerBlock > 0 && ClusterConfig::iSize * ClusterConfig::jSize == GpuConfig::warpSize);
    assert(block.dim_threads().z == warpsPerBlock);
    const auto warp   = cg::tiled_partition<GpuConfig::warpSize>(block);
    const auto thread = cg::this_thread();

    constexpr auto pbc = BoundaryType::periodic;
    const bool anyPbc  = box.boundaryX() == pbc | box.boundaryY() == pbc | box.boundaryZ() == pbc;

    const unsigned numIClusters = iceil(lastBody - firstBody, ClusterConfig::iSize);

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
        if constexpr (bypassL1CacheOnLoads)
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
    preloadNextICluster();
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
        const Th hi = h[i];
#endif

        // const bool usePbc   = warp.any(anyPbc & !insideBox(iPos, {2 * hi, 2 * hi, 2 * hi}, box));

        const auto distSq = [&](const Vec3<Tc>& jPos)
        {
            return anyPbc ? distanceSq<true>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box)
                          : distanceSq<false>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box);
        };

        Tr sum                               = 0;
        const auto computeClusterInteraction = [&](unsigned jCluster)
        {
            const unsigned j = jCluster * ClusterConfig::jSize + block.thread_index().y;
            if (i < lastBody & j < lastBody)
            {
                const Vec3<Tc> jPos{x[j], y[j], z[j]};
                const Th d2 = distSq(jPos);
                if (d2 < 4 * hi * hi) sum += contribution(i, iPos, hi, j, jPos, std::sqrt(d2));
            }
        };

#pragma unroll
        for (unsigned jCluster = iCluster * ClusterConfig::iSize / ClusterConfig::jSize;
             jCluster < (iCluster * ClusterConfig::iSize +
                         (ClusterConfig::iSize > ClusterConfig::jSize ? ClusterConfig::iSize : ClusterConfig::jSize)) /
                            ClusterConfig::jSize;
             ++jCluster)
            computeClusterInteraction(jCluster);

        const unsigned iClusterNeighborsCount = imin(ncClustered[iCluster], ncmax);
        NeighborListDecompressor decomp(&nidxClustered[clusterNeighborIndex(iCluster, 0, ncmax)],
                                        sizeof(unsigned) * ncmax);
        for (unsigned jCluster : decomp)
            computeClusterInteraction(jCluster);

#pragma unroll
        for (unsigned offset = GpuConfig::warpSize / 2; offset >= ClusterConfig::iSize; offset /= 2)
            sum += warp.shfl_down(sum, offset);

        if (block.thread_index().y == 0) result[i] = sum;
    }
}

} // namespace cstone

#undef CSTONE_USE_CUDA_PIPELINE
