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

#include "cstone/traversal/find_neighbors.cuh"

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

    alignas(16) __shared__ Tc xiSharedBuffer[warpsPerBlock][ClusterConfig::iSize];
    alignas(16) __shared__ Tc yiSharedBuffer[warpsPerBlock][ClusterConfig::iSize];
    alignas(16) __shared__ Tc ziSharedBuffer[warpsPerBlock][ClusterConfig::iSize];
    alignas(16) __shared__ Th hiSharedBuffer[warpsPerBlock][ClusterConfig::iSize];
    Tc* const xiShared = xiSharedBuffer[block.thread_index().z];
    Tc* const yiShared = yiSharedBuffer[block.thread_index().z];
    Tc* const ziShared = ziSharedBuffer[block.thread_index().z];
    Th* const hiShared = hiSharedBuffer[block.thread_index().z];

    auto iPipeline = cuda::make_pipeline();

    constexpr auto pbc = BoundaryType::periodic;
    const bool anyPbc  = box.boundaryX() == pbc | box.boundaryY() == pbc | box.boundaryZ() == pbc;

    const unsigned numIClusters = iceil(lastBody - firstBody, ClusterConfig::iSize);
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

    while (true)
    {
        iCluster = nextICluster;

        if (iCluster >= numIClusters) return;

        const unsigned i = iCluster * ClusterConfig::iSize + block.thread_index().x;

        iPipeline.consumer_wait();
        warp.sync();
        const Vec3<Tc> iPos = {xiShared[block.thread_index().x], yiShared[block.thread_index().x],
                               ziShared[block.thread_index().x]};
        const Th hi         = hiShared[block.thread_index().x];

        if (warp.thread_rank() == 0) nextICluster = atomicAdd(&targetCounterGlob, 1);
        nextICluster = warp.shfl(nextICluster, 0);
        iPipeline.consumer_release();
        if (nextICluster < numIClusters) preloadNextICluster();

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

} // namespace cstone
