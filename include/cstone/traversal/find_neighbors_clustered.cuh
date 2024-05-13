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
    constexpr unsigned blockSize = TravConfig::targetSize / ClusterConfig::iSize;
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
            if (ncc > 0 && nidx(iClusterWarp, warpTarget, ncc - 1) == jCluster) return;

            unsigned nb = 0, last = ncc;
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
__launch_bounds__(TravConfig::numThreads) void findNeighborsClustered(cstone::LocalIndex firstBody,
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
                        const auto dist = distancePBC(box, hi, iPos[0], iPos[1], iPos[2], jPos[0], jPos[1], jPos[2]);
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
                        const auto dist = distancePBC(box, hi, iPos[0], iPos[1], iPos[2], jPos[0], jPos[1], jPos[2]);
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
                    const auto d2 = distSq(jPos);
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
                    const auto d2 = distSq(jPos);
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

} // namespace cstone
