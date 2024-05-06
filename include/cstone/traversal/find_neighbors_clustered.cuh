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

template<class Tc, class Th>
__global__
__launch_bounds__(TravConfig::numThreads) void findNeighborsClustered(cstone::LocalIndex firstBody,
                                                                      cstone::LocalIndex lastBody,
                                                                      const Tc* __restrict__ x,
                                                                      const Tc* __restrict__ y,
                                                                      const Tc* __restrict__ z,
                                                                      const Th* __restrict__ h,
                                                                      const Box<Tc> box,
                                                                      unsigned* __restrict__ nc,
                                                                      unsigned* __restrict__ nidx,
                                                                      unsigned ngmax,
                                                                      const unsigned* __restrict__ ncClustered,
                                                                      const unsigned* __restrict__ nidxClustered,
                                                                      unsigned ncmax)
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
        const cstone::LocalIndex bodyEnd     = imin(bodyBegin + TravConfig::targetSize, lastBody);
        unsigned* warpNidx                   = nidx + targetIdx * TravConfig::targetSize * ngmax;
        const cstone::LocalIndex bodyIdxLane = bodyBegin + laneIdx;

        constexpr auto pbc = BoundaryType::periodic;
        const bool anyPbc  = box.boundaryX() == pbc || box.boundaryY() == pbc || box.boundaryZ() == pbc;

        auto pos_i = loadTarget(bodyBegin, bodyEnd, laneIdx, x, y, z, h);
        unsigned nc_i[TravConfig::nwt];
        for (int k = 0; k < TravConfig::nwt; ++k)
        {
            nc_i[k] = 0;

            bool usePbc   = anyPbc && !insideBox(Vec3<Tc>{pos_i[k][0], pos_i[k][1], pos_i[k][2]},
                                                 {pos_i[k][3], pos_i[k][3], pos_i[k][3]}, box);
            auto radiusSq = pos_i[k][3] * pos_i[k][3];

            auto i                                = bodyIdxLane + k * GpuConfig::warpSize;
            auto iCluster                         = i / ClusterConfig::iSize;
            const unsigned iClusterNeighborsCount = ncClustered[iCluster];

            for (unsigned jc = 0; jc < imin(iClusterNeighborsCount, ncmax); ++jc)
            {
                auto jCluster = nidxClustered[clusterNeighborIndex(iCluster, jc, ncmax)];
                for (unsigned j = jCluster * ClusterConfig::jSize;
                     j < imin((jCluster + 1) * ClusterConfig::jSize, lastBody); ++j)
                {
                    if (i != j)
                    {
                        const Vec3<Tc> pos_j = {x[j], y[j], z[j]};
                        auto d2 = usePbc ? distanceSq<true>(pos_j[0], pos_j[1], pos_j[2], pos_i[k][0], pos_i[k][1],
                                                            pos_i[k][2], box)
                                         : distanceSq<false>(pos_j[0], pos_j[1], pos_j[2], pos_i[k][0], pos_i[k][1],
                                                             pos_i[k][2], box);
                        if (d2 < radiusSq)
                        {
                            if (nc_i[k] < ngmax) warpNidx[TravConfig::targetSize * nc_i[k] + laneIdx] = j;
                            ++nc_i[k];
                        }
                    }
                }
            }
        }

        for (int i = 0; i < TravConfig::nwt; i++)
        {
            const cstone::LocalIndex bodyIdx = bodyIdxLane + i * GpuConfig::warpSize;
            if (bodyIdx < bodyEnd) { nc[bodyIdx] = nc_i[i]; }
        }
    }
}

template<class Tc, class Th>
__global__
__launch_bounds__(TravConfig::numThreads) void findNeighborsClustered2(cstone::LocalIndex firstBody,
                                                                       cstone::LocalIndex lastBody,
                                                                       const Tc* __restrict__ x,
                                                                       const Tc* __restrict__ y,
                                                                       const Tc* __restrict__ z,
                                                                       const Th* __restrict__ h,
                                                                       const Box<Tc> box,
                                                                       unsigned* __restrict__ nc,
                                                                       unsigned* __restrict__ nidx,
                                                                       unsigned ngmax,
                                                                       const unsigned* __restrict__ ncClustered,
                                                                       const unsigned* __restrict__ nidxClustered,
                                                                       unsigned ncmax)
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

        static_assert(TravConfig::targetSize == GpuConfig::warpSize, "Requires targetSize == warpSize");
        static_assert(ClusterConfig::iSize * ClusterConfig::jSize == GpuConfig::warpSize,
                      "Single warp required per cluster-cluster interaction");
        constexpr unsigned clustersPerWarp = GpuConfig::warpSize / ClusterConfig::iSize;

        const auto iSuperCluster        = firstBody + GpuConfig::warpSize * targetIdx + laneIdx;
        const auto iSuperClusterClamped = imin(iSuperCluster, lastBody - 1);

        Vec4<Tc> iPosSuperCluster      = {x[iSuperClusterClamped], y[iSuperClusterClamped], z[iSuperClusterClamped],
                                          h[iSuperClusterClamped] * 2};
        unsigned neighborsSuperCluster = 0;

        for (unsigned c = 0; c < clustersPerWarp; ++c)
        {
            const auto i = (iSuperCluster / GpuConfig::warpSize) * GpuConfig::warpSize +
                           laneIdx % ClusterConfig::iSize + c * ClusterConfig::iSize;

            // TODO: remove once nidx is not used anymore
            if (i < lastBody) nc[i] = 0;
            __syncthreads();

            const auto iCluster                   = imin(i, lastBody - 1) / ClusterConfig::iSize;
            const unsigned iClusterNeighborsCount = ncClustered[iCluster];

            const unsigned iPosSrcLane = laneIdx % ClusterConfig::iSize + c * ClusterConfig::iSize;
            const Vec4<Tc> iPos{shflSync(iPosSuperCluster[0], iPosSrcLane), shflSync(iPosSuperCluster[1], iPosSrcLane),
                                shflSync(iPosSuperCluster[2], iPosSrcLane), shflSync(iPosSuperCluster[3], iPosSrcLane)};
            const auto radiusSq = iPos[3] * iPos[3];

            unsigned neighbors = 0;
            for (unsigned jCluster = iCluster * ClusterConfig::iSize / ClusterConfig::jSize;
                 jCluster <
                 (iCluster * ClusterConfig::iSize +
                  (ClusterConfig::iSize > ClusterConfig::jSize ? ClusterConfig::iSize : ClusterConfig::jSize)) /
                     ClusterConfig::jSize;
                 ++jCluster)
            {
                const auto j = jCluster * ClusterConfig::jSize + laneIdx / ClusterConfig::iSize;
                if (i < lastBody && j < lastBody && i != j)
                {
                    const Vec3<Tc> jPos{x[j], y[j], z[j]};
                    const auto d2 = distanceSq<true>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box);
                    // neighbors += d2 < radiusSq;
                    if (d2 < radiusSq)
                    {
                        unsigned nb                                                    = atomicAdd(&nc[i], 1);
                        nidx[(i / TravConfig::targetSize) * TravConfig::targetSize * ngmax +
                             TravConfig::targetSize * nb + i % TravConfig::targetSize] = j;
                    }
                }
            }

            for (unsigned jc = 0; jc < imin(ncmax, iClusterNeighborsCount); ++jc)
            {
                const unsigned jCluster = nidxClustered[clusterNeighborIndex(iCluster, jc, ncmax)];

                const auto j = jCluster * ClusterConfig::jSize + laneIdx / ClusterConfig::iSize;
                if (i < lastBody && j < lastBody && j / ClusterConfig::iSize != iCluster)
                {
                    const Vec3<Tc> jPos{x[j], y[j], z[j]};
                    const auto d2 = distanceSq<true>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box);
                    // neighbors += d2 < radiusSq;
                    if (d2 < radiusSq)
                    {
                        unsigned nb                                                    = atomicAdd(&nc[i], 1);
                        nidx[(i / TravConfig::targetSize) * TravConfig::targetSize * ngmax +
                             TravConfig::targetSize * nb + i % TravConfig::targetSize] = j;
                    }
                }
            }

            for (unsigned offset = GpuConfig::warpSize / 2; offset >= ClusterConfig::iSize; offset /= 2)
                neighbors += shflDownSync(neighbors, offset);

            neighbors = shflSync(neighbors, laneIdx % ClusterConfig::iSize);
            if (laneIdx / ClusterConfig::iSize == c) neighborsSuperCluster = neighbors;
        }

        // if (iSuperCluster < lastBody) nc[iSuperCluster] = neighborsSuperCluster;
    }
}

} // namespace cstone
