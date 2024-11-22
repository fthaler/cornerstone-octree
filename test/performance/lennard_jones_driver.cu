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
 * @brief  Lennard-Jones kernel with various neighbor search strategies
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#include <iostream>
#include <map>
#include <mutex>
#include <shared_mutex>

#include <cuda/annotated_ptr>

#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/findneighbors.hpp"

#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/traversal/find_neighbors_clustered.cuh"

#include "../coord_samples/face_centered_cubic.hpp"

using namespace cstone;

constexpr unsigned ncmax = 256;

template<class Tc, class Th, class KeyType>
std::tuple<std::vector<LocalIndex>, std::vector<unsigned>> buildNeighborhoodCPU(std::size_t firstBody,
                                                                                std::size_t lastBody,
                                                                                const Tc* x,
                                                                                const Tc* y,
                                                                                const Tc* z,
                                                                                const Th* h,
                                                                                OctreeNsView<Tc, KeyType> tree,
                                                                                const Box<Tc>& box,
                                                                                unsigned ngmax)
{
    std::vector<LocalIndex> neighbors(ngmax * lastBody);
    std::vector<unsigned> neighborsCount(lastBody);

#pragma omp parallel for simd
    for (std::size_t i = firstBody; i < lastBody; ++i)
    {
        neighborsCount[i] = findNeighbors(i, x, y, z, h, tree, box, ngmax, neighbors.data() + i * ngmax);
    }

    return {neighbors, neighborsCount};
}

template<class Tc, class T>
void computeLjCPU(const std::size_t firstBody,
                  const std::size_t lastBody,

                  const Tc* __restrict__ x,
                  const Tc* __restrict__ y,
                  const Tc* __restrict__ z,
                  const T* __restrict__ h,
                  const T lj1,
                  const T lj2,
                  const Box<Tc>& box,
                  const unsigned ngmax,
                  T* __restrict__ afx,
                  T* __restrict__ afy,
                  T* __restrict__ afz,
                  const std::tuple<std::vector<LocalIndex>, std::vector<unsigned>>& neighborhood)
{
    auto& [neighbors, neighborsCount] = neighborhood;
#pragma omp parallel for simd
    for (std::size_t i = firstBody; i < lastBody; ++i)
    {
        const Tc xi = x[i];
        const Tc yi = y[i];
        const Tc zi = z[i];
        const T hi  = h[i];
        T afxi      = 0;
        T afyi      = 0;
        T afzi      = 0;

        const unsigned nbs = neighborsCount[i];
        for (unsigned nb = 0; nb < nbs; ++nb)
        {
            const unsigned j = neighbors[i * ngmax + nb];
            T xx             = xi - x[j];
            T yy             = yi - y[j];
            T zz             = zi - z[j];
            applyPBC(box, T(2) * hi, xx, yy, zz);
            const T rsq = xx * xx + yy * yy + zz * zz;

            const T r2inv   = 1.0 / rsq;
            const T r6inv   = r2inv * r2inv * r2inv;
            const T forcelj = r6inv * (lj1 * r6inv - lj2);
            const T fpair   = forcelj * r2inv;

            afxi += xx * fpair;
            afyi += yy * fpair;
            afzi += zz * fpair;
        }

        afx[i] = afxi;
        afy[i] = afyi;
        afz[i] = afzi;
    }
}

template<class Tc, class T, class KeyType>
std::tuple<thrust::device_vector<LocalIndex>, thrust::device_vector<unsigned>, OctreeNsView<Tc, KeyType>>
buildNeighborhoodNaiveDirect(std::size_t firstBody,
                             std::size_t lastBody,
                             const Tc* x,
                             const Tc* y,
                             const Tc* z,
                             const T* h,
                             OctreeNsView<Tc, KeyType> tree,
                             const Box<Tc>& box,
                             unsigned ngmax)
{
    thrust::device_vector<LocalIndex> neighbors(ngmax * lastBody);
    thrust::device_vector<unsigned> neighborsCount(lastBody);
    return {neighbors, neighborsCount, tree};
}

template<class Tc, class T, class KeyType>
__global__ void computeLjNaiveDirectKernel(const Tc* __restrict__ x,
                                           const Tc* __restrict__ y,
                                           const Tc* __restrict__ z,
                                           const T* h,
                                           const T lj1,
                                           const T lj2,
                                           const LocalIndex firstId,
                                           const LocalIndex lastId,
                                           const Box<Tc> box,
                                           const OctreeNsView<Tc, KeyType> tree,
                                           unsigned* __restrict__ neighbors,
                                           unsigned* __restrict__ neighborsCount,
                                           const unsigned ngmax,
                                           T* __restrict__ afx,
                                           T* __restrict__ afy,
                                           T* __restrict__ afz)
{
    cstone::LocalIndex tid = blockDim.x * blockIdx.x + threadIdx.x;
    cstone::LocalIndex i   = firstId + tid;
    if (i >= lastId) { return; }

    neighborsCount[i] = findNeighbors(i, x, y, z, h, tree, box, ngmax, neighbors + tid * ngmax);

    const Tc xi = x[i];
    const Tc yi = y[i];
    const Tc zi = z[i];
    const T hi  = h[i];
    T afxi      = 0;
    T afyi      = 0;
    T afzi      = 0;

    const unsigned nbs = imin(neighborsCount[i], ngmax);
    for (unsigned nb = 0; nb < nbs; ++nb)
    {
        const unsigned j = neighbors[i * ngmax + nb];
        T xx             = xi - x[j];
        T yy             = yi - y[j];
        T zz             = zi - z[j];
        applyPBC(box, T(2) * hi, xx, yy, zz);
        const T rsq = xx * xx + yy * yy + zz * zz;

        const T r2inv   = 1.0 / rsq;
        const T r6inv   = r2inv * r2inv * r2inv;
        const T forcelj = r6inv * (lj1 * r6inv - lj2);
        const T fpair   = forcelj * r2inv;

        afxi += xx * fpair;
        afyi += yy * fpair;
        afzi += zz * fpair;
    }

    afx[i] = afxi;
    afy[i] = afyi;
    afz[i] = afzi;
}

template<class Tc, class T, class KeyType>
void computeLjNaiveDirect(
    const std::size_t firstBody,
    const std::size_t lastBody,
    const Tc* x,
    const Tc* y,
    const Tc* z,
    const T* h,
    const T lj1,
    const T lj2,
    const Box<Tc>& box,
    const unsigned ngmax,
    T* __restrict__ afx,
    T* __restrict__ afy,
    T* __restrict__ afz,
    std::tuple<thrust::device_vector<LocalIndex>, thrust::device_vector<unsigned>, OctreeNsView<Tc, KeyType>>&
        neighborhood)
{
    auto& [neighbors, neighborsCount, tree] = neighborhood;
    computeLjNaiveDirectKernel<<<iceil(lastBody - firstBody, 128), 128>>>(x, y, z, h, lj1, lj2, firstBody, lastBody,
                                                                          box, tree, rawPtr(neighbors),
                                                                          rawPtr(neighborsCount), ngmax, afx, afy, afz);
}

template<class Tc, class T, class KeyType>
std::tuple<thrust::device_vector<LocalIndex>,
           thrust::device_vector<unsigned>,
           thrust::device_vector<int>,
           OctreeNsView<Tc, KeyType>>
buildNeighborhoodBatchedDirect(std::size_t firstBody,
                               std::size_t lastBody,
                               const Tc* x,
                               const Tc* y,
                               const Tc* z,
                               const T* h,
                               OctreeNsView<Tc, KeyType> tree,
                               const Box<Tc>& box,
                               unsigned ngmax)
{
    unsigned numBodies = lastBody - firstBody;
    unsigned numBlocks = TravConfig::numBlocks(numBodies);
    unsigned poolSize  = TravConfig::poolSize(numBodies);
    thrust::device_vector<LocalIndex> neighbors(ngmax * numBlocks * (TravConfig::numThreads / GpuConfig::warpSize) *
                                                TravConfig::targetSize);
    thrust::device_vector<unsigned> neighborsCount(lastBody);
    thrust::device_vector<int> globalPool(poolSize);

    return {neighbors, neighborsCount, globalPool, tree};
}

template<class Tc, class T, class KeyType>
__global__
__launch_bounds__(TravConfig::numThreads) void computeLjBatchedDirectKernel(cstone::LocalIndex firstBody,
                                                                            cstone::LocalIndex lastBody,
                                                                            const Tc* __restrict__ x,
                                                                            const Tc* __restrict__ y,
                                                                            const Tc* __restrict__ z,
                                                                            const T* __restrict__ h,
                                                                            const T lj1,
                                                                            const T lj2,
                                                                            const Box<Tc> box,
                                                                            const OctreeNsView<Tc, KeyType> tree,
                                                                            T* __restrict__ afx,
                                                                            T* __restrict__ afy,
                                                                            T* __restrict__ afz,
                                                                            unsigned* __restrict__ neighborsCount,
                                                                            unsigned* __restrict__ neighbors,
                                                                            const unsigned ngmax,
                                                                            int* __restrict__ globalPool)
{
    const unsigned laneIdx     = threadIdx.x & (GpuConfig::warpSize - 1);
    const unsigned numTargets  = (lastBody - firstBody - 1) / TravConfig::targetSize + 1;
    const unsigned warpIdxGrid = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;
    int targetIdx              = 0;

    unsigned* warpNidx = neighbors + warpIdxGrid * TravConfig::targetSize * ngmax;

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&targetCounterGlob, 1); }
        targetIdx = shflSync(targetIdx, 0);

        if (targetIdx >= numTargets) break;

        const cstone::LocalIndex bodyBegin = firstBody + targetIdx * TravConfig::targetSize;
        const cstone::LocalIndex bodyEnd   = imin(bodyBegin + TravConfig::targetSize, lastBody);

        unsigned nc_i[TravConfig::nwt] = {0};

        auto handleInteraction = [&](int warpTarget, cstone::LocalIndex j)
        {
            if (nc_i[warpTarget] < ngmax)
                warpNidx[nc_i[warpTarget] * TravConfig::targetSize + warpTarget * GpuConfig::warpSize + laneIdx] = j;
            ++nc_i[warpTarget];
        };

        traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, handleInteraction, globalPool);

        for (unsigned warpTarget = 0; warpTarget < TravConfig::nwt; ++warpTarget)
        {
            const cstone::LocalIndex i = bodyBegin + warpTarget * GpuConfig::warpSize + laneIdx;
            const unsigned* nidx       = warpNidx + warpTarget * GpuConfig::warpSize + laneIdx;
            if (i < bodyEnd)
            {
                const Tc xi = x[i];
                const Tc yi = y[i];
                const Tc zi = z[i];
                const T hi  = h[i];
                T afxi      = 0;
                T afyi      = 0;
                T afzi      = 0;

                const unsigned nbs = imin(nc_i[warpTarget], ngmax);
                for (unsigned nb = 0; nb < nbs; ++nb)
                {
                    const unsigned j = nidx[nb * TravConfig::targetSize];
                    T xx             = xi - x[j];
                    T yy             = yi - y[j];
                    T zz             = zi - z[j];
                    applyPBC(box, T(2) * hi, xx, yy, zz);
                    const T rsq = xx * xx + yy * yy + zz * zz;

                    const T r2inv   = 1.0 / rsq;
                    const T r6inv   = r2inv * r2inv * r2inv;
                    const T forcelj = r6inv * (lj1 * r6inv - lj2);
                    const T fpair   = forcelj * r2inv;

                    afxi += xx * fpair;
                    afyi += yy * fpair;
                    afzi += zz * fpair;
                }

                afx[i] = afxi;
                afy[i] = afyi;
                afz[i] = afzi;
            }
        }
    }
    cuda::discard_memory(warpNidx, TravConfig::targetSize * ngmax * sizeof(unsigned));
}

template<class Tc, class T, class KeyType>
void computeLjBatchedDirect(const std::size_t firstBody,
                            const std::size_t lastBody,
                            const Tc* __restrict__ x,
                            const Tc* __restrict__ y,
                            const Tc* __restrict__ z,
                            const T* __restrict__ h,
                            const T lj1,
                            const T lj2,
                            const Box<Tc>& box,
                            const unsigned ngmax,
                            T* __restrict__ afx,
                            T* __restrict__ afy,
                            T* __restrict__ afz,
                            std::tuple<thrust::device_vector<LocalIndex>,
                                       thrust::device_vector<unsigned>,
                                       thrust::device_vector<int>,
                                       OctreeNsView<Tc, KeyType>>& neighborhood)
{
    auto& [neighbors, neighborsCount, globalPool, tree] = neighborhood;
    unsigned numBodies                                  = lastBody - firstBody;
    unsigned numBlocks                                  = TravConfig::numBlocks(numBodies);
    resetTraversalCounters<<<1, 1>>>();
    computeLjBatchedDirectKernel<<<numBlocks, TravConfig::numThreads>>>(firstBody, lastBody, x, y, z, h, lj1, lj2, box,
                                                                        tree, afx, afy, afz, rawPtr(neighborsCount),
                                                                        rawPtr(neighbors), ngmax, rawPtr(globalPool));
}

template<class Tc, class T, class KeyType>
__global__ void buildNeighborhoodNaiveKernel(const Tc* x,
                                             const Tc* y,
                                             const Tc* z,
                                             const T* h,
                                             LocalIndex firstId,
                                             LocalIndex lastId,
                                             const Box<Tc> box,
                                             const OctreeNsView<Tc, KeyType> treeView,
                                             unsigned ngmax,
                                             LocalIndex* neighbors,
                                             unsigned* neighborsCount)
{
    cstone::LocalIndex tid = blockDim.x * blockIdx.x + threadIdx.x;
    cstone::LocalIndex id  = firstId + tid;
    if (id >= lastId) { return; }

    neighborsCount[id] = findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors + id, lastId);
}

template<class Tc, class T, class KeyType>
std::tuple<thrust::device_vector<LocalIndex>, thrust::device_vector<unsigned>>
buildNeighborhoodNaive(std::size_t firstBody,
                       std::size_t lastBody,
                       const Tc* x,
                       const Tc* y,
                       const Tc* z,
                       const T* h,
                       OctreeNsView<Tc, KeyType> tree,
                       const Box<Tc>& box,
                       unsigned ngmax)
{
    thrust::device_vector<LocalIndex> neighbors(ngmax * lastBody);
    thrust::device_vector<unsigned> neighborsCount(lastBody);

    buildNeighborhoodNaiveKernel<<<iceil(lastBody - firstBody, 128), 128>>>(
        x, y, z, h, firstBody, lastBody, box, tree, ngmax, rawPtr(neighbors), rawPtr(neighborsCount));

    return {neighbors, neighborsCount};
}

template<class Tc, class T>
__global__ void __maxnreg__(40) computeLjNaiveKernel(const Tc* __restrict__ x,
                                                     const Tc* __restrict__ y,
                                                     const Tc* __restrict__ z,
                                                     const T* __restrict__ h,
                                                     const T lj1,
                                                     const T lj2,
                                                     const LocalIndex firstId,
                                                     const LocalIndex lastId,
                                                     const Box<Tc> box,
                                                     const unsigned* neighbors,
                                                     const unsigned* neighborsCount,
                                                     const unsigned ngmax,
                                                     T* __restrict__ afx,
                                                     T* __restrict__ afy,
                                                     T* __restrict__ afz)
{
    cstone::LocalIndex tid = blockDim.x * blockIdx.x + threadIdx.x;
    cstone::LocalIndex i   = firstId + tid;
    if (i >= lastId) { return; }

    const Tc xi = x[i];
    const Tc yi = y[i];
    const Tc zi = z[i];
    const T hi  = h[i];
    T afxi      = 0;
    T afyi      = 0;
    T afzi      = 0;

    const unsigned nbs = neighborsCount[i];
    for (unsigned nb = 0; nb < nbs; ++nb)
    {
        const unsigned j = neighbors[i + (unsigned long)nb * lastId];
        T xx             = xi - x[j];
        T yy             = yi - y[j];
        T zz             = zi - z[j];
        applyPBC(box, T(2) * hi, xx, yy, zz);
        const T rsq = xx * xx + yy * yy + zz * zz;

        const T r2inv   = T(1) / rsq;
        const T r6inv   = r2inv * r2inv * r2inv;
        const T forcelj = r6inv * (lj1 * r6inv - lj2);
        const T fpair   = forcelj * r2inv;

        afxi += xx * fpair;
        afyi += yy * fpair;
        afzi += zz * fpair;
    }

    afx[i] = afxi;
    afy[i] = afyi;
    afz[i] = afzi;
}

template<class Tc, class T>
void computeLjNaive(const std::size_t firstBody,
                    const std::size_t lastBody,
                    const Tc* __restrict__ x,
                    const Tc* __restrict__ y,
                    const Tc* __restrict__ z,
                    const T* __restrict__ h,
                    const T lj1,
                    const T lj2,
                    const Box<Tc>& box,
                    const unsigned ngmax,
                    T* __restrict__ afx,
                    T* __restrict__ afy,
                    T* __restrict__ afz,
                    const std::tuple<thrust::device_vector<LocalIndex>, thrust::device_vector<unsigned>>& neighborhood)
{
    auto& [neighbors, neighborsCount] = neighborhood;
    computeLjNaiveKernel<<<iceil(lastBody - firstBody, 768), 768>>>(x, y, z, h, lj1, lj2, firstBody, lastBody, box,
                                                                    rawPtr(neighbors), rawPtr(neighborsCount), ngmax,
                                                                    afx, afy, afz);
}

template<class Tc, class Th, class KeyType>
__global__ __launch_bounds__(TravConfig::numThreads) void buildNeighborhoodBatchedKernel(cstone::LocalIndex firstBody,
                                                                                         cstone::LocalIndex lastBody,
                                                                                         const Tc* __restrict__ x,
                                                                                         const Tc* __restrict__ y,
                                                                                         const Tc* __restrict__ z,
                                                                                         const Th* __restrict__ h,
                                                                                         OctreeNsView<Tc, KeyType> tree,
                                                                                         const Box<Tc> box,
                                                                                         unsigned* __restrict__ nc,
                                                                                         unsigned* __restrict__ nidx,
                                                                                         unsigned ngmax,
                                                                                         int* __restrict__ globalPool)
{
    const unsigned laneIdx    = threadIdx.x & (GpuConfig::warpSize - 1);
    const unsigned numTargets = (lastBody - firstBody - 1) / TravConfig::targetSize + 1;
    int targetIdx             = 0;

    while (true)
    {
        if (laneIdx == 0) { targetIdx = atomicAdd(&targetCounterGlob, 1); }
        targetIdx = shflSync(targetIdx, 0);

        if (targetIdx >= numTargets) return;

        const cstone::LocalIndex bodyBegin = firstBody + targetIdx * TravConfig::targetSize;
        const cstone::LocalIndex bodyEnd   = imin(bodyBegin + TravConfig::targetSize, lastBody);

        unsigned nc_i[TravConfig::nwt] = {0};

        auto handleInteraction = [&](int warpTarget, cstone::LocalIndex j)
        {
            if (nc_i[warpTarget] < ngmax)
            {
                const cstone::LocalIndex i = bodyBegin + warpTarget * GpuConfig::warpSize + laneIdx;
                if (i < bodyEnd) nidx[i + (unsigned long)nc_i[warpTarget] * lastBody] = j;
            }
            ++nc_i[warpTarget];
        };

        traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, handleInteraction, globalPool);

        const cstone::LocalIndex bodyIdxLane = bodyBegin + laneIdx;
#pragma unroll
        for (int warpTarget = 0; warpTarget < TravConfig::nwt; warpTarget++)
        {
            const cstone::LocalIndex bodyIdx = bodyIdxLane + warpTarget * GpuConfig::warpSize;
            if (bodyIdx < bodyEnd) { nc[bodyIdx] = nc_i[warpTarget]; }
        }
    }
}

template<class Tc, class T, class KeyType>
std::tuple<thrust::device_vector<LocalIndex>, thrust::device_vector<unsigned>>
buildNeighborhoodBatched(std::size_t firstBody,
                         std::size_t lastBody,
                         const Tc* x,
                         const Tc* y,
                         const Tc* z,
                         const T* h,
                         OctreeNsView<Tc, KeyType> tree,
                         const Box<Tc>& box,
                         unsigned ngmax)
{
    unsigned numBodies = lastBody - firstBody;
    unsigned numBlocks = TravConfig::numBlocks(numBodies);
    unsigned poolSize  = TravConfig::poolSize(numBodies);
    thrust::device_vector<LocalIndex> neighbors(ngmax * lastBody);
    thrust::device_vector<unsigned> neighborsCount(lastBody);
    thrust::device_vector<int> globalPool(poolSize);

    resetTraversalCounters<<<1, 1>>>();
    buildNeighborhoodBatchedKernel<<<numBlocks, TravConfig::numThreads>>>(firstBody, lastBody, x, y, z, h, tree, box,
                                                                          rawPtr(neighborsCount), rawPtr(neighbors),
                                                                          ngmax, rawPtr(globalPool));

    return {neighbors, neighborsCount};
}

template<class Tc, class T>
__global__ __maxnreg__(40) void computeLjBatchedKernel(cstone::LocalIndex firstBody,
                                                       cstone::LocalIndex lastBody,
                                                       const Tc* __restrict__ x,
                                                       const Tc* __restrict__ y,
                                                       const Tc* __restrict__ z,
                                                       const T* __restrict__ h,
                                                       const T lj1,
                                                       const T lj2,
                                                       const Box<Tc> box,
                                                       T* __restrict__ afx,
                                                       T* __restrict__ afy,
                                                       T* __restrict__ afz,
                                                       const unsigned* __restrict__ neighborsCount,
                                                       const unsigned* __restrict__ neighbors,
                                                       const unsigned ngmax)
{
    const unsigned laneIdx    = threadIdx.x & (GpuConfig::warpSize - 1);
    const unsigned numTargets = (lastBody - firstBody - 1) / TravConfig::targetSize + 1;
    int targetIdx             = 0;

    while (true)
    {
        if (laneIdx == 0) { targetIdx = atomicAdd(&targetCounterGlob, 1); }
        targetIdx = shflSync(targetIdx, 0);

        if (targetIdx >= numTargets) return;

        const cstone::LocalIndex bodyBegin = firstBody + targetIdx * TravConfig::targetSize;
#pragma unroll
        for (unsigned warpTarget = 0; warpTarget < TravConfig::nwt; ++warpTarget)
        {
            const cstone::LocalIndex i = bodyBegin + warpTarget * GpuConfig::warpSize + laneIdx;
            if (i < lastBody)
            {
                const Tc xi = x[i];
                const Tc yi = y[i];
                const Tc zi = z[i];
                const T hi  = h[i];
                T afxi      = 0;
                T afyi      = 0;
                T afzi      = 0;

                const unsigned nbs = imin(neighborsCount[i], ngmax);
                for (unsigned nb = 0; nb < nbs; ++nb)
                {
                    const unsigned j = neighbors[i + (unsigned long)nb * lastBody];
                    T xx             = xi - x[j];
                    T yy             = yi - y[j];
                    T zz             = zi - z[j];
                    applyPBC(box, T(2) * hi, xx, yy, zz);
                    const T rsq = xx * xx + yy * yy + zz * zz;

                    const T r2inv   = T(1) / rsq;
                    const T r6inv   = r2inv * r2inv * r2inv;
                    const T forcelj = r6inv * (lj1 * r6inv - lj2);
                    const T fpair   = forcelj * r2inv;

                    afxi += xx * fpair;
                    afyi += yy * fpair;
                    afzi += zz * fpair;
                }

                afx[i] = afxi;
                afy[i] = afyi;
                afz[i] = afzi;
            }
        }
    }
}

template<class Tc, class T>
void computeLjBatched(
    const std::size_t firstBody,
    const std::size_t lastBody,
    const Tc* __restrict__ x,
    const Tc* __restrict__ y,
    const Tc* __restrict__ z,
    const T* __restrict__ h,
    const T lj1,
    const T lj2,
    const Box<Tc>& box,
    const unsigned ngmax,
    T* __restrict__ afx,
    T* __restrict__ afy,
    T* __restrict__ afz,
    const std::tuple<thrust::device_vector<LocalIndex>, thrust::device_vector<unsigned>>& neighborhood)
{
    auto& [neighbors, neighborsCount] = neighborhood;
    constexpr unsigned blocks         = 1 << 10;
    constexpr unsigned threads        = 768;
    resetTraversalCounters<<<1, 1>>>();
    computeLjBatchedKernel<<<blocks, threads>>>(firstBody, lastBody, x, y, z, h, lj1, lj2, box, afx, afy, afz,
                                                rawPtr(neighborsCount), rawPtr(neighbors), ngmax);
}

namespace gromacs_like
{

// Constants from GROMACS

constexpr unsigned clusterSize               = 8;
constexpr unsigned clusterPairSplit          = GpuConfig::warpSize == 64 ? 1 : 2;
constexpr unsigned numClusterPerSupercluster = 8;
constexpr unsigned jGroupSize                = 32 / numClusterPerSupercluster;
constexpr unsigned superClusterSize          = numClusterPerSupercluster * clusterSize;
constexpr unsigned exclSize                  = clusterSize * clusterSize / clusterPairSplit;

// Gromacs-like data structures

struct Sci
{
    unsigned sci, cjPackedBegin, cjPackedEnd;
};

struct Excl
{
    std::array<unsigned, exclSize> pair = {0};

    bool operator==(Excl const& other) const { return pair == other.pair; }
    bool operator<(Excl const& other) const { return pair < other.pair; }
};

struct ImEi
{
    unsigned imask   = 0u;
    unsigned exclInd = 0u;
};

struct CjPacked
{
    std::array<unsigned, jGroupSize> cj;
    std::array<ImEi, clusterPairSplit> imei;
};

// Helpers for building the GROMACS-like NB lists

constexpr bool includeNb(unsigned i, unsigned j)
{
    const unsigned ci  = i / clusterSize;
    const unsigned sci = ci / numClusterPerSupercluster;
    const unsigned cj  = j / clusterSize;
    const unsigned scj = cj / numClusterPerSupercluster;
    return cstone::detail::includeNbSymmetric(sci, scj) || (sci == scj && ci < cj) || (ci == cj && i < j);
}

struct CjData
{
    Excl excl;
    unsigned imask;
};

std::map<unsigned, std::array<CjData, clusterPairSplit>>
clusterNeighborsOfSuperCluster(const thrust::universal_vector<unsigned>& neighbors,
                               const thrust::universal_vector<unsigned>& neighborsCount,
                               const unsigned ngmax,
                               const unsigned sci)
{
    const unsigned long lastBody        = neighborsCount.size();
    constexpr unsigned splitClusterSize = clusterSize / clusterPairSplit;
    std::map<unsigned, std::array<CjData, clusterPairSplit>> superClusterNeighbors;
    for (unsigned cii = 0; cii < numClusterPerSupercluster; ++cii)
    {
        const unsigned ci = sci * numClusterPerSupercluster + cii;
        for (unsigned ii = 0; ii < clusterSize; ++ii)
        {
            const unsigned i   = ci * clusterSize + ii;
            const unsigned nci = std::min(neighborsCount[i], ngmax);
            for (unsigned nb = 0; nb < nci; ++nb)
            {
                const unsigned j = neighbors[i + nb * lastBody];
                if (includeNb(i, j))
                {
                    const unsigned cj    = j / clusterSize;
                    const unsigned jj    = j - cj * clusterSize;
                    auto [it, _]         = superClusterNeighbors.emplace(cj, std::array<CjData, clusterPairSplit>({0}));
                    const unsigned split = jj / splitClusterSize;
                    it->second.at(split).imask |= 1 << cii;
                    const unsigned thread = ii + (jj % splitClusterSize) * clusterSize;
                    it->second.at(split).excl.pair[thread] |= 1 << cii;
                }
            }
        }
    }
    return superClusterNeighbors;
}

void optimizeExcl(const unsigned lastBody,
                  const unsigned sci,
                  const CjPacked& cjPacked,
                  std::array<Excl, clusterPairSplit>& excl)
{
    // Keep exact interaction on last super cluster to avoid OOB accesses
    const unsigned numSuperclusters = iceil(lastBody, superClusterSize);
    if (sci == numSuperclusters - 1) return;

    const unsigned numClusters = iceil(lastBody, clusterSize);
    bool selfInteraction       = false;
    for (unsigned cj : cjPacked.cj)
    {
        // Keep exact interaction on last j-cluster to avoid OOB accesses
        if (cj == numClusters - 1) return;
        if (cj / numClusterPerSupercluster == sci) selfInteraction = true;
    }

    if (!selfInteraction)
    {
        // Compute all interactions if j-clusters do not overlap super cluster
        for (unsigned split = 0; split < clusterPairSplit; ++split)
            excl[split].pair.fill(0xffffffffu);
        return;
    }

    // Use triangular masks (i < j) where j-clusters overlap with super cluster
    for (unsigned jj = 0; jj < clusterSize; ++jj)
    {
        for (unsigned ii = 0; ii < clusterSize; ++ii)
        {
            const unsigned thread = ii + (jj % (clusterSize / clusterPairSplit)) * clusterSize;
            const unsigned split  = jj / (clusterSize / clusterPairSplit);
            for (unsigned jm = 0; jm < jGroupSize; ++jm)
            {
                for (unsigned cii = 0; cii < numClusterPerSupercluster; ++cii)
                {
                    const unsigned ci = sci * numClusterPerSupercluster + cii;
                    const unsigned i  = ci * clusterSize + ii;
                    const unsigned j  = cjPacked.cj[jm] * clusterSize + jj;
                    excl[split].pair[thread] |= includeNb(i, j) ? 1 << (cii + jm * numClusterPerSupercluster) : 0;
                }
            }
        }
    }
}

template<class Tc, class T, class KeyType>
std::tuple<thrust::universal_vector<Sci>, thrust::universal_vector<CjPacked>, thrust::universal_vector<Excl>>
buildNeighborhoodClustered(const std::size_t firstBody,
                           const std::size_t lastBody,
                           const Tc* x,
                           const Tc* y,
                           const Tc* z,
                           const T* h,
                           OctreeNsView<Tc, KeyType> tree,
                           const Box<Tc>& box,
                           unsigned ngmax)
{
    thrust::universal_vector<unsigned> neighbors(ncmax * lastBody);
    thrust::universal_vector<unsigned> neighborsCount(lastBody);

    buildNeighborhoodNaiveKernel<<<iceil(lastBody, 128), 128>>>(x, y, z, h, firstBody, lastBody, box, tree, ngmax,
                                                                rawPtr(neighbors), rawPtr(neighborsCount));
    checkGpuErrors(cudaDeviceSynchronize());

    const unsigned numSuperclusters = iceil(lastBody, superClusterSize);
    const unsigned numClusters      = iceil(lastBody, clusterSize);

    thrust::universal_vector<Sci> sciSorted(numSuperclusters);
    thrust::universal_vector<CjPacked> cjPacked;
    cjPacked.reserve(thrust::reduce(neighborsCount.begin(), neighborsCount.end(), 0u, std::plus()) / jGroupSize);
    thrust::universal_vector<Excl> excl(1);
    excl.front().pair.fill(0xffffffffu);

    std::map<Excl, unsigned> exclIndexMap;
    exclIndexMap[excl.front()] = 0;

    std::shared_mutex exclMutex, cjPackedMutex;

    const auto exclIndex = [&](Excl const& e)
    {
        {
            std::shared_lock lock(exclMutex);
            const auto it = exclIndexMap.find(e);
            if (it != exclIndexMap.end()) return it->second;
        }
        {
            std::unique_lock lock(exclMutex);
            const auto it = exclIndexMap.find(e);
            if (it != exclIndexMap.end()) return it->second;
            const unsigned index = excl.size();
            excl.push_back(e);
            exclIndexMap[e] = index;
            return index;
        }
    };

#pragma omp parallel for shared(sciSorted, cjPacked, excl, exclIndexMap, exclMutex, cjPackedMutex, neighbors,          \
                                    neighborsCount)
    for (unsigned sci = 0; sci < numSuperclusters; ++sci)
    {
        const auto superClusterNeighbors = clusterNeighborsOfSuperCluster(neighbors, neighborsCount, ngmax, sci);

        const unsigned ncjPacked = iceil(superClusterNeighbors.size(), jGroupSize);
        auto it                  = superClusterNeighbors.begin();
        unsigned cjPackedBegin, cjPackedEnd;
        {
            std::unique_lock lock(cjPackedMutex);
            cjPackedBegin = cjPacked.size();
            cjPackedEnd   = cjPackedBegin + ncjPacked;
            cjPacked.resize(cjPackedEnd);
        }
        for (unsigned n = 0; n < ncjPacked; ++n)
        {
            CjPacked next                               = {0};
            std::array<Excl, clusterPairSplit> nextExcl = {0};
            for (unsigned jm = 0; jm < jGroupSize; ++jm, ++it)
            {
                if (it == superClusterNeighbors.end()) break;

                const auto& [cj, data] = *it;
                next.cj[jm]            = cj;
                for (unsigned split = 0; split < clusterPairSplit; ++split)
                {
                    next.imei[split].imask |= data[split].imask << (jm * numClusterPerSupercluster);
                    for (unsigned e = 0; e < exclSize; ++e)
                        nextExcl[split].pair[e] |= data[split].excl.pair[e] << (jm * numClusterPerSupercluster);
                }
            }
            optimizeExcl(lastBody, sci, next, nextExcl);
            for (unsigned split = 0; split < clusterPairSplit; ++split)
                next.imei[split].exclInd = exclIndex(nextExcl[split]);
            {
                std::shared_lock lock(cjPackedMutex);
                cjPacked[cjPackedBegin + n] = next;
            }
        }

        sciSorted[sci] = {sci, cjPackedBegin, cjPackedEnd};
    }
    // for (auto const& sci : sciSorted)
    // printf("sci: %d, cjPackedBegin: %d, cjPackedEnd: %d\n", sci.sci, sci.cjPackedBegin, sci.cjPackedEnd);
    // for (auto const& cj : cjPacked)
    // printf("[%5d, %5d, %5d, %5d], %08x %08x\n", cj.cj[0], cj.cj[1], cj.cj[2], cj.cj[3], cj.imei[0].imask,
    // cj.imei[1].imask);

    return {sciSorted, cjPacked, excl};
}

template<class Tc, class T>
__global__
__launch_bounds__(clusterSize* clusterSize) void computeLjClusteredKernel(cstone::LocalIndex firstBody,
                                                                          cstone::LocalIndex lastBody,
                                                                          const Tc* __restrict__ x,
                                                                          const Tc* __restrict__ y,
                                                                          const Tc* __restrict__ z,
                                                                          const T* __restrict__ h,
                                                                          const T lj1,
                                                                          const T lj2,
                                                                          const Box<Tc> box,
                                                                          T* __restrict__ afx,
                                                                          T* __restrict__ afy,
                                                                          T* __restrict__ afz,
                                                                          const Sci* __restrict__ sciSorted,
                                                                          const CjPacked* __restrict__ cjPacked,
                                                                          const Excl* __restrict__ excl)
{
    namespace cg = cooperative_groups;

    constexpr unsigned superClusterInteractionMask = (1u << numClusterPerSupercluster) - 1u;

    const auto block = cg::this_thread_block();
    const auto warp  = cg::tiled_partition<GpuConfig::warpSize>(block);

    const Sci nbSci               = sciSorted[block.group_index().x];
    const unsigned sci            = nbSci.sci;
    const unsigned cijPackedBegin = nbSci.cjPackedBegin;
    const unsigned cijPackedEnd   = nbSci.cjPackedEnd;
    T* const af[3]                = {afx, afy, afz};

    using T3  = std::conditional_t<std::is_same_v<T, double>, double3, float3>;
    using Tc3 = std::conditional_t<std::is_same_v<Tc, double>, double3, float3>;
    using Tc4 = std::conditional_t<std::is_same_v<T, double>, double4, float4>;

    __shared__ Tc4 xqib[clusterSize * numClusterPerSupercluster];

    constexpr bool loadUsingAllXYThreads = clusterSize == numClusterPerSupercluster;
    if (loadUsingAllXYThreads || block.thread_index().y < numClusterPerSupercluster)
    {
        const unsigned ci = sci * numClusterPerSupercluster + block.thread_index().y;
        const unsigned ai = ci * clusterSize + block.thread_index().x;
        xqib[block.thread_index().y * clusterSize + block.thread_index().x] = {x[ai], y[ai], z[ai], Tc(h[ai])};
    }
    block.sync();

    T3 fciBuf[numClusterPerSupercluster];
    for (unsigned i = 0; i < numClusterPerSupercluster; ++i)
        fciBuf[i] = {T(0), T(0), T(0)};

    for (unsigned jPacked = cijPackedBegin; jPacked < cijPackedEnd; ++jPacked)
    {
        const unsigned wexclIdx = cjPacked[jPacked].imei[warp.meta_group_rank()].exclInd;
        const unsigned imask    = cjPacked[jPacked].imei[warp.meta_group_rank()].imask;
        const unsigned wexcl    = excl[wexclIdx].pair[warp.thread_rank()];
        if (imask)
        {
            for (unsigned jm = 0; jm < jGroupSize; ++jm)
            {
                if (imask & (superClusterInteractionMask << (jm * numClusterPerSupercluster)))
                {
                    unsigned maskJi   = 1u << (jm * numClusterPerSupercluster);
                    const unsigned cj = cjPacked[jPacked].cj[jm];
                    const unsigned aj = cj * clusterSize + block.thread_index().y;
                    const Tc3 xj      = {x[aj], y[aj], z[aj]};
                    T3 fcjBuf         = {T(0), T(0), T(0)};

#pragma unroll
                    for (unsigned i = 0; i < numClusterPerSupercluster; ++i)
                    {
                        if (imask & maskJi)
                        {
                            const unsigned ci = sci * numClusterPerSupercluster + i;
                            const unsigned ai = ci * clusterSize + block.thread_index().x;
                            if (ai < lastBody)
                            {
                                const Tc4 xi = xqib[i * clusterSize + block.thread_index().x];
                                const T hi   = xi.w;
                                T3 rv        = {T(xi.x - xj.x), T(xi.y - xj.y), T(xi.z - xj.z)};
                                applyPBC(box, T(2) * hi, rv.x, rv.y, rv.z);
                                const T r2 = rv.x * rv.x + rv.y * rv.y + rv.z * rv.z;
                                const T intBit = (wexcl & maskJi) ? T(1) : T(0);
                                if ((r2 < 4 * hi * hi) * intBit)
                                {
                                    const T rinv    = rsqrt(r2);
                                    const T r2inv   = rinv * rinv;
                                    const T r6inv   = r2inv * r2inv * r2inv;
                                    const T forcelj = r6inv * (lj1 * r6inv - lj2);
                                    const T fpair   = forcelj * r2inv;
                                    const T3 fij    = {rv.x * fpair, rv.y * fpair, rv.z * fpair};
                                    fcjBuf.x -= fij.x;
                                    fcjBuf.y -= fij.y;
                                    fcjBuf.z -= fij.z;
                                    fciBuf[i].x += fij.x;
                                    fciBuf[i].y += fij.y;
                                    fciBuf[i].z += fij.z;
                                }
                            }
                        }
                        maskJi += maskJi;
                    }

                    fcjBuf.x += warp.shfl_down(fcjBuf.x, 1);
                    fcjBuf.y += warp.shfl_up(fcjBuf.y, 1);
                    fcjBuf.z += warp.shfl_down(fcjBuf.z, 1);
                    if (block.thread_index().x & 1) fcjBuf.x = fcjBuf.y;
                    fcjBuf.x += warp.shfl_down(fcjBuf.x, 2);
                    fcjBuf.z += warp.shfl_up(fcjBuf.z, 2);
                    if (block.thread_index().x & 2) fcjBuf.x = fcjBuf.z;
                    fcjBuf.x += warp.shfl_down(fcjBuf.x, 4);
                    if (block.thread_index().x < 3 && aj < lastBody)
                        atomicAdd(af[block.thread_index().x] + aj, fcjBuf.x);
                }
            }
        }
    }

    for (unsigned i = 0; i < numClusterPerSupercluster; ++i)
    {
        const unsigned ai = (sci * numClusterPerSupercluster + i) * clusterSize + block.thread_index().x;
        fciBuf[i].x += warp.shfl_down(fciBuf[i].x, clusterSize);
        fciBuf[i].y += warp.shfl_up(fciBuf[i].y, clusterSize);
        fciBuf[i].z += warp.shfl_down(fciBuf[i].z, clusterSize);
        if (block.thread_index().y & 1) fciBuf[i].x = fciBuf[i].y;
        fciBuf[i].x += warp.shfl_down(fciBuf[i].x, 2 * clusterSize);
        fciBuf[i].z += warp.shfl_up(fciBuf[i].z, 2 * clusterSize);
        if (block.thread_index().y & 2) fciBuf[i].x = fciBuf[i].z;
        if ((block.thread_index().y & 3) < 3 && ai < lastBody)
            atomicAdd(af[block.thread_index().y & 3] + ai, fciBuf[i].x);
    }
}

template<class Tc, class T>
void computeLjClustered(
    const std::size_t firstBody,
    const std::size_t lastBody,
    const Tc* __restrict__ x,
    const Tc* __restrict__ y,
    const Tc* __restrict__ z,
    const T* __restrict__ h,
    const T lj1,
    const T lj2,
    const Box<Tc>& box,
    const unsigned ngmax,
    T* __restrict__ afx,
    T* __restrict__ afy,
    T* __restrict__ afz,
    const std::tuple<thrust::universal_vector<Sci>, thrust::universal_vector<CjPacked>, thrust::universal_vector<Excl>>&
        neighborhood)
{
    auto& [sciSorted, cjPacked, excl] = neighborhood;

    checkGpuErrors(cudaMemsetAsync(afx, 0, sizeof(T) * lastBody));
    checkGpuErrors(cudaMemsetAsync(afy, 0, sizeof(T) * lastBody));
    checkGpuErrors(cudaMemsetAsync(afz, 0, sizeof(T) * lastBody));

    const dim3 blockSize     = {clusterSize, clusterSize, 1};
    const unsigned numBlocks = sciSorted.size();
    computeLjClusteredKernel<<<numBlocks, blockSize>>>(firstBody, lastBody, x, y, z, h, lj1, lj2, box, afx, afy, afz,
                                                       rawPtr(sciSorted), rawPtr(cjPacked), rawPtr(excl));
    checkGpuErrors(cudaGetLastError());
}

} // namespace gromacs_like

template<bool Compress, bool Symmetric, class Tc, class T, class KeyType>
std::tuple<thrust::device_vector<LocalIndex>, thrust::device_vector<unsigned>>
buildNeighborhoodClustered(std::size_t firstBody,
                           std::size_t lastBody,
                           const Tc* x,
                           const Tc* y,
                           const Tc* z,
                           const T* h,
                           OctreeNsView<Tc, KeyType> tree,
                           const Box<Tc>& box,
                           unsigned ngmax)
{
    unsigned numBodies                           = lastBody - firstBody;
    unsigned numBlocks                           = TravConfig::numBlocks(numBodies);
    unsigned poolSize                            = TravConfig::poolSize(numBodies);
    std::size_t iClusters                        = iceil(lastBody, ClusterConfig::iSize);
    constexpr unsigned long nbStoragePerICluster = Compress ? ncmax / ClusterConfig::expectedCompressionRate : ncmax;
    thrust::device_vector<LocalIndex> clusterNeighbors(nbStoragePerICluster * iClusters);
    thrust::device_vector<unsigned> clusterNeighborsCount;
    if constexpr (!Compress) clusterNeighborsCount.resize(iClusters);
    thrust::device_vector<int> globalPool(poolSize);

    constexpr unsigned threads       = Compress ? 64 : 32;
    constexpr unsigned warpsPerBlock = threads / GpuConfig::warpSize;
    dim3 blockSize = {ClusterConfig::iSize, GpuConfig::warpSize / ClusterConfig::iSize, warpsPerBlock};

    resetTraversalCounters<<<1, 1>>>();
    findClusterNeighbors<warpsPerBlock, true, true, ncmax, Compress, Symmetric>
        <<<numBlocks, blockSize>>>(firstBody, lastBody, x, y, z, h, tree, box, rawPtr(clusterNeighborsCount),
                                   rawPtr(clusterNeighbors), rawPtr(globalPool));
    checkGpuErrors(cudaGetLastError());

    return {clusterNeighbors, clusterNeighborsCount};
}

template<bool Compress, bool Symmetric, class Tc, class T>
void computeLjClustered(
    const std::size_t firstBody,
    const std::size_t lastBody,
    const Tc* __restrict__ x,
    const Tc* __restrict__ y,
    const Tc* __restrict__ z,
    const T* __restrict__ h,
    const T lj1,
    const T lj2,
    const Box<Tc>& box,
    const unsigned ngmax,
    T* __restrict__ afx,
    T* __restrict__ afy,
    T* __restrict__ afz,
    const std::tuple<thrust::device_vector<LocalIndex>, thrust::device_vector<unsigned>>& neighborhood)
{
    auto& [clusterNeighbors, clusterNeighborsCount] = neighborhood;

    if constexpr (Symmetric)
    {
        checkGpuErrors(cudaMemsetAsync(afx, 0, sizeof(T) * lastBody));
        checkGpuErrors(cudaMemsetAsync(afy, 0, sizeof(T) * lastBody));
        checkGpuErrors(cudaMemsetAsync(afz, 0, sizeof(T) * lastBody));
    }

    resetTraversalCounters<<<1, 1>>>();
    auto computeLj = [=] __device__(unsigned i, auto iPos, T hi, unsigned j, auto jPos, auto ijPosDiff, T rsq)
    {
        const T r2inv   = T(1) / rsq;
        const T r6inv   = r2inv * r2inv * r2inv;
        const T forcelj = r6inv * (lj1 * r6inv - lj2);
        const T fpair   = i == j ? 0 : forcelj * r2inv;

        return std::make_tuple(ijPosDiff[0] * fpair, ijPosDiff[1] * fpair, ijPosDiff[2] * fpair);
    };

    constexpr unsigned threads       = Compress ? 256 : 256;
    constexpr unsigned warpsPerBlock = threads / GpuConfig::warpSize;
    const dim3 blockSize     = {ClusterConfig::iSize, GpuConfig::warpSize / ClusterConfig::iSize, warpsPerBlock};
    const unsigned numBlocks = 1 << 11;
    findNeighborsClustered<warpsPerBlock, true, true, ncmax, Compress, Symmetric ? -1 : 0>
        <<<numBlocks, blockSize>>>(firstBody, lastBody, x, y, z, h, box, rawPtr(clusterNeighborsCount),
                                   rawPtr(clusterNeighbors), computeLj, afx, afy, afz);
    checkGpuErrors(cudaGetLastError());
}

template<class Tc, class T, class StrongKeyType, class BuildNeighborhoodF, class ComputeLjF>
void benchmarkGPU(BuildNeighborhoodF buildNeighborhood, ComputeLjF computeLj)
{
    using KeyType = typename StrongKeyType::ValueType;

    constexpr int nx = 200;
    Box<Tc> box{0, 1.6795962 * nx, BoundaryType::periodic};

    FaceCenteredCubicCoordinates<Tc, StrongKeyType> coords(nx, nx, nx, box);
    const int n = coords.x().size();
    std::vector<T> h(n, 3.8 / 2);
    printf("Number of atoms: %d\n", n);

    const int ngmax = 256;

    const Tc* x       = coords.x().data();
    const Tc* y       = coords.y().data();
    const Tc* z       = coords.z().data();
    const auto* codes = (KeyType*)(coords.particleKeys().data());

    unsigned bucketSize   = 64;
    auto [csTree, counts] = computeOctree(codes, codes + n, bucketSize);
    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(csTree));
    updateInternalTree<KeyType>(csTree, octree.data());
    const TreeNodeIndex* childOffsets = octree.childOffsets.data();
    const TreeNodeIndex* toLeafOrder  = octree.internalToLeaf.data();

    std::vector<LocalIndex> layout(nNodes(csTree) + 1);
    std::exclusive_scan(counts.begin(), counts.end() + 1, layout.begin(), 0);

    std::vector<Vec3<Tc>> centers(octree.numNodes), sizes(octree.numNodes);
    gsl::span<const KeyType> nodeKeys(octree.prefixes.data(), octree.numNodes);
    nodeFpCenters<KeyType>(nodeKeys, centers.data(), sizes.data(), box);

    OctreeNsView<Tc, KeyType> nsView{octree.prefixes.data(),
                                     octree.childOffsets.data(),
                                     octree.internalToLeaf.data(),
                                     octree.levelRange.data(),
                                     layout.data(),
                                     centers.data(),
                                     sizes.data()};

    std::vector<T> afx(n), afy(n), afz(n);
    auto neighborhoodCPU = buildNeighborhoodCPU(0, n, x, y, z, h.data(), nsView, box, ngmax);
    printf("Number of neighbors: %d\n", std::get<1>(neighborhoodCPU).at(0));
    const T lj1 = 48;
    const T lj2 = 24;
    computeLjCPU(0, n, x, y, z, h.data(), lj1, lj2, box, ngmax, afx.data(), afy.data(), afz.data(), neighborhoodCPU);

    thrust::device_vector<Tc> d_x(coords.x().begin(), coords.x().end());
    thrust::device_vector<Tc> d_y(coords.y().begin(), coords.y().end());
    thrust::device_vector<Tc> d_z(coords.z().begin(), coords.z().end());
    thrust::device_vector<T> d_h = h;
    thrust::device_vector<T> d_afx(n, std::numeric_limits<T>::quiet_NaN());
    thrust::device_vector<T> d_afy(n, std::numeric_limits<T>::quiet_NaN());
    thrust::device_vector<T> d_afz(n, std::numeric_limits<T>::quiet_NaN());

    thrust::device_vector<KeyType> d_prefixes             = octree.prefixes;
    thrust::device_vector<TreeNodeIndex> d_childOffsets   = octree.childOffsets;
    thrust::device_vector<TreeNodeIndex> d_internalToLeaf = octree.internalToLeaf;
    thrust::device_vector<TreeNodeIndex> d_levelRange     = octree.levelRange;
    thrust::device_vector<LocalIndex> d_layout            = layout;
    thrust::device_vector<Vec3<Tc>> d_centers             = centers;
    thrust::device_vector<Vec3<Tc>> d_sizes               = sizes;

    OctreeNsView<Tc, KeyType> nsViewGpu{rawPtr(d_prefixes),   rawPtr(d_childOffsets), rawPtr(d_internalToLeaf),
                                        rawPtr(d_levelRange), rawPtr(d_layout),       rawPtr(d_centers),
                                        rawPtr(d_sizes)};

    thrust::device_vector<KeyType> d_codes(coords.particleKeys().begin(), coords.particleKeys().end());
    const auto* deviceKeys = (const KeyType*)(rawPtr(d_codes));

    auto neighborhoodGPU =
        buildNeighborhood(0, n, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_h), nsViewGpu, box, ngmax);

    std::array<float, 5> times;
    std::array<cudaEvent_t, times.size() + 1> events;
    for (auto& event : events)
        cudaEventCreate(&event);
    cudaEventRecord(events[0]);
    for (std::size_t i = 1; i < events.size(); ++i)
    {
        computeLj(0, n, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_h), lj1, lj2, box, ngmax, rawPtr(d_afx),
                  rawPtr(d_afy), rawPtr(d_afz), neighborhoodGPU);
        cudaEventRecord(events[i]);
    }
    cudaEventSynchronize(events.back());

    for (std::size_t i = 0; i < times.size(); ++i)
    {
        cudaEventElapsedTime(&times[i], events[i], events[i + 1]);
        cudaEventDestroy(events[i]);
    }

    printf("GPU times [s]: ");
    for (auto t : times)
        printf("%7.6fs ", t / 1000);
    printf("\n");
    printf("Gatom-step/s: ");
    for (auto t : times)
        printf("%7.6fs ", n / 1.0e6 / t);
    printf("\n");

    std::vector<T> fxGPU(n), fyGPU(n), fzGPU(n), afxGPU(n), afyGPU(n), afzGPU(n);
    thrust::copy(d_afx.begin(), d_afx.end(), afxGPU.begin());
    thrust::copy(d_afy.begin(), d_afy.end(), afyGPU.begin());
    thrust::copy(d_afz.begin(), d_afz.end(), afzGPU.begin());

    int numFails = 0;
    auto isclose = [](double a, double b)
    {
        double atol = 1e-6;
        double rtol = 1e-5;
        return std::abs(a - b) <= atol + rtol * std::abs(b);
    };
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
        if (!isclose(afxGPU[i], afx[i]) || !isclose(afyGPU[i], afy[i]) || !isclose(afzGPU[i], afz[i]))
        {
            int failNum;
#pragma omp atomic capture
            failNum = numFails++;
            if (failNum < 10)
            {
#pragma omp critical
                printf("%i (%.10f, %.10f, %.10f) (%.10f, %.10f, %.10f)\n", i, afxGPU[i], afyGPU[i], afzGPU[i], afx[i],
                       afy[i], afz[i]);
            }
        }
    }
    std::cout << "numFails: " << numFails << std::endl;
}

int main()
{
    using Tc            = double;
    using T             = double;
    using StrongKeyType = HilbertKey<uint64_t>;
    using KeyType       = typename StrongKeyType::ValueType;

    std::cout << "--- NAIVE DIRECT ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(buildNeighborhoodNaiveDirect<Tc, T, KeyType>,
                                       computeLjNaiveDirect<Tc, T, KeyType>);
    std::cout << "--- BATCHED DIRECT ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(buildNeighborhoodBatchedDirect<Tc, T, KeyType>,
                                       computeLjBatchedDirect<Tc, T, KeyType>);

    std::cout << "--- NAIVE TWO-STAGE ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(buildNeighborhoodNaive<Tc, T, KeyType>, computeLjNaive<Tc, T>);
    std::cout << "--- BATCHED TWO-STAGE ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(buildNeighborhoodBatched<Tc, T, KeyType>, computeLjBatched<Tc, T>);
    std::cout << "--- GROMACS CLUSTERED TWO-STAGE ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(gromacs_like::buildNeighborhoodClustered<Tc, T, KeyType>,
                                       gromacs_like::computeLjClustered<Tc, T>);
    std::cout << "--- CLUSTERED TWO-STAGE ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(buildNeighborhoodClustered<false, false, Tc, T, KeyType>,
                                       computeLjClustered<false, false, Tc, T>);
    std::cout << "--- COMPRESSED CLUSTERED TWO-STAGE ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(buildNeighborhoodClustered<true, false, Tc, T, KeyType>,
                                       computeLjClustered<true, false, Tc, T>);
    std::cout << "--- CLUSTERED TWO-STAGE SYMMETRIC ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(buildNeighborhoodClustered<false, true, Tc, T, KeyType>,
                                       computeLjClustered<false, true, Tc, T>);
    std::cout << "--- COMPRESSED CLUSTERED TWO-STAGE SYMMETRIC ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(buildNeighborhoodClustered<true, true, Tc, T, KeyType>,
                                       computeLjClustered<true, true, Tc, T>);

    return 0;
}
