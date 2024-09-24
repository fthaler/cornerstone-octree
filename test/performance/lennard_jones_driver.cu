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
 * @brief  SPH density kernel with various neighbor search strategies
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#include <iostream>

#include <cuda/annotated_ptr>

#include <thrust/device_vector.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/findneighbors.hpp"

#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/traversal/find_neighbors_clustered.cuh"

#include "../coord_samples/face_centered_cubic.hpp"

using namespace cstone;

constexpr unsigned ncmax = 512;

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
                  T* __restrict__ fx,
                  T* __restrict__ fy,
                  T* __restrict__ fz,
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
        T fxi       = 0;
        T fyi       = 0;
        T fzi       = 0;

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

            fxi += xx * fpair;
            fyi += yy * fpair;
            fzi += zz * fpair;
        }

        // printf("%.10g %.10g %.10g\n", fxi, fyi, fzi);
        fx[i] = fxi;
        fy[i] = fyi;
        fz[i] = fzi;
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
                                           T* __restrict__ fx,
                                           T* __restrict__ fy,
                                           T* __restrict__ fz)
{
    cstone::LocalIndex tid = blockDim.x * blockIdx.x + threadIdx.x;
    cstone::LocalIndex i   = firstId + tid;
    if (i >= lastId) { return; }

    neighborsCount[i] = findNeighbors(i, x, y, z, h, tree, box, ngmax, neighbors + tid * ngmax);

    const Tc xi = x[i];
    const Tc yi = y[i];
    const Tc zi = z[i];
    const T hi  = h[i];
    T fxi       = 0;
    T fyi       = 0;
    T fzi       = 0;

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

        fxi += xx * fpair;
        fyi += yy * fpair;
        fzi += zz * fpair;
    }

    fx[i] = fxi;
    fy[i] = fyi;
    fz[i] = fzi;
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
    T* fx,
    T* fy,
    T* fz,
    std::tuple<thrust::device_vector<LocalIndex>, thrust::device_vector<unsigned>, OctreeNsView<Tc, KeyType>>&
        neighborhood)
{
    auto& [neighbors, neighborsCount, tree] = neighborhood;
    computeLjNaiveDirectKernel<<<iceil(lastBody - firstBody, 128), 128>>>(x, y, z, h, lj1, lj2, firstBody, lastBody,
                                                                          box, tree, rawPtr(neighbors),
                                                                          rawPtr(neighborsCount), ngmax, fx, fy, fz);
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
                                                                            T* __restrict__ fx,
                                                                            T* __restrict__ fy,
                                                                            T* __restrict__ fz,
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
                T fxi       = 0;
                T fyi       = 0;
                T fzi       = 0;

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

                    fxi += xx * fpair;
                    fyi += yy * fpair;
                    fzi += zz * fpair;
                }

                fx[i] = fxi;
                fy[i] = fyi;
                fz[i] = fzi;
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
                            T* __restrict__ fx,
                            T* __restrict__ fy,
                            T* __restrict__ fz,
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
                                                                        tree, fx, fy, fz, rawPtr(neighborsCount),
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
__global__ void computeLjNaiveKernel(const Tc* __restrict__ x,
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
                                     T* fx,
                                     T* fy,
                                     T* fz)
{
    cstone::LocalIndex tid = blockDim.x * blockIdx.x + threadIdx.x;
    cstone::LocalIndex i   = firstId + tid;
    if (i >= lastId) { return; }

    const Tc xi = x[i];
    const Tc yi = y[i];
    const Tc zi = z[i];
    const T hi  = h[i];
    T fxi       = 0;
    T fyi       = 0;
    T fzi       = 0;

    const unsigned nbs = neighborsCount[i];
    for (unsigned nb = 0; nb < nbs; ++nb)
    {
        const unsigned j = neighbors[i + (unsigned long)nb * lastId];
        T xx             = xi - x[j];
        T yy             = yi - y[j];
        T zz             = zi - z[j];
        applyPBC(box, T(2) * hi, xx, yy, zz);
        const T rsq = xx * xx + yy * yy + zz * zz;

        const T r2inv   = 1.0 / rsq;
        const T r6inv   = r2inv * r2inv * r2inv;
        const T forcelj = r6inv * (lj1 * r6inv - lj2);
        const T fpair   = forcelj * r2inv;

        fxi += xx * fpair;
        fyi += yy * fpair;
        fzi += zz * fpair;
    }

    fx[i] = fxi;
    fy[i] = fyi;
    fz[i] = fzi;
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
                    T* __restrict__ fx,
                    T* __restrict__ fy,
                    T* __restrict__ fz,
                    const std::tuple<thrust::device_vector<LocalIndex>, thrust::device_vector<unsigned>>& neighborhood)
{
    auto& [neighbors, neighborsCount] = neighborhood;
    computeLjNaiveKernel<<<iceil(lastBody - firstBody, 128), 128>>>(
        x, y, z, h, lj1, lj2, firstBody, lastBody, box, rawPtr(neighbors), rawPtr(neighborsCount), ngmax, fx, fy, fz);
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
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&targetCounterGlob, 1); }
        targetIdx = shflSync(targetIdx, 0);

        if (targetIdx >= numTargets) return;

        const cstone::LocalIndex bodyBegin = firstBody + targetIdx * TravConfig::targetSize;
        const cstone::LocalIndex bodyEnd   = imin(bodyBegin + TravConfig::targetSize, lastBody);
        unsigned* warpNidx                 = nidx + targetIdx * TravConfig::targetSize * ngmax;

        unsigned nc_i[TravConfig::nwt] = {0};

        auto handleInteraction = [&](int warpTarget, cstone::LocalIndex j)
        {
            if (nc_i[warpTarget] < ngmax)
                warpNidx[nc_i[warpTarget] * TravConfig::targetSize + laneIdx + warpTarget * GpuConfig::warpSize] = j;
            ++nc_i[warpTarget];
        };

        traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, handleInteraction, globalPool);

        const cstone::LocalIndex bodyIdxLane = bodyBegin + laneIdx;
        for (int i = 0; i < TravConfig::nwt; i++)
        {
            const cstone::LocalIndex bodyIdx = bodyIdxLane + i * GpuConfig::warpSize;
            if (bodyIdx < bodyEnd) { nc[bodyIdx] = nc_i[i]; }
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
__global__
__launch_bounds__(TravConfig::numThreads) void computeLjBatchedKernel(cstone::LocalIndex firstBody,
                                                                      cstone::LocalIndex lastBody,
                                                                      const Tc* __restrict__ x,
                                                                      const Tc* __restrict__ y,
                                                                      const Tc* __restrict__ z,
                                                                      const T* __restrict__ h,
                                                                      const T lj1,
                                                                      const T lj2,
                                                                      const Box<Tc> box,
                                                                      T* __restrict__ fx,
                                                                      T* __restrict__ fy,
                                                                      T* __restrict__ fz,
                                                                      const unsigned* __restrict__ neighborsCount,
                                                                      const unsigned* __restrict__ neighbors,
                                                                      const unsigned ngmax)
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

        const cstone::LocalIndex bodyBegin = firstBody + targetIdx * TravConfig::targetSize;
        const cstone::LocalIndex bodyEnd   = imin(bodyBegin + TravConfig::targetSize, lastBody);
        for (unsigned warpTarget = 0; warpTarget < TravConfig::nwt; ++warpTarget)
        {
            const cstone::LocalIndex i = bodyBegin + warpTarget * GpuConfig::warpSize + laneIdx;
            const unsigned* nidx =
                neighbors + targetIdx * TravConfig::targetSize * ngmax + warpTarget * GpuConfig::warpSize + laneIdx;
            if (i < bodyEnd)
            {
                const Tc xi = x[i];
                const Tc yi = y[i];
                const Tc zi = z[i];
                const T hi  = h[i];
                T fxi       = 0;
                T fyi       = 0;
                T fzi       = 0;

                const unsigned nbs = imin(neighborsCount[i], ngmax);
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

                    fxi += xx * fpair;
                    fyi += yy * fpair;
                    fzi += zz * fpair;
                }

                fx[i] = fxi;
                fy[i] = fyi;
                fz[i] = fzi;
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
    T* __restrict__ fx,
    T* __restrict__ fy,
    T* __restrict__ fz,
    const std::tuple<thrust::device_vector<LocalIndex>, thrust::device_vector<unsigned>>& neighborhood)
{
    auto& [neighbors, neighborsCount] = neighborhood;
    unsigned numBodies                = lastBody - firstBody;
    unsigned numBlocks                = TravConfig::numBlocks(numBodies);
    resetTraversalCounters<<<1, 1>>>();
    computeLjBatchedKernel<<<numBlocks, TravConfig::numThreads>>>(
        firstBody, lastBody, x, y, z, h, lj1, lj2, box, fx, fy, fz, rawPtr(neighborsCount), rawPtr(neighbors), ngmax);
}

template<class Tc, class T, class KeyType>
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
    unsigned numBodies    = lastBody - firstBody;
    unsigned numBlocks    = TravConfig::numBlocks(numBodies);
    unsigned poolSize     = TravConfig::poolSize(numBodies);
    std::size_t iClusters = iceil(lastBody, ClusterConfig::iSize);
    thrust::device_vector<LocalIndex> clusterNeighbors(ncmax * iClusters);
    thrust::device_vector<unsigned> clusterNeighborsCount(iClusters);
    thrust::device_vector<int> globalPool(poolSize);

    // TODO: own traversal config for cluster kernels
    constexpr unsigned threads       = 128;
    constexpr unsigned warpsPerBlock = threads / GpuConfig::warpSize;
    dim3 blockSize = {ClusterConfig::iSize, GpuConfig::warpSize / ClusterConfig::iSize, warpsPerBlock};

    resetTraversalCounters<<<1, 1>>>();
    findClusterNeighbors9<warpsPerBlock, true, true, ncmax, false>
        <<<numBlocks, blockSize>>>(firstBody, lastBody, x, y, z, h, tree, box, rawPtr(clusterNeighborsCount),
                                   rawPtr(clusterNeighbors), rawPtr(globalPool));
    checkGpuErrors(cudaGetLastError());

    return {clusterNeighbors, clusterNeighborsCount};
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
    T* __restrict__ fx,
    T* __restrict__ fy,
    T* __restrict__ fz,
    const std::tuple<thrust::device_vector<LocalIndex>, thrust::device_vector<unsigned>>& neighborhood)
{
    auto& [clusterNeighbors, clusterNeighborsCount] = neighborhood;
    unsigned numBodies                              = lastBody - firstBody;
    unsigned numBlocks                              = TravConfig::numBlocks(numBodies);

    resetTraversalCounters<<<1, 1>>>();
    auto computeLj = [=] __device__(unsigned i, auto iPos, T hi, unsigned j, auto jPos, auto ijPosDiff, T rsq)
    {
        const T r2inv   = T(1) / rsq;
        const T r6inv   = r2inv * r2inv * r2inv;
        const T forcelj = r6inv * (lj1 * r6inv - lj2);
        const T fpair   = i == j ? 0 : forcelj * r2inv;

        return std::make_tuple(ijPosDiff[0] * fpair, ijPosDiff[1] * fpair, ijPosDiff[2] * fpair);
    };

    constexpr unsigned threads       = 512;
    constexpr unsigned warpsPerBlock = threads / GpuConfig::warpSize;
    dim3 blockSize                   = {ClusterConfig::iSize, ClusterConfig::jSize, warpsPerBlock};
    numBlocks                        = 1 << 11;
    findNeighborsClustered8<warpsPerBlock, true, ncmax, false>
        <<<numBlocks, blockSize>>>(firstBody, lastBody, x, y, z, h, box, rawPtr(clusterNeighborsCount),
                                   rawPtr(clusterNeighbors), computeLj, fx, fy, fz);
    checkGpuErrors(cudaGetLastError());
}

template<class Tc, class T, class KeyType>
thrust::device_vector<LocalIndex> buildNeighborhoodCompressedClustered(std::size_t firstBody,
                                                                       std::size_t lastBody,
                                                                       const Tc* x,
                                                                       const Tc* y,
                                                                       const Tc* z,
                                                                       const T* h,
                                                                       OctreeNsView<Tc, KeyType> tree,
                                                                       const Box<Tc>& box,
                                                                       unsigned ngmax)
{
    unsigned numBodies    = lastBody - firstBody;
    unsigned numBlocks    = TravConfig::numBlocks(numBodies);
    unsigned poolSize     = TravConfig::poolSize(numBodies);
    std::size_t iClusters = iceil(lastBody, ClusterConfig::iSize);
    thrust::device_vector<LocalIndex> clusterNeighbors((ncmax / ClusterConfig::expectedCompressionRate) * iClusters);
    thrust::device_vector<int> globalPool(poolSize);

    constexpr unsigned threads       = 64;
    constexpr unsigned warpsPerBlock = threads / GpuConfig::warpSize;
    dim3 blockSize = {ClusterConfig::iSize, GpuConfig::warpSize / ClusterConfig::iSize, warpsPerBlock};

    resetTraversalCounters<<<1, 1>>>();
    findClusterNeighbors9<warpsPerBlock, true, true, ncmax, true><<<numBlocks, blockSize>>>(
        firstBody, lastBody, x, y, z, h, tree, box, nullptr, rawPtr(clusterNeighbors), rawPtr(globalPool));
    checkGpuErrors(cudaGetLastError());

    return clusterNeighbors;
}

template<class Tc, class T>
void computeLjCompressedClustered(const std::size_t firstBody,
                                  const std::size_t lastBody,
                                  const Tc* __restrict__ x,
                                  const Tc* __restrict__ y,
                                  const Tc* __restrict__ z,
                                  const T* __restrict__ h,
                                  const T lj1,
                                  const T lj2,
                                  const Box<Tc>& box,
                                  const unsigned ngmax,
                                  T* __restrict__ fx,
                                  T* __restrict__ fy,
                                  T* __restrict__ fz,
                                  const thrust::device_vector<LocalIndex>& clusterNeighbors)
{
    unsigned numBodies = lastBody - firstBody;
    unsigned numBlocks = TravConfig::numBlocks(numBodies);

    resetTraversalCounters<<<1, 1>>>();
    auto computeLj = [=] __device__(unsigned i, auto iPos, T hi, unsigned j, auto jPos, auto ijPosDiff, T rsq)
    {
        const T r2inv   = 1.0 / rsq;
        const T r6inv   = r2inv * r2inv * r2inv;
        const T forcelj = r6inv * (lj1 * r6inv - lj2);
        const T fpair   = i == j ? 0 : forcelj * r2inv;

        return std::make_tuple(ijPosDiff[0] * fpair, ijPosDiff[1] * fpair, ijPosDiff[2] * fpair);
    };

    constexpr unsigned threads       = 256;
    constexpr unsigned warpsPerBlock = threads / GpuConfig::warpSize;
    dim3 blockSize                   = {ClusterConfig::iSize, ClusterConfig::jSize, warpsPerBlock};
    numBlocks                        = 1 << 11;
    findNeighborsClustered8<warpsPerBlock, true, ncmax, true><<<numBlocks, blockSize>>>(
        firstBody, lastBody, x, y, z, h, box, nullptr, rawPtr(clusterNeighbors), computeLj, fx, fy, fz);
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
    std::vector<T> h(n, 2.8 / 2);
    printf("%d\n", n);

    int ngmax = 256;

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

    std::vector<T> fx(n), fy(n), fz(n);
    auto neighborhoodCPU = buildNeighborhoodCPU(0, n, x, y, z, h.data(), nsView, box, ngmax);
    printf("Number of neighbors: %d\n", std::get<1>(neighborhoodCPU).at(0));
    const T lj1 = 48;
    const T lj2 = 24;
    computeLjCPU(0, n, x, y, z, h.data(), lj1, lj2, box, ngmax, fx.data(), fy.data(), fz.data(), neighborhoodCPU);

    thrust::device_vector<Tc> d_x(coords.x().begin(), coords.x().end());
    thrust::device_vector<Tc> d_y(coords.y().begin(), coords.y().end());
    thrust::device_vector<Tc> d_z(coords.z().begin(), coords.z().end());
    thrust::device_vector<T> d_h = h;
    thrust::device_vector<T> d_fx(n);
    thrust::device_vector<T> d_fy(n);
    thrust::device_vector<T> d_fz(n);

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

    std::array<float, 21> times;
    std::array<cudaEvent_t, times.size() + 1> events;
    for (auto& event : events)
        cudaEventCreate(&event);
    cudaEventRecord(events[0]);
    for (std::size_t i = 1; i < events.size(); ++i)
    {
        computeLj(0, n, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_h), lj1, lj2, box, ngmax, rawPtr(d_fx),
                  rawPtr(d_fy), rawPtr(d_fz), neighborhoodGPU);
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

    std::vector<T> fxGPU(n), fyGPU(n), fzGPU(n);
    thrust::copy(d_fx.begin(), d_fx.end(), fxGPU.begin());
    thrust::copy(d_fy.begin(), d_fy.end(), fyGPU.begin());
    thrust::copy(d_fz.begin(), d_fz.end(), fzGPU.begin());

    int numFails = 0;
    auto isclose = [](double a, double b)
    {
        double atol = 1e-6;
        double rtol = 1e-5;
        return std::abs(a - b) <= atol + rtol * std::abs(b);
    };
    for (int i = 0; i < n; ++i)
    {
        if (!isclose(fxGPU[i], fx[i]) || !isclose(fyGPU[i], fy[i]) || !isclose(fzGPU[i], fz[i]))
        {
            printf("%i (%.10f, %.10f, %.10f) (%.10f, %.10f, %.10f)\n", i, fxGPU[i], fyGPU[i], fzGPU[i], fx[i], fy[i],
                   fz[i]);
            ++numFails;
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
    std::cout << "--- CLUSTERED TWO-STAGE ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(buildNeighborhoodClustered<Tc, T, KeyType>, computeLjClustered<Tc, T>);
    std::cout << "--- COMPRESSED CLUSTERED TWO-STAGE ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(buildNeighborhoodCompressedClustered<Tc, T, KeyType>,
                                       computeLjCompressedClustered<Tc, T>);

    return 0;
}
