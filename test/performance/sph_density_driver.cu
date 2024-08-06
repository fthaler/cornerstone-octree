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

#include "../coord_samples/random.hpp"

using namespace cstone;

/* smoothing kernel evaluation functionality borrowed from SPH-EXA */

constexpr int kTableSize                 = 20000;
constexpr bool kUseTable                 = false;
constexpr bool kUseCacheResidencyControl = false;

template<typename T>
__host__ __device__ inline T wharmonic_std(T v)
{
    if (v == 0.0) { return 1.0; }

    const T Pv = T(M_PI_2) * v;
    return std::sin(Pv) / Pv;
}

template<class T, std::size_t N, class F>
std::array<T, N> tabulateFunction(F&& func, double lowerSupport, double upperSupport)
{
    constexpr int numIntervals = N - 1;
    std::array<T, N> table;

    const T dx = (upperSupport - lowerSupport) / numIntervals;
    for (size_t i = 0; i < N; ++i)
    {
        T normalizedVal = lowerSupport + i * dx;
        table[i]        = func(normalizedVal);
    }

    return table;
}

template<class T>
std::array<T, kTableSize> kernelTable()
{
    return tabulateFunction<T, kTableSize>([](T x) { return std::pow(wharmonic_std(x), 6.0); }, 0.0, 2.0);
}

template<bool useKernelTable = kUseTable, class T>
__host__ __device__ inline T table_lookup(const T* table, T v)
{
    if constexpr (useKernelTable)
    {
        constexpr int numIntervals = kTableSize - 1;
        constexpr T support        = 2.0;
        constexpr T dx             = support / numIntervals;
        constexpr T invDx          = T(1) / dx;

        int idx = v * invDx;

        T derivative = (idx >= numIntervals) ? 0.0 : (table[idx + 1] - table[idx]) * invDx;
        return (idx >= numIntervals) ? 0.0 : table[idx] + derivative * (v - T(idx) * dx);
    }
    else
    {
        T w  = wharmonic_std(v);
        T w2 = w * w;
        return w2 * w2 * w2;
    }
}

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

#pragma omp parallel for
    for (std::size_t i = firstBody; i < lastBody; ++i)
    {
        neighborsCount[i] = findNeighbors(i, x, y, z, h, tree, box, ngmax, neighbors.data() + i * ngmax);
    }

    return {neighbors, neighborsCount};
}

template<class Tc, class T, class Tm>
void computeDensityCPU(const std::size_t firstBody,
                       const std::size_t lastBody,
                       const Tc* x,
                       const Tc* y,
                       const Tc* z,
                       const T* h,
                       const Tm* m,
                       const Box<Tc>& box,
                       const unsigned ngmax,
                       const T* wh,
                       T* rho,
                       const std::tuple<std::vector<LocalIndex>, std::vector<unsigned>>& neighborhood)
{
    auto& [neighbors, neighborsCount] = neighborhood;
#pragma omp parallel for
    for (std::size_t i = firstBody; i < lastBody; ++i)
    {
        const Tc xi  = x[i];
        const Tc yi  = y[i];
        const Tc zi  = z[i];
        const T hi   = h[i];
        const T mi   = m[i];
        const T hInv = 1.0 / hi;

        unsigned nbs = neighborsCount[i];
        T rhoi       = mi;
        for (unsigned nb = 0; nb < nbs; ++nb)
        {
            unsigned j = neighbors[i * ngmax + nb];
            T dist     = distancePBC(box, hi, xi, yi, zi, x[j], y[j], z[j]);
            T vloc     = dist * hInv;
            T w        = table_lookup<true>(wh, vloc);

            rhoi += w * m[j];
        }

        rho[i] = rhoi;
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
__global__ void computeDensityNaiveDirectKernel(const Tc* x,
                                                const Tc* y,
                                                const Tc* z,
                                                const T* h,
                                                const T* m,
                                                const T* wh,
                                                const LocalIndex firstId,
                                                const LocalIndex lastId,
                                                const Box<Tc> box,
                                                const OctreeNsView<Tc, KeyType> tree,
                                                unsigned* neighbors,
                                                unsigned* neighborsCount,
                                                const unsigned ngmax,
                                                T* rho)
{
    cstone::LocalIndex tid = blockDim.x * blockIdx.x + threadIdx.x;
    cstone::LocalIndex i   = firstId + tid;
    if (i >= lastId) { return; }

    neighborsCount[i] = findNeighbors(i, x, y, z, h, tree, box, ngmax, neighbors + tid * ngmax);

    const Tc xi  = x[i];
    const Tc yi  = y[i];
    const Tc zi  = z[i];
    const T hi   = h[i];
    const T mi   = m[i];
    const T hInv = 1.0 / hi;

    unsigned nbs = imin(neighborsCount[i], ngmax);
    T rhoi       = mi;
    for (unsigned nb = 0; nb < nbs; ++nb)
    {
        unsigned j = neighbors[i * ngmax + nb];
        T dist     = distancePBC(box, hi, xi, yi, zi, x[j], y[j], z[j]);
        T vloc     = dist * hInv;
        T w        = table_lookup(wh, vloc);

        rhoi += w * m[j];
    }

    rho[i] = rhoi;
}

template<class Tc, class T, class KeyType>
void computeDensityNaiveDirect(
    const std::size_t firstBody,
    const std::size_t lastBody,
    const Tc* x,
    const Tc* y,
    const Tc* z,
    const T* h,
    const T* m,
    const Box<Tc>& box,
    const unsigned ngmax,
    const T* wh,
    T* rho,
    std::tuple<thrust::device_vector<LocalIndex>, thrust::device_vector<unsigned>, OctreeNsView<Tc, KeyType>>&
        neighborhood)
{
    auto& [neighbors, neighborsCount, tree] = neighborhood;
    computeDensityNaiveDirectKernel<<<iceil(lastBody - firstBody, 128), 128>>>(
        x, y, z, h, m, wh, firstBody, lastBody, box, tree, rawPtr(neighbors), rawPtr(neighborsCount), ngmax, rho);
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
__launch_bounds__(TravConfig::numThreads) void computeDensityBatchedDirectKernel(cstone::LocalIndex firstBody,
                                                                                 cstone::LocalIndex lastBody,
                                                                                 const Tc* __restrict__ x,
                                                                                 const Tc* __restrict__ y,
                                                                                 const Tc* __restrict__ z,
                                                                                 const T* __restrict__ h,
                                                                                 const T* __restrict__ m,
                                                                                 const Box<Tc> box,
                                                                                 const OctreeNsView<Tc, KeyType> tree,
                                                                                 const T* __restrict__ wh,
                                                                                 T* __restrict__ rho,
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
                const Tc xi  = x[i];
                const Tc yi  = y[i];
                const Tc zi  = z[i];
                const T hi   = h[i];
                const T mi   = m[i];
                const T hInv = 1.0 / hi;

                unsigned nbs = imin(nc_i[warpTarget], ngmax);
                T rhoi       = mi;
                for (unsigned nb = 0; nb < nbs; ++nb)
                {
                    unsigned j = nidx[nb * TravConfig::targetSize];
                    T dist     = distancePBC(box, hi, xi, yi, zi, x[j], y[j], z[j]);
                    T vloc     = dist * hInv;
                    T w        = table_lookup(wh, vloc);

                    rhoi += w * m[j];
                }

                rho[i] = rhoi;
            }
        }
    }
    cuda::discard_memory(warpNidx, TravConfig::targetSize * ngmax * sizeof(unsigned));
}

template<class Tc, class T, class KeyType>
void computeDensityBatchedDirect(const std::size_t firstBody,
                                 const std::size_t lastBody,
                                 const Tc* x,
                                 const Tc* y,
                                 const Tc* z,
                                 const T* h,
                                 const T* m,
                                 const Box<Tc>& box,
                                 const unsigned ngmax,
                                 const T* wh,
                                 T* rho,
                                 std::tuple<thrust::device_vector<LocalIndex>,
                                            thrust::device_vector<unsigned>,
                                            thrust::device_vector<int>,
                                            OctreeNsView<Tc, KeyType>>& neighborhood)
{
    auto& [neighbors, neighborsCount, globalPool, tree] = neighborhood;
    unsigned numBodies                                  = lastBody - firstBody;
    unsigned numBlocks                                  = TravConfig::numBlocks(numBodies);
    resetTraversalCounters<<<1, 1>>>();
    computeDensityBatchedDirectKernel<<<numBlocks, TravConfig::numThreads>>>(
        firstBody, lastBody, x, y, z, h, m, box, tree, wh, rho, rawPtr(neighborsCount), rawPtr(neighbors), ngmax,
        rawPtr(globalPool));
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

    neighborsCount[id] = findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors + tid * ngmax);
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
__global__ void computeDensityNaiveKernel(const Tc* x,
                                          const Tc* y,
                                          const Tc* z,
                                          const T* h,
                                          const T* m,
                                          const T* wh,
                                          const LocalIndex firstId,
                                          const LocalIndex lastId,
                                          const Box<Tc> box,
                                          const unsigned* neighbors,
                                          const unsigned* neighborsCount,
                                          const unsigned ngmax,
                                          T* rho)
{
    cstone::LocalIndex tid = blockDim.x * blockIdx.x + threadIdx.x;
    cstone::LocalIndex i   = firstId + tid;
    if (i >= lastId) { return; }

    const Tc xi  = x[i];
    const Tc yi  = y[i];
    const Tc zi  = z[i];
    const T hi   = h[i];
    const T mi   = m[i];
    const T hInv = 1.0 / hi;

    unsigned nbs = imin(neighborsCount[i], ngmax);
    T rhoi       = mi;
    for (unsigned nb = 0; nb < nbs; ++nb)
    {
        unsigned j = neighbors[i * ngmax + nb];
        T dist     = distancePBC(box, hi, xi, yi, zi, x[j], y[j], z[j]);
        T vloc     = dist * hInv;
        T w        = table_lookup(wh, vloc);

        rhoi += w * m[j];
    }

    rho[i] = rhoi;
}

template<class Tc, class T>
void computeDensityNaive(
    const std::size_t firstBody,
    const std::size_t lastBody,
    const Tc* x,
    const Tc* y,
    const Tc* z,
    const T* h,
    const T* m,
    const Box<Tc>& box,
    const unsigned ngmax,
    const T* wh,
    T* rho,
    const std::tuple<thrust::device_vector<LocalIndex>, thrust::device_vector<unsigned>>& neighborhood)
{
    auto& [neighbors, neighborsCount] = neighborhood;
    computeDensityNaiveKernel<<<iceil(lastBody - firstBody, 128), 128>>>(
        x, y, z, h, m, wh, firstBody, lastBody, box, rawPtr(neighbors), rawPtr(neighborsCount), ngmax, rho);
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
__launch_bounds__(TravConfig::numThreads) void computeDensityBatchedKernel(cstone::LocalIndex firstBody,
                                                                           cstone::LocalIndex lastBody,
                                                                           const Tc* __restrict__ x,
                                                                           const Tc* __restrict__ y,
                                                                           const Tc* __restrict__ z,
                                                                           const T* __restrict__ h,
                                                                           const T* __restrict__ m,
                                                                           const Box<Tc> box,
                                                                           const T* __restrict__ wh,
                                                                           T* __restrict__ rho,
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
                const Tc xi  = x[i];
                const Tc yi  = y[i];
                const Tc zi  = z[i];
                const T hi   = h[i];
                const T mi   = m[i];
                const T hInv = 1 / hi;

                unsigned nbs = imin(neighborsCount[i], ngmax);
                T rhoi       = mi;
                for (unsigned nb = 0; nb < nbs; ++nb)
                {
                    const unsigned j = nidx[nb * TravConfig::targetSize];
                    const T dist     = distancePBC(box, hi, xi, yi, zi, x[j], y[j], z[j]);
                    const T vloc     = dist * hInv;
                    const T w        = table_lookup(wh, vloc);

                    rhoi += w * m[j];
                }

                rho[i] = rhoi;
            }
        }
    }
}

template<class Tc, class T>
void computeDensityBatched(
    const std::size_t firstBody,
    const std::size_t lastBody,
    const Tc* x,
    const Tc* y,
    const Tc* z,
    const T* h,
    const T* m,
    const Box<Tc>& box,
    const unsigned ngmax,
    const T* wh,
    T* rho,
    const std::tuple<thrust::device_vector<LocalIndex>, thrust::device_vector<unsigned>>& neighborhood)
{
    auto& [neighbors, neighborsCount] = neighborhood;
    unsigned numBodies                = lastBody - firstBody;
    unsigned numBlocks                = TravConfig::numBlocks(numBodies);
    resetTraversalCounters<<<1, 1>>>();
    computeDensityBatchedKernel<<<numBlocks, TravConfig::numThreads>>>(
        firstBody, lastBody, x, y, z, h, m, box, wh, rho, rawPtr(neighborsCount), rawPtr(neighbors), ngmax);
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
    unsigned ncmax        = ngmax;
    unsigned numBodies    = lastBody - firstBody;
    unsigned numBlocks    = TravConfig::numBlocks(numBodies);
    unsigned poolSize     = TravConfig::poolSize(numBodies);
    std::size_t iClusters = iceil(lastBody, ClusterConfig::iSize);
    thrust::device_vector<LocalIndex> clusterNeighbors(ncmax * iClusters);
    thrust::device_vector<unsigned> clusterNeighborsCount(iClusters);
    thrust::device_vector<int> globalPool(poolSize);

    resetTraversalCounters<<<1, 1>>>();
    findClusterNeighbors4<<<numBlocks, TravConfig::numThreads>>>(firstBody, lastBody, x, y, z, h, tree, box,
                                                                 rawPtr(clusterNeighborsCount),
                                                                 rawPtr(clusterNeighbors), ncmax, rawPtr(globalPool));

    return {clusterNeighbors, clusterNeighborsCount};
}

template<class Tc, class T>
void computeDensityClustered(
    const std::size_t firstBody,
    const std::size_t lastBody,
    const Tc* __restrict__ x,
    const Tc* __restrict__ y,
    const Tc* __restrict__ z,
    const T* __restrict__ h,
    const T* __restrict__ m,
    const Box<Tc>& box,
    const unsigned ngmax,
    const T* __restrict__ wh,
    T* __restrict__ rho,
    const std::tuple<thrust::device_vector<LocalIndex>, thrust::device_vector<unsigned>>& neighborhood)
{
    auto& [clusterNeighbors, clusterNeighborsCount] = neighborhood;
    unsigned numBodies                              = lastBody - firstBody;
    unsigned numBlocks                              = TravConfig::numBlocks(numBodies);

    resetTraversalCounters<<<1, 1>>>();
    auto computeDensity = [=] __device__(unsigned i, auto iPos, T hi, unsigned j, auto jPos, T dist)
    {
        T mj = m[j];
        if (i == j) return mj;
        const T vloc = dist * (1 / hi);
        const T w    = table_lookup(wh, vloc);
        return i == j ? mj : w * mj;
    };

    unsigned ncmax = ngmax;
    dim3 blockSize = {ClusterConfig::iSize, ClusterConfig::jSize, 512 / GpuConfig::warpSize};
    numBlocks      = 1 << 11;
    findNeighborsClustered8<512 / GpuConfig::warpSize>
        <<<numBlocks, blockSize>>>(firstBody, lastBody, x, y, z, h, box, rawPtr(clusterNeighborsCount),
                                   rawPtr(clusterNeighbors), ncmax, computeDensity, rho);
}

template<class Tc, class T, class KeyType>
std::tuple<thrust::device_vector<LocalIndex>, thrust::device_vector<unsigned>>
buildNeighborhoodCompressedClustered(std::size_t firstBody,
                                     std::size_t lastBody,
                                     const Tc* x,
                                     const Tc* y,
                                     const Tc* z,
                                     const T* h,
                                     OctreeNsView<Tc, KeyType> tree,
                                     const Box<Tc>& box,
                                     unsigned ngmax)
{
    auto [clusterNeighbors, clusterNeighborsCount] =
        buildNeighborhoodClustered(firstBody, lastBody, x, y, z, h, tree, box, ngmax);

    unsigned ncmax        = ngmax;
    std::size_t iClusters = iceil(lastBody, ClusterConfig::iSize);
    compressClusterNeighbors<<<1024, 1024>>>(iClusters, rawPtr(clusterNeighborsCount), rawPtr(clusterNeighbors), ncmax);

    return {clusterNeighbors, clusterNeighborsCount};
}

template<class Tc, class T>
void computeDensityCompressedClustered(
    const std::size_t firstBody,
    const std::size_t lastBody,
    const Tc* __restrict__ x,
    const Tc* __restrict__ y,
    const Tc* __restrict__ z,
    const T* __restrict__ h,
    const T* __restrict__ m,
    const Box<Tc>& box,
    const unsigned ngmax,
    const T* __restrict__ wh,
    T* __restrict__ rho,
    const std::tuple<thrust::device_vector<LocalIndex>, thrust::device_vector<unsigned>>& neighborhood)
{
    auto& [clusterNeighbors, clusterNeighborsCount] = neighborhood;
    unsigned numBodies                              = lastBody - firstBody;
    unsigned numBlocks                              = TravConfig::numBlocks(numBodies);

    resetTraversalCounters<<<1, 1>>>();
    auto computeDensity = [=] __device__(unsigned i, auto iPos, T hi, unsigned j, auto jPos, T dist)
    {
        T mj         = m[j];
        const T vloc = dist * (1 / hi);
        const T w    = table_lookup(wh, vloc);
        return i == j ? mj : w * mj;
    };

    unsigned ncmax = ngmax;
    dim3 blockSize = {ClusterConfig::iSize, ClusterConfig::jSize, 512 / GpuConfig::warpSize};
    numBlocks      = 1 << 11;
    findNeighborsClustered10<512 / GpuConfig::warpSize>
        <<<numBlocks, blockSize>>>(firstBody, lastBody, x, y, z, h, box, rawPtr(clusterNeighborsCount),
                                   rawPtr(clusterNeighbors), ncmax, computeDensity, rho);
}

template<class Tc, class T, class StrongKeyType, class BuildNeighborhoodF, class ComputeDensityF>
void benchmarkGPU(BuildNeighborhoodF buildNeighborhood, ComputeDensityF computeDensity)
{
    using KeyType = typename StrongKeyType::ValueType;

    Box<Tc> box{0, 1, BoundaryType::periodic};
    int n = 2000000;

    RandomCoordinates<Tc, StrongKeyType> coords(n, box);
    std::vector<T> h(n, 0.012);

    const double r                  = 2 * h[0];
    const double expected_neighbors = 4.0 / 3.0 * M_PI * r * r * r * n;
    std::cout << "Expected average number of neighbors: " << expected_neighbors << std::endl;

    // RandomGaussianCoordinates<T, StrongKeyType> coords(n, box);
    // adjustSmoothingLength<KeyType>(n, 100, 200, coords.x(), coords.y(), coords.z(), h, box);

    int ngmax = 200;

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

    std::vector<T> rho(n), m(n, 1.0);
    auto wh              = kernelTable<T>();
    auto neighborhoodCPU = buildNeighborhoodCPU(0, n, x, y, z, h.data(), nsView, box, ngmax);
    computeDensityCPU(0, n, x, y, z, h.data(), m.data(), box, ngmax, wh.data(), rho.data(), neighborhoodCPU);

    thrust::device_vector<Tc> d_x(coords.x().begin(), coords.x().end());
    thrust::device_vector<Tc> d_y(coords.y().begin(), coords.y().end());
    thrust::device_vector<Tc> d_z(coords.z().begin(), coords.z().end());
    thrust::device_vector<T> d_h = h;
    thrust::device_vector<T> d_m = m;
    thrust::device_vector<T> d_rho(n);
    thrust::device_vector<T> d_wh(wh.size());
    thrust::copy(wh.begin(), wh.end(), d_wh.begin());

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

    cudaStreamAttrValue streamAttr;
    if constexpr (kUseTable && kUseCacheResidencyControl)
    {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        int maxPersistentSize = std::min(int(prop.l2CacheSize * 0.5), prop.persistingL2CacheMaxSize);
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, maxPersistentSize);

        streamAttr.accessPolicyWindow.num_bytes =
            std::min(prop.accessPolicyMaxWindowSize, int(sizeof(T) * d_wh.size()));
        streamAttr.accessPolicyWindow.hitRatio = 1.0;
        streamAttr.accessPolicyWindow.hitProp  = cudaAccessPropertyPersisting;
        streamAttr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

        cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &streamAttr);
    }

    std::array<float, 5> times;
    std::array<cudaEvent_t, times.size() + 1> events;
    for (auto& event : events)
        cudaEventCreate(&event);
    cudaEventRecord(events[0]);
    for (std::size_t i = 1; i < events.size(); ++i)
    {
        computeDensity(0, n, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_h), rawPtr(d_m), box, ngmax, rawPtr(d_wh),
                       rawPtr(d_rho), neighborhoodGPU);
        cudaEventRecord(events[i]);
    }
    cudaEventSynchronize(events.back());

    if constexpr (kUseTable && kUseCacheResidencyControl)
    {
        streamAttr.accessPolicyWindow.num_bytes = 0;
        cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &streamAttr);
        cudaCtxResetPersistingL2Cache();
    }

    for (std::size_t i = 0; i < times.size(); ++i)
    {
        cudaEventElapsedTime(&times[i], events[i], events[i + 1]);
        cudaEventDestroy(events[i]);
    }

    printf("GPU times: ");
    for (auto t : times)
        printf("%7.6fs ", t / 1000);
    printf("\n");

    std::vector<T> rhoGPU(n);
    thrust::copy(d_rho.begin(), d_rho.end(), rhoGPU.begin());

    int numFails = 0;
    auto isclose = [](double a, double b)
    {
        double atol = 0.0;
        double rtol = 1e-5;
        return std::abs(a - b) <= atol + rtol * std::abs(b);
    };
    for (int i = 0; i < n; ++i)
    {
        if (!isclose(rhoGPU[i], rho[i]))
        {
            printf("%i %.10f %.10f\n", i, rhoGPU[i], rho[i]);
            ++numFails;
        }
    }
    std::cout << "numFails: " << numFails << std::endl;
}

int main()
{
    using Tc            = double;
    using T             = float;
    using StrongKeyType = HilbertKey<uint64_t>;
    using KeyType       = typename StrongKeyType::ValueType;

    std::cout << "--- NAIVE DIRECT ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(buildNeighborhoodNaiveDirect<Tc, T, KeyType>,
                                       computeDensityNaiveDirect<Tc, T, KeyType>);
    std::cout << "--- BATCHED DIRECT ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(buildNeighborhoodBatchedDirect<Tc, T, KeyType>,
                                       computeDensityBatchedDirect<Tc, T, KeyType>);

    std::cout << "--- NAIVE TWO-STAGE ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(buildNeighborhoodNaive<Tc, T, KeyType>, computeDensityNaive<Tc, T>);
    std::cout << "--- BATCHED TWO-STAGE ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(buildNeighborhoodBatched<Tc, T, KeyType>, computeDensityBatched<Tc, T>);
    std::cout << "--- CLUSTERED TWO-STAGE ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(buildNeighborhoodClustered<Tc, T, KeyType>, computeDensityClustered<Tc, T>);
    std::cout << "--- COMPRESSED CLUSTERED TWO-STAGE ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(buildNeighborhoodCompressedClustered<Tc, T, KeyType>,
                                       computeDensityCompressedClustered<Tc, T>);

    return 0;
}
