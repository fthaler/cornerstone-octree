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
 * @brief Neighbor search on GPU
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <algorithm>
#include <utility>
#include <vector>

#include "cstone/traversal/ijloop/ijloop.hpp"
#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/tree/octree.hpp"

namespace cstone::ijloop
{

namespace detail
{

template<bool UsePbc, class Tc, class Th, class KeyType, class In, class Out, class Interaction>
__global__ __launch_bounds__(TravConfig::numThreads) void gpuAlwaysTraverseNeighborhoodKernel(
    const OctreeNsView<Tc, KeyType> tree, // TODO: __grid_constant__?
    const Box<Tc> box,                    // TODO: __grid_constant__?
    const LocalIndex firstBody,
    const LocalIndex lastBody,
    const Tc* __restrict__ x,
    const Tc* __restrict__ y,
    const Tc* __restrict__ z,
    const Th* __restrict__ h,
    const In input,
    const Out output,
    const Interaction interaction,
    const unsigned ngmax,
    LocalIndex* __restrict__ neighbors,
    int* __restrict__ globalPool)
{
    const unsigned laneIdx     = threadIdx.x & (GpuConfig::warpSize - 1);
    const unsigned numTargets  = (lastBody - firstBody - 1) / TravConfig::targetSize + 1;
    const unsigned warpIdxGrid = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;
    int targetIdx              = 0;

    unsigned* warpNidx = neighbors + warpIdxGrid * TravConfig::targetSize * ngmax;

    while (true)
    {
        if (laneIdx == 0) targetIdx = atomicAdd(&targetCounterGlob, 1);
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

#pragma unroll
        for (unsigned warpTarget = 0; warpTarget < TravConfig::nwt; ++warpTarget)
        {
            const cstone::LocalIndex i = bodyBegin + warpTarget * GpuConfig::warpSize + laneIdx;
            const LocalIndex* nidx     = warpNidx + warpTarget * GpuConfig::warpSize + laneIdx;
            if (i < bodyEnd)
            {
                const auto iData = loadParticleData(x, y, z, h, input, i);

                const unsigned nbs = imin(nc_i[warpTarget], ngmax);
                auto result        = interaction(iData, iData, Tc(0));
                for (unsigned nb = 0; nb < nbs; ++nb)
                {
                    const LocalIndex j = nidx[nb * TravConfig::targetSize];
                    const auto jData   = loadParticleData(x, y, z, h, input, j);
                    const Tc distSq    = distanceSquared(UsePbc, box, iData, jData);

                    updateResult(result, interaction(iData, jData, distSq));
                }

                storeParticleData(output, i, result);
            }
        }
    }
    cuda::discard_memory(warpNidx, TravConfig::targetSize * ngmax * sizeof(LocalIndex));
}

template<class Tc, class KeyType, class Th>
struct GpuAlwaysTraverseNeighborhoodImpl
{
    OctreeNsView<Tc, KeyType> tree;
    Box<Tc> box;
    LocalIndex firstBody, lastBody;
    const Tc *x, *y, *z;
    const Th* h;
    unsigned ngmax;
    thrust::device_vector<LocalIndex> neighbors;
    thrust::device_vector<int> globalPool;

    template<class... In, class... Out, class Interaction, class Symmetry = symmetry::Asymmetric>
    void ijLoop(std::tuple<const In*...> const& input,
                std::tuple<Out*...> const& output,
                Interaction&& interaction,
                Symmetry = symmetry::asymmetric)
    {
        resetTraversalCounters<<<1, 1>>>();
        if (box.boundaryX() == BoundaryType::periodic | box.boundaryY() == BoundaryType::periodic |
            box.boundaryZ() == BoundaryType::periodic)
        {
            gpuAlwaysTraverseNeighborhoodKernel<true><<<TravConfig::numBlocks(), TravConfig::numThreads>>>(
                tree, box, firstBody, lastBody, x, y, z, h, input, output, std::forward<Interaction>(interaction),
                ngmax, rawPtr(neighbors), rawPtr(globalPool));
        }
        else
        {
            gpuAlwaysTraverseNeighborhoodKernel<false><<<TravConfig::numBlocks(), TravConfig::numThreads>>>(
                tree, box, firstBody, lastBody, x, y, z, h, input, output, std::forward<Interaction>(interaction),
                ngmax, rawPtr(neighbors), rawPtr(globalPool));
        }
        checkGpuErrors(cudaDeviceSynchronize());
    }
};
} // namespace detail

struct GpuAlwaysTraverseNeighborhood
{
    unsigned ngmax;

    template<class Tc, class KeyType, class Th>
    detail::GpuAlwaysTraverseNeighborhoodImpl<Tc, KeyType, Th> build(OctreeNsView<Tc, KeyType> tree,
                                                                     Box<Tc> box,
                                                                     LocalIndex firstBody,
                                                                     LocalIndex lastBody,
                                                                     const Tc* x,
                                                                     const Tc* y,
                                                                     const Tc* z,
                                                                     const Th* h) const
    {
        return {std::move(tree),
                std::move(box),
                firstBody,
                lastBody,
                x,
                y,
                z,
                h,
                ngmax,
                thrust::device_vector<LocalIndex>(ngmax * TravConfig::numBlocks() *
                                                  (TravConfig::numThreads / GpuConfig::warpSize) *
                                                  TravConfig::targetSize),
                thrust::device_vector<int>(TravConfig::poolSize())};
    }
};

} // namespace cstone::ijloop
