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
 * @brief Neighbor list compression
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <cassert>
#include <cstdint>

#include <cooperative_groups.h>
#include <cub/cub.cuh>

#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/clz.hpp"

namespace cstone
{

namespace detail
{

template<unsigned NumWarps, unsigned ItemsPerThread, class T, class Op>
__device__ __forceinline__ void warpInclusiveScan(T items[ItemsPerThread], Op&& op)
{
    namespace cg    = cooperative_groups;
    const auto warp = cg::tiled_partition<GpuConfig::warpSize>(cg::this_thread_block());
    assert(warp.meta_group_size() == NumWarps);

    T global = items[0];
#pragma unroll
    for (unsigned i = 1; i < ItemsPerThread; ++i)
        global = op(global, items[i]);

    using Scan = cub::WarpScan<T, GpuConfig::warpSize>;
    __shared__ typename Scan::TempStorage tmp[NumWarps];

    if constexpr (std::is_same_v<std::decay_t<Op>, cub::Sum>)
        Scan(tmp[warp.meta_group_rank()]).ExclusiveSum(global, global);
    else
        Scan(tmp[warp.meta_group_rank()]).ExclusiveScan(global, global, std::forward<Op>(op));

    if (warp.thread_rank() != 0) items[0] = op(items[0], global);
#pragma unroll
    for (unsigned i = 1; i < ItemsPerThread; ++i)
        items[i] = op(items[i], items[i - 1]);
}

template<unsigned NumWarps, unsigned ItemsPerThread, class T>
__device__ __forceinline__ void warpInclusiveSum(T items[ItemsPerThread])
{
    warpInclusiveScan<NumWarps, ItemsPerThread>(items, cub::Sum());
}

template<unsigned NumWarps, unsigned ItemsPerThread, class T>
__device__ __forceinline__ void warpInclusiveMax(T items[ItemsPerThread])
{
    warpInclusiveScan<NumWarps, ItemsPerThread>(items, cub::Max());
}

template<unsigned NumWarps, unsigned ItemsPerThread, class T>
__device__ __forceinline__ void warpStreamCompact(T items[ItemsPerThread], bool keep[ItemsPerThread], unsigned& kept)
{
    namespace cg    = cooperative_groups;
    const auto warp = cg::tiled_partition<GpuConfig::warpSize>(cg::this_thread_block());

    unsigned indices[ItemsPerThread];
#pragma unroll
    for (unsigned i = 0; i < ItemsPerThread; ++i)
        indices[i] = keep[i];
    warpInclusiveSum<NumWarps, ItemsPerThread>(indices);

    __shared__ T warpItems[NumWarps][ItemsPerThread * GpuConfig::warpSize];
#pragma unroll
    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        if (keep[i]) warpItems[warp.meta_group_rank()][indices[i] - keep[i]] = items[i];
    }
    kept = warp.shfl(indices[ItemsPerThread - 1], GpuConfig::warpSize - 1);
#pragma unroll
    for (unsigned i = 0; i < ItemsPerThread; ++i)
        items[i] = warpItems[warp.meta_group_rank()][warp.thread_rank() * ItemsPerThread + i];
}

} // namespace detail

template<unsigned NumWarps, unsigned ItemsPerThread>
__device__ __forceinline__ void
warpCompressNeighbors(std::uint32_t neighbors[ItemsPerThread], char* output, const unsigned n)
{
    // TODO: add a buffer size limit, currently we just overflow

    namespace cg = cooperative_groups;
    auto warp    = cg::tiled_partition<GpuConfig::warpSize>(cg::this_thread_block());

    if (n == 0)
    {
        if (warp.thread_rank() == 0) *((unsigned*)output) = 0;
        return;
    }

    unsigned diff[ItemsPerThread];
    const unsigned leftNeighbor = warp.shfl_up(neighbors[ItemsPerThread - 1], 1);
    diff[0]                     = neighbors[0] - (warp.thread_rank() > 0 ? leftNeighbor : 0);
#pragma unroll
    for (unsigned i = 1; i < ItemsPerThread; ++i)
        diff[i] = neighbors[i] - neighbors[i - 1];
    unsigned cumulativeOnes[ItemsPerThread];
    for (unsigned i = 0; i < ItemsPerThread; ++i)
        cumulativeOnes[i] = diff[i] == 1;
    detail::warpInclusiveSum<NumWarps, ItemsPerThread>(cumulativeOnes);

    const unsigned leftDiff = warp.shfl_up(diff[ItemsPerThread - 1], 1);
    unsigned consecutiveOnes[ItemsPerThread];
    consecutiveOnes[0] = ((leftDiff == 1) & (diff[0] != 1)) ? cumulativeOnes[0] : 0;
#pragma unroll
    for (unsigned i = 1; i < ItemsPerThread; ++i)
        consecutiveOnes[i] = ((diff[i - 1] == 1) & (diff[i] != 1)) ? cumulativeOnes[i] : 0;
    detail::warpInclusiveMax<NumWarps, ItemsPerThread>(consecutiveOnes);
#pragma unroll
    for (unsigned i = 0; i < ItemsPerThread; ++i)
        consecutiveOnes[i] = cumulativeOnes[i] - consecutiveOnes[i];

    bool keep[ItemsPerThread];
#pragma unroll
    for (unsigned i = 0; i < ItemsPerThread - 1; ++i)
    {
        const unsigned index = ItemsPerThread * warp.thread_rank() + i;
        keep[i] = index < n - 1 ? (consecutiveOnes[i] % 6 == 0) | (consecutiveOnes[i + 1] == 0) : index == n - 1;
    }
    const unsigned rightConsecutiveOnes = warp.shfl_down(consecutiveOnes[0], 1);
    const unsigned index                = ItemsPerThread * warp.thread_rank() + ItemsPerThread - 1;
    keep[ItemsPerThread - 1] =
        index < n - 1 ? (consecutiveOnes[ItemsPerThread - 1] % 6 == 0) | (rightConsecutiveOnes == 0) : index == n - 1;

    unsigned infoNibblesCount;
    detail::warpStreamCompact<NumWarps, ItemsPerThread>(diff, keep, infoNibblesCount);
    detail::warpStreamCompact<NumWarps, ItemsPerThread>(consecutiveOnes, keep, infoNibblesCount);

    std::uint8_t nnibbles[ItemsPerThread];
#pragma unroll
    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        const std::uint8_t bits = 32 - ::countLeadingZeros(diff[i]);
        const unsigned index    = ItemsPerThread * warp.thread_rank() + i;
        nnibbles[i]             = (index >= infoNibblesCount) | (diff[i] == 1) ? 0 : (bits + 3) / 4;
    }

    const unsigned leftConsecutiveOnes = warp.shfl_up(consecutiveOnes[ItemsPerThread - 1], 1);
    if (consecutiveOnes[0] > 6) consecutiveOnes[0] -= leftConsecutiveOnes;
#pragma unroll
    for (unsigned i = 1; i < ItemsPerThread; ++i)
        if (consecutiveOnes[i] > 6) consecutiveOnes[i] -= consecutiveOnes[i - 1];

    std::uint8_t infoNibbles[ItemsPerThread];
#pragma unroll
    for (unsigned i = 0; i < ItemsPerThread; ++i)
        infoNibbles[i] = consecutiveOnes[i] == 0 ? nnibbles[i] : consecutiveOnes[i] + 9;

    if (warp.thread_rank() == 0) *((unsigned*)output) = infoNibblesCount;
    std::uint8_t* infoNibblesBuffer = (std::uint8_t*)((unsigned*)output + 1);

    const auto writeNibble = [](std::uint8_t* buffer, unsigned index, std::uint8_t value, bool odd)
    {
        assert(value < 16);
        if (odd == index % 2)
        {
            std::uint8_t byte = odd ? buffer[index / 2] : 0;
            byte |= (value << ((index % 2) * 4));
            buffer[index / 2] = byte;
        }
    };

#pragma unroll
    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        const unsigned index = ItemsPerThread * warp.thread_rank() + i;
        if (index < infoNibblesCount) writeNibble(infoNibblesBuffer, index, infoNibbles[i], false);
    }
    if constexpr (ItemsPerThread % 2) warp.sync();
#pragma unroll
    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        const unsigned index = ItemsPerThread * warp.thread_rank() + i;
        if (index < infoNibblesCount) writeNibble(infoNibblesBuffer, index, infoNibbles[i], true);
    }

    std::uint8_t* dataNibblesBuffer = infoNibblesBuffer + (infoNibblesCount + 1) / 2;

    unsigned nibbleIndices[ItemsPerThread];
#pragma unroll
    for (unsigned i = 0; i < ItemsPerThread; ++i)
        nibbleIndices[i] = nnibbles[i];
    detail::warpInclusiveSum<NumWarps, ItemsPerThread>(nibbleIndices);

#pragma unroll
    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        const unsigned index = nibbleIndices[i] - nnibbles[i];
        for (unsigned j = 0; j < nnibbles[i]; ++j)
            writeNibble(dataNibblesBuffer, index + j, (diff[i] >> (4 * j)) & 0xf, false);
    }
    warp.sync();
#pragma unroll
    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        const unsigned index = nibbleIndices[i] - nnibbles[i];
        for (unsigned j = 0; j < nnibbles[i]; ++j)
            writeNibble(dataNibblesBuffer, index + j, (diff[i] >> (4 * j)) & 0xf, true);
    }
}

template<unsigned NumWarps, unsigned ItemsPerThread>
__device__ __forceinline__ void
warpDecompressNeighbors(const char* __restrict__ input, std::uint32_t* __restrict__ neighbors, unsigned& n)
{
    namespace cg = cooperative_groups;
    auto warp    = cg::tiled_partition<GpuConfig::warpSize>(cg::this_thread_block());

    const unsigned infoNibblesCount = *((unsigned*)input);

    if (infoNibblesCount == 0)
    {
        n = 0;
        return;
    }

    const std::uint8_t* infoNibblesBuffer = (const std::uint8_t*)((unsigned*)input + 1);
    const std::uint8_t* dataNibblesBuffer = infoNibblesBuffer + (infoNibblesCount + 1) / 2;

    const auto readNibble = [](const std::uint8_t* buffer, unsigned index)
    {
        const std::uint8_t byte = buffer[index / 2];
        return (byte >> ((index % 2) * 4)) & 0xf;
    };

    std::uint8_t infoNibbles[ItemsPerThread];
#pragma unroll
    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        const unsigned index = ItemsPerThread * warp.thread_rank() + i;
        infoNibbles[i]       = index < infoNibblesCount ? readNibble(infoNibblesBuffer, index) : 0;
    }

    std::uint8_t nnibbles[ItemsPerThread], ones[ItemsPerThread], items[ItemsPerThread];
    unsigned dataIndices[ItemsPerThread], neighborIndices[ItemsPerThread];
#pragma unroll
    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        nnibbles[i]          = infoNibbles[i] >= 9 ? 0 : infoNibbles[i];
        ones[i]              = infoNibbles[i] >= 9 ? infoNibbles[i] - 9 : 0;
        const unsigned index = ItemsPerThread * warp.thread_rank() + i;
        items[i]             = index >= infoNibblesCount ? 0 : ones[i] > 0 ? ones[i] : 1;
        dataIndices[i]       = nnibbles[i];
        neighborIndices[i]   = items[i];
    }
    detail::warpInclusiveSum<NumWarps, ItemsPerThread>(dataIndices);
    detail::warpInclusiveSum<NumWarps, ItemsPerThread>(neighborIndices);
    n = warp.shfl(neighborIndices[ItemsPerThread - 1], GpuConfig::warpSize - 1);

#pragma unroll
    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        dataIndices[i] -= nnibbles[i];
        neighborIndices[i] -= items[i];
    }

#pragma unroll
    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        const unsigned neighborIndex = neighborIndices[i];
        if (ones[i])
        {
            for (unsigned j = 0; j < ones[i]; ++j)
                neighbors[neighborIndex + j] = 1;
        }
        else
        {
            std::uint32_t data = 0;
            for (unsigned j = 0; j < nnibbles[i]; ++j)
                data |= readNibble(dataNibblesBuffer, dataIndices[i] + j) << (4 * j);
            neighbors[neighborIndex] = data;
        }
    }

    warp.sync();
    std::uint32_t neighborItems[ItemsPerThread];
#pragma unroll
    for (unsigned i = 0; i < ItemsPerThread; ++i)
        neighborItems[i] = neighbors[ItemsPerThread * warp.thread_rank() + i];
    detail::warpInclusiveSum<NumWarps, ItemsPerThread>(neighborItems);
#pragma unroll
    for (unsigned i = 0; i < ItemsPerThread; ++i)
        neighbors[ItemsPerThread * warp.thread_rank() + i] = neighborItems[i];
}

} // namespace cstone
