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
 * @brief Neighbor list compression tests
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#include <cstdint>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "gtest/gtest.h"

#include "cstone/compressneighbors.cuh"
#include "cstone/cuda/cuda_utils.cuh"

using namespace cstone;

template<unsigned ItemsPerThread, class T>
__device__ void loadWarpItems(T const* __restrict__ input, T items[ItemsPerThread], unsigned n, const T fill)
{
    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        const unsigned j = threadIdx.x * ItemsPerThread + i;
        items[i]         = j < n ? input[j] : fill;
    }
}

template<unsigned ItemsPerThread, class T>
__device__ void storeWarpItems(T items[ItemsPerThread], T* __restrict__ output, unsigned n)
{
    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        const unsigned j = threadIdx.x * ItemsPerThread + i;
        if (j < n) output[j] = items[i];
    }
}

template<unsigned ItemsPerThread>
__global__ void
runWarpInclusiveSum(std::uint32_t const* __restrict__ input, std::uint32_t* __restrict__ output, unsigned n)
{
    std::uint32_t items[ItemsPerThread];
    loadWarpItems<ItemsPerThread>(input, items, n, 0u);
    cstone::detail::warpInclusiveSum<1, ItemsPerThread>(items);
    storeWarpItems<ItemsPerThread>(items, output, n);
}

TEST(CompressNeighborsGpu, warpInclusiveSum)
{
    thrust::device_vector<std::uint32_t> input(53), output(53);
    for (unsigned i = 0; i < input.size(); ++i)
        input[i] = 2 * i;

    runWarpInclusiveSum<2><<<1, GpuConfig::warpSize>>>(rawPtr(input), rawPtr(output), input.size());
    checkGpuErrors(cudaDeviceSynchronize());

    unsigned sum = 0;
    for (unsigned i = 0; i < input.size(); ++i)
    {
        sum += input[i];
        EXPECT_EQ(output[i], sum);
    }
}

template<unsigned ItemsPerThread>
__global__ void
runWarpInclusiveMax(std::uint32_t const* __restrict__ input, std::uint32_t* __restrict__ output, unsigned n)
{
    std::uint32_t items[ItemsPerThread];
    loadWarpItems<ItemsPerThread>(input, items, n, 0u);
    cstone::detail::warpInclusiveMax<1, ItemsPerThread>(items);
    storeWarpItems<ItemsPerThread>(items, output, n);
}

TEST(CompressNeighborsGpu, warpInclusiveMax)
{
    thrust::device_vector<std::uint32_t> input(53), output(53);
    for (unsigned i = 0; i < input.size(); ++i)
        input[i] = i % 23 - i % 7;

    runWarpInclusiveMax<2><<<1, GpuConfig::warpSize>>>(rawPtr(input), rawPtr(output), input.size());
    checkGpuErrors(cudaDeviceSynchronize());

    unsigned sum = 0;
    for (unsigned i = 0; i < input.size(); ++i)
    {
        sum = std::max(unsigned(input[i]), sum);
        EXPECT_EQ(output[i], sum);
    }
}

template<unsigned ItemsPerThread>
__global__ void runWarpStreamCompact(std::uint32_t const* __restrict__ input,
                                     bool const* __restrict__ keep,
                                     std::uint32_t* __restrict__ output,
                                     unsigned* kept,
                                     unsigned n)
{
    std::uint32_t items[ItemsPerThread];
    loadWarpItems<ItemsPerThread>(input, items, n, ~0u);
    bool keeps[ItemsPerThread];
    loadWarpItems<ItemsPerThread>(keep, keeps, n, false);
    cstone::detail::warpStreamCompact<1, ItemsPerThread>(items, keeps, *kept);
    storeWarpItems<ItemsPerThread>(items, output, n);
}

TEST(CompressNeighborsGpu, warpStreamCompact)
{
    thrust::device_vector<std::uint32_t> input(53), output(53), kept(1);
    thrust::device_vector<bool> keep(53);
    unsigned expected_kept = 0;
    for (unsigned i = 0; i < input.size(); ++i)
    {
        input[i] = i;
        keep[i]  = (i % 5 == 0) | (i % 7 == 0);
        expected_kept += keep[i];
    }

    runWarpStreamCompact<2>
        <<<1, GpuConfig::warpSize>>>(rawPtr(input), rawPtr(keep), rawPtr(output), rawPtr(kept), input.size());
    checkGpuErrors(cudaDeviceSynchronize());

    ASSERT_EQ(kept[0], expected_kept);

    for (unsigned i = 0, j = 0; i < expected_kept; ++i, ++j)
    {
        while (!keep[j])
        {
            ++j;
            ASSERT_LT(j, input.size());
        }
        EXPECT_EQ(output[i], input[j]);
    }
}

template<unsigned ItemsPerThread>
__global__ void rountrip(std::uint32_t const* __restrict__ input,
                         std::uint32_t* __restrict__ output,
                         unsigned n_input,
                         unsigned* n_output)
{
    extern __shared__ char compressed[];

    std::uint32_t items[ItemsPerThread];
    loadWarpItems<ItemsPerThread>(input, items, n_input, ~0u);

    warpCompressNeighbors<1, ItemsPerThread>(items, compressed, n_input);
    warpDecompressNeighbors<1, ItemsPerThread>(compressed, output, *n_output);
}

TEST(CompressNeighborsGpu, roundtrip)
{
    thrust::device_vector<std::uint32_t> nbs = {300, 301, 302, 100, 101, 200, 400, 402, 403,
                                                404, 405, 406, 407, 408, 409, 410, 411};
    thrust::device_vector<std::uint32_t> roundtripped(nbs.size());
    thrust::device_vector<unsigned> output_nb_count(1);

    rountrip<2><<<1, 32, sizeof(std::uint32_t) * nbs.size()>>>(rawPtr(nbs), rawPtr(roundtripped), nbs.size(),
                                                               rawPtr(output_nb_count));

    ASSERT_EQ(output_nb_count[0], nbs.size());
    EXPECT_EQ(roundtripped, nbs);
}

TEST(CompressNeighborsGpu, empty)
{
    thrust::device_vector<std::uint32_t> nbs(0);
    thrust::device_vector<std::uint32_t> roundtripped(nbs.size());
    thrust::device_vector<unsigned> output_nb_count(1);

    rountrip<2>
        <<<1, 32, sizeof(std::uint32_t)>>>(rawPtr(nbs), rawPtr(roundtripped), nbs.size(), rawPtr(output_nb_count));

    ASSERT_EQ(output_nb_count[0], nbs.size());
    EXPECT_EQ(nbs, roundtripped);
}
