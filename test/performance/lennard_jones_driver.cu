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

#include <thrust/universal_vector.h>
#include <thrust/universal_vector.h>

#include "cstone/traversal/ijloop/cpu.hpp"
#include "cstone/traversal/ijloop/gpu_alwaystraverse.cuh"
#include "cstone/traversal/ijloop/gpu_clusternblist.cuh"
#include "cstone/traversal/ijloop/gpu_fullnblist.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/traversal/find_neighbors_clustered.cuh"

#include "../coord_samples/face_centered_cubic.hpp"
#include "./gromacs_ijloop.cuh"

using namespace cstone;

constexpr unsigned ngmax = 224;

template<class T>
struct LjKernelFun
{
    T lj1, lj2;

    template<class ParticleData, class Tc>
    constexpr __host__ __device__ auto
    operator()(ParticleData const& iData, ParticleData const& jData, Vec3<Tc> ijPosDiff, T distSq) const
    {
        const auto [i, iPos, hi] = iData;
        const auto [j, jPos, hj] = jData;
        const T r2inv            = T(1) / distSq;
        const T r6inv            = r2inv * r2inv * r2inv;
        const T forcelj          = r6inv * (lj1 * r6inv - lj2);
        const T fpair            = i == j ? 0 : forcelj * r2inv;
        return std::make_tuple(T(ijPosDiff[0]) * fpair, T(ijPosDiff[1]) * fpair, T(ijPosDiff[2]) * fpair);
    }
};

template<class Tc, class T, class StrongKeyType, class Neighborhood>
void benchmarkGPU(Neighborhood const& neighborhood)
{
    using KeyType = typename StrongKeyType::ValueType;

    constexpr int nx = 200;
    Box<Tc> box{0, 1.6795962 * nx, BoundaryType::periodic};

    FaceCenteredCubicCoordinates<Tc, StrongKeyType> coords(nx, nx, nx, box);
    const int n = coords.x().size();
    std::vector<T> h(n, 3.8 / 2);
    printf("Number of atoms: %d\n", n);

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

    OctreeNsView<Tc, KeyType> nsView{octree.numLeafNodes,
                                     octree.prefixes.data(),
                                     octree.childOffsets.data(),
                                     octree.internalToLeaf.data(),
                                     octree.levelRange.data(),
                                     nullptr,
                                     layout.data(),
                                     centers.data(),
                                     sizes.data()};

    const T lj1 = 48;
    const T lj2 = 24;
    std::vector<T> afx(n), afy(n), afz(n);
    ijloop::CpuDirectNeighborhood{ngmax}
        .build(nsView, box, 0, n, x, y, z, h.data())
        .ijLoop(std::make_tuple(), std::make_tuple(afx.data(), afy.data(), afz.data()), LjKernelFun<T>{lj1, lj2},
                ijloop::symmetry::odd);

    thrust::universal_vector<Tc> d_x(coords.x().begin(), coords.x().end());
    thrust::universal_vector<Tc> d_y(coords.y().begin(), coords.y().end());
    thrust::universal_vector<Tc> d_z(coords.z().begin(), coords.z().end());
    thrust::universal_vector<T> d_h = h;
    thrust::universal_vector<T> d_afx(n, std::numeric_limits<T>::quiet_NaN());
    thrust::universal_vector<T> d_afy(n, std::numeric_limits<T>::quiet_NaN());
    thrust::universal_vector<T> d_afz(n, std::numeric_limits<T>::quiet_NaN());
    printf("Memory usage of particle data: %.2f MB\n",
           (sizeof(Tc) * (d_x.size() + d_y.size() + d_z.size()) +
            sizeof(T) * (d_h.size() + d_afx.size() + d_afy.size() + d_afz.size())) /
               1.0e6);

    thrust::universal_vector<KeyType> d_prefixes             = octree.prefixes;
    thrust::universal_vector<TreeNodeIndex> d_childOffsets   = octree.childOffsets;
    thrust::universal_vector<TreeNodeIndex> d_internalToLeaf = octree.internalToLeaf;
    thrust::universal_vector<TreeNodeIndex> d_levelRange     = octree.levelRange;
    thrust::universal_vector<LocalIndex> d_layout            = layout;
    thrust::universal_vector<Vec3<Tc>> d_centers             = centers;
    thrust::universal_vector<Vec3<Tc>> d_sizes               = sizes;

    OctreeNsView<Tc, KeyType> nsViewGpu{octree.numLeafNodes,      rawPtr(d_prefixes),   rawPtr(d_childOffsets),
                                        rawPtr(d_internalToLeaf), rawPtr(d_levelRange), nullptr,
                                        rawPtr(d_layout),         rawPtr(d_centers),    rawPtr(d_sizes)};

    thrust::universal_vector<KeyType> d_codes(coords.particleKeys().begin(), coords.particleKeys().end());
    const auto* deviceKeys = (const KeyType*)(rawPtr(d_codes));

    auto neighborhoodGPU = neighborhood.build(nsViewGpu, box, 0, n, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_h));

    std::array<float, 11> times;
    std::array<cudaEvent_t, times.size() + 1> events;
    for (auto& event : events)
        cudaEventCreate(&event);
    cudaEventRecord(events[0]);
    for (std::size_t i = 1; i < events.size(); ++i)
    {
        neighborhoodGPU.ijLoop(std::make_tuple(), std::make_tuple(rawPtr(d_afx), rawPtr(d_afy), rawPtr(d_afz)),
                               LjKernelFun<T>{lj1, lj2}, ijloop::symmetry::odd);
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
        constexpr bool isDouble = std::is_same_v<T, double>;
        constexpr double atol   = isDouble ? 1e-6 : 1e-5;
        constexpr double rtol   = isDouble ? 1e-5 : 1e-4;
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

    std::cout << "--- BATCHED DIRECT ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(ijloop::GpuAlwaysTraverseNeighborhood{ngmax});

    std::cout << "--- NAIVE TWO-STAGE ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(ijloop::GpuFullNbListNeighborhood{ngmax});

    std::cout << "--- GROMACS CLUSTERED TWO-STAGE ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(ijloop::GromacsLikeNeighborhood{ngmax});

    using BaseClusterNb = ijloop::GpuClusterNbListNeighborhood<>::withNcMax<160>::withClusterSize<4, 4>;
    std::cout << "--- CLUSTERED TWO-STAGE ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(BaseClusterNb::withoutSymmetry::withoutCompression{});
    std::cout << "--- COMPRESSED CLUSTERED TWO-STAGE ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(BaseClusterNb::withoutSymmetry::withCompression<8>{});
    using SymmetricClusterNb = BaseClusterNb::withNcMax<128>::withSymmetry;
    std::cout << "--- CLUSTERED TWO-STAGE SYMMETRIC ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(SymmetricClusterNb::withoutCompression{});
    std::cout << "--- COMPRESSED CLUSTERED TWO-STAGE SYMMETRIC ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(SymmetricClusterNb::withCompression<7>{});

    return 0;
}
