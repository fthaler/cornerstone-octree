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

#include <thrust/device_vector.h>

#include "cstone/cuda/thrust_util.cuh"
#include "cstone/traversal/ijloop/gpu_alwaystraverse.cuh"
#include "cstone/traversal/ijloop/gpu_clusternblist.cuh"
#include "cstone/traversal/ijloop/gpu_fullnblist.cuh"
#include "cstone/traversal/ijloop/cpu.hpp"

#include "../coord_samples/random.hpp"

using namespace cstone;

constexpr unsigned ngmax = 256;
constexpr unsigned ncmax = 256;

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

class CudaAutoTimer
{
    cudaEvent_t start, end;
    const char* fmt;

public:
    CudaAutoTimer(const char* fmt)
        : fmt(fmt)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);
    }
    ~CudaAutoTimer()
    {
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float duration;
        cudaEventElapsedTime(&duration, start, end);
        cudaEventDestroy(start);
        cudaEventDestroy(end);
        printf(fmt, duration / 1000.0);
    }
};

template<class T>
struct DensityKernelFun
{
    const T* wh;

    template<class ParticleData>
    constexpr __host__ __device__ auto operator()(ParticleData const& iData, ParticleData const& jData, T distSq) const
    {
        const auto [i, xi, yi, zi, hi, _]  = iData;
        const auto [j, xj, yj, zj, hj, mj] = jData;
        const T dist                       = std::sqrt(distSq);
        const T vloc                       = dist * (T(1) / hi);
        const T w                          = i == j ? T(1) : table_lookup(wh, vloc);
        return std::make_tuple(w * mj);
    }
};

template<class Tc, class T, class StrongKeyType, class Neighborhood>
void benchmarkGPU(Neighborhood const& neighborhood)
{
    using KeyType = typename StrongKeyType::ValueType;

    Box<Tc> box{0, 1, BoundaryType::periodic};
    int scale = 10;
    int n     = 100000 * scale;

    RandomCoordinates<Tc, StrongKeyType> coords(n, box);
    std::vector<T> h(n, 0.75 / 20 / std::cbrt(scale));

    const double r                  = 2 * h[0];
    const double expected_neighbors = 4.0 / 3.0 * M_PI * r * r * r * n;
    std::cout << "Number of particles: " << n << std::endl;
    std::cout << "Expected average number of neighbors: " << expected_neighbors << std::endl;

    const Tc* x       = coords.x().data();
    const Tc* y       = coords.y().data();
    const Tc* z       = coords.z().data();
    const KeyType* codes = coords.particleKeys().data();

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

    std::vector<T> rho(n), m(n, 1.0);
    auto wh = kernelTable<T>();
    ijloop::CpuDirectNeighborhood{ngmax}
        .build(nsView, box, 0, n, x, y, z, h.data())
        .ijLoop(std::make_tuple(m.data()), std::make_tuple(rho.data()), DensityKernelFun<T>{wh.data()},
                ijloop::symmetry::even);

    thrust::device_vector<Tc> d_x(coords.x().begin(), coords.x().end());
    thrust::device_vector<Tc> d_y(coords.y().begin(), coords.y().end());
    thrust::device_vector<Tc> d_z(coords.z().begin(), coords.z().end());
    thrust::device_vector<T> d_h = h;
    thrust::device_vector<T> d_m = m;
    thrust::device_vector<T> d_rho(n);
    thrust::device_vector<T> d_wh(wh.size());
    thrust::copy(wh.begin(), wh.end(), d_wh.begin());

    printf("Memory usage of particle data: %.2f MB\n", (sizeof(Tc) * (d_x.size() + d_y.size() + d_z.size()) +
                                                        sizeof(T) * (d_h.size() + d_m.size() + d_rho.size())) /
                                                           1.0e6);

    thrust::device_vector<KeyType> d_prefixes             = octree.prefixes;
    thrust::device_vector<TreeNodeIndex> d_childOffsets   = octree.childOffsets;
    thrust::device_vector<TreeNodeIndex> d_internalToLeaf = octree.internalToLeaf;
    thrust::device_vector<TreeNodeIndex> d_levelRange     = octree.levelRange;
    thrust::device_vector<LocalIndex> d_layout            = layout;
    thrust::device_vector<Vec3<Tc>> d_centers             = centers;
    thrust::device_vector<Vec3<Tc>> d_sizes               = sizes;

    printf("Memory usage of tree data: %.2f MB\n",
           (sizeof(KeyType) * d_prefixes.size() +
            sizeof(TreeNodeIndex) * (d_childOffsets.size() + d_internalToLeaf.size() + d_levelRange.size()) +
            sizeof(LocalIndex) * d_layout.size() + sizeof(Vec3<Tc>) * (d_centers.size() + d_sizes.size())) /
               1.0e6);

    OctreeNsView<Tc, KeyType> nsViewGpu{octree.numLeafNodes,      rawPtr(d_prefixes),   rawPtr(d_childOffsets),
                                        rawPtr(d_internalToLeaf), rawPtr(d_levelRange), nullptr,
                                        rawPtr(d_layout),         rawPtr(d_centers),    rawPtr(d_sizes)};

    thrust::device_vector<KeyType> d_codes(coords.particleKeys().begin(), coords.particleKeys().end());
    const auto* deviceKeys = (const KeyType*)(rawPtr(d_codes));

    auto neighborhoodGPU = neighborhood.build(nsViewGpu, box, 0, n, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_h));

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

    std::array<float, 11> times;
    std::array<cudaEvent_t, times.size() + 1> events;
    for (auto& event : events)
        cudaEventCreate(&event);
    cudaEventRecord(events[0]);
    for (std::size_t i = 1; i < events.size(); ++i)
    {
        neighborhoodGPU.ijLoop(std::make_tuple(rawPtr(d_m)), std::make_tuple(rawPtr(d_rho)),
                               DensityKernelFun<T>{rawPtr(d_wh)}, ijloop::symmetry::even);
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

    printf("GPU times [s]: ");
    for (auto t : times)
        printf("%7.6fs ", t / 1000);
    printf("\n");
    printf("Gparticle-updates/s: ");
    for (auto t : times)
        printf("%7.6fs ", n / 1.0e6 / t);
    printf("\n");

    std::vector<T> rhoGPU(n);
    thrust::copy(d_rho.begin(), d_rho.end(), rhoGPU.begin());

    int numFails = 0;
    auto isclose = [](double a, double b)
    {
        constexpr bool isDouble = std::is_same_v<T, double>;
        constexpr double atol   = 0;
        constexpr double rtol   = isDouble ? 1e-5 : 1e-4;
        return std::abs(a - b) <= atol + rtol * std::abs(b);
    };
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
        if (!isclose(rhoGPU[i], rho[i]))
        {
            int failNum;
#pragma omp atomic capture
            failNum = numFails++;
            if (failNum < 10)
            {
#pragma omp critical
                printf("%i %.10f %.10f\n", i, rhoGPU[i], rho[i]);
            }
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

    std::cout << "--- BATCHED DIRECT ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(ijloop::GpuAlwaysTraverseNeighborhood{ngmax});

    std::cout << "--- NAIVE TWO-STAGE ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(ijloop::GpuFullNbListNeighborhood{ngmax});

    using BaseClusterNb = ijloop::GpuClusterNbListNeighborhood<>::withNcMax<ncmax>::withClusterSize<4, 4>;
    std::cout << "--- CLUSTERED TWO-STAGE ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(BaseClusterNb::withoutSymmetry::withoutCompression{});
    std::cout << "--- COMPRESSED CLUSTERED TWO-STAGE ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(BaseClusterNb::withoutSymmetry::withCompression<10>{});
    std::cout << "--- CLUSTERED TWO-STAGE SYMMETRIC ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(BaseClusterNb::withSymmetry::withoutCompression{});
    std::cout << "--- COMPRESSED CLUSTERED TWO-STAGE SYMMETRIC ---" << std::endl;
    benchmarkGPU<Tc, T, StrongKeyType>(BaseClusterNb::withSymmetry::withCompression<10>{});

    return 0;
}
