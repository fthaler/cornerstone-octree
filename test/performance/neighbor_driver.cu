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
 * @brief  Find neighbors in Morton code sorted x,y,z arrays
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iomanip>
#include <iostream>
#include <iterator>

#include <thrust/universal_vector.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/findneighbors.hpp"

#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/traversal/find_neighbors_clustered.cuh"

#include "../coord_samples/random.hpp"
#include "timing.cuh"

using namespace cstone;

//! @brief depth-first traversal based neighbor search
template<class T, class KeyType>
__global__ void findNeighborsKernel(const T* x,
                                    const T* y,
                                    const T* z,
                                    const T* h,
                                    LocalIndex firstId,
                                    LocalIndex lastId,
                                    const Box<T> box,
                                    const OctreeNsView<T, KeyType> treeView,
                                    unsigned ngmax,
                                    LocalIndex* neighbors,
                                    unsigned* neighborsCount)
{
    cstone::LocalIndex tid = blockDim.x * blockIdx.x + threadIdx.x;
    cstone::LocalIndex id  = firstId + tid;
    if (id >= lastId) { return; }

    neighborsCount[id] = findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors + tid * ngmax);
}

/*! @brief Neighbor search for bodies within the specified range
 *
 * @param[in]    firstBody           index of first body in @p bodyPos to compute acceleration for
 * @param[in]    lastBody            index (exclusive) of last body in @p bodyPos to compute acceleration for
 * @param[in]    rootRange           (start,end) index pair of cell indices to start traversal from
 * @param[in]    x,y,z,h             bodies, in SFC order and as referenced by @p layout
 * @param[in]    tree.childOffsets   location (index in [0:numTreeNodes]) of first child of each cell, 0 indicates a
 *                                   leaf
 * @param[in]    tree.internalToLeaf for each cell in [0:numTreeNodes], stores the leaf cell (cstone) index in
 *                                   [0:numLeaves] if the cell is not a leaf, the value is negative
 * @param[in]    tree.layout         for each leaf cell in [0:numLeaves], stores the index of the first body in the cell
 * @param[in]    tree.centers        x,y,z geometric center of each cell in [0:numTreeNodes]
 * @param[in]    tree.sizes          x,y,z geometric size of each cell in [0:numTreeNodes]
 * @param[in]    box                 global coordinate bounding box
 * @param[out]   nc                  neighbor counts of bodies with indices in [firstBody, lastBody]
 * @param[-]     globalPool          temporary storage for the cell traversal stack, uninitialized
 *                                   each active warp needs space for TravConfig::memPerWarp int32,
 *                                   so the total size is TravConfig::memPerWarp * numWarpsPerBlock * numBlocks
 */
template<class Tc, class Th, class KeyType>
__global__ __launch_bounds__(TravConfig::numThreads) void traverseBT(cstone::LocalIndex firstBody,
                                                                     cstone::LocalIndex lastBody,
                                                                     const Tc* __restrict__ x,
                                                                     const Tc* __restrict__ y,
                                                                     const Tc* __restrict__ z,
                                                                     const Th* __restrict__ h,
                                                                     OctreeNsView<Tc, KeyType> tree,
                                                                     const Box<Tc> box,
                                                                     unsigned* nc,
                                                                     unsigned* nidx,
                                                                     unsigned ngmax,
                                                                     int* globalPool)
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

template<class Tc, class Th, class KeyType>
auto findNeighborsBT(size_t firstBody,
                     size_t lastBody,
                     const Tc* x,
                     const Tc* y,
                     const Tc* z,
                     const Th* h,
                     OctreeNsView<Tc, KeyType> tree,
                     const Box<Tc>& box,
                     unsigned* nc,
                     unsigned* nidx,
                     unsigned ngmax)
{
    unsigned numBodies = lastBody - firstBody;
    unsigned numBlocks = TravConfig::numBlocks(numBodies);
    unsigned poolSize  = TravConfig::poolSize(numBodies);
    static thrust::universal_vector<int> globalPool;
    globalPool.resize(poolSize);

    printf("launching %d blocks\n", numBlocks);
    resetTraversalCounters<<<1, 1>>>();
    auto t0 = std::chrono::high_resolution_clock::now();
    traverseBT<<<numBlocks, TravConfig::numThreads>>>(firstBody, lastBody, x, y, z, h, tree, box, nc, nidx, ngmax,
                                                      rawPtr(globalPool));
    kernelSuccess("traverseBT");

    auto t1   = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();

    NcStats::type stats[NcStats::numStats];
    checkGpuErrors(cudaMemcpyFromSymbol(stats, ncStats, NcStats::numStats * sizeof(uint64_t)));

    NcStats::type sumP2P   = stats[NcStats::sumP2P];
    NcStats::type maxP2P   = stats[NcStats::maxP2P];
    NcStats::type maxStack = stats[NcStats::maxStack];

    util::array<Tc, 2> interactions;
    interactions[0] = Tc(sumP2P) / Tc(numBodies);
    interactions[1] = Tc(maxP2P);

    fprintf(stdout, "Traverse : %.7f s (%.7f TFlops) P2P %f, maxP2P %f, maxStack %llu\n", dt, 11.0 * sumP2P / dt / 1e12,
            interactions[0], interactions[1], maxStack);

    return interactions;
}

template<class Tc, class Th, class KeyType>
std::array<std::vector<unsigned>, 2> findClusterNeighborsCPU(std::size_t firstBody,
                                                             std::size_t lastBody,
                                                             const Tc* x,
                                                             const Tc* y,
                                                             const Tc* z,
                                                             const Th* h,
                                                             const OctreeNsView<Tc, KeyType>& tree,
                                                             const Box<Tc>& box,
                                                             unsigned ngmax,
                                                             unsigned ncmax)
{
    std::vector<unsigned> nc(lastBody), nidx(lastBody * ngmax);
    for (auto i = firstBody; i < lastBody; ++i)
        nc[i] = findNeighbors(i, x, y, z, h, tree, box, ngmax, nidx.data() + i * ngmax);

    std::size_t iClusters = iceil(lastBody, ClusterConfig::iSize);
    std::vector<unsigned> clusterNeighborsCount(iClusters, 0);
    std::vector<unsigned> clusterNeighbors(iClusters * ncmax);

    for (auto i = firstBody; i < lastBody; ++i)
    {
        auto iCluster               = i / ClusterConfig::iSize;
        unsigned* iClusterNeighbors = rawPtr(clusterNeighbors) + iCluster * ncmax;
        unsigned nci                = nc[i];
        for (unsigned j = 0; j < nci; ++j)
        {
            unsigned nj       = nidx[i * ngmax + j];
            unsigned jCluster = nj / ClusterConfig::jSize;
            if (i / ClusterConfig::jSize == jCluster || nj / ClusterConfig::iSize == iCluster) continue;
            bool alreadyIn = false;
            for (unsigned k = 0; k < std::min(clusterNeighborsCount[iCluster], ncmax); ++k)
            {
                if (iClusterNeighbors[k] == jCluster)
                {
                    alreadyIn = true;
                    break;
                }
            }
            if (!alreadyIn)
            {
                if (clusterNeighborsCount[iCluster] < ncmax)
                    iClusterNeighbors[clusterNeighborsCount[iCluster]] = jCluster;
                ++clusterNeighborsCount[iCluster];
            }
        }
    }
    return {clusterNeighborsCount, clusterNeighbors};
}

template<class Tc, class Th, class KeyType>
void findNeighborsC(std::size_t firstBody,
                    std::size_t lastBody,
                    const Tc* x,
                    const Tc* y,
                    const Tc* z,
                    const Th* h,
                    OctreeNsView<Tc, KeyType> tree,
                    const Box<Tc>& box,
                    unsigned* nc,
                    unsigned* nidx,
                    unsigned ngmax,
                    unsigned ncmax)
{
    unsigned numBodies = lastBody - firstBody;
    unsigned numBlocks = TravConfig::numBlocks(numBodies);
    unsigned poolSize  = TravConfig::poolSize(numBodies);
    static thrust::universal_vector<unsigned> clusterNeighbors, clusterNeighborsCount;
    static thrust::universal_vector<int> globalPool;
    clusterNeighbors.resize(iceil(lastBody - 1, ClusterConfig::iSize) * ncmax);
    clusterNeighborsCount.resize(iceil(lastBody - 1, ClusterConfig::iSize));
    globalPool.resize(poolSize);

    resetTraversalCounters<<<1, 1>>>();
    auto t0 = std::chrono::high_resolution_clock::now();
    findClusterNeighbors3<<<numBlocks, TravConfig::numThreads>>>(firstBody, lastBody, x, y, z, h, tree, box,
                                                                 rawPtr(clusterNeighborsCount),
                                                                 rawPtr(clusterNeighbors), ncmax, rawPtr(globalPool));
    kernelSuccess("findClusterNeighbors");
    auto t1   = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Clustered NB build time " << dt << " s" << std::endl;

    if (false) // enable for debugging cluster neighbors
    {

        auto [clusterNeighborsCountCPU, clusterNeighborsCPU] =
            findClusterNeighborsCPU(firstBody, lastBody, x, y, z, h, tree, box, ngmax, ncmax);

        bool fail = false;
        for (std::size_t i = 0; i < clusterNeighborsCountCPU.size(); ++i)
        {
            if (clusterNeighborsCountCPU[i] != clusterNeighborsCount[i])
            {
                std::cout << i << " " << clusterNeighborsCountCPU[i] << " " << clusterNeighborsCount[i] << "\n";
                fail = true;
            }
        }
        if (fail) { std::cout << "Cluster neighbor count failed ^" << std::endl; }
        else
        {
            std::cout << "Cluster neighbor count passed" << std::endl;
            bool fail = false;
            for (std::size_t iCluster = 0; iCluster < clusterNeighborsCountCPU.size(); ++iCluster)
            {
                unsigned nc = clusterNeighborsCountCPU[iCluster];
                std::sort(clusterNeighborsCPU.begin() + iCluster * ncmax,
                          clusterNeighborsCPU.begin() + iCluster * ncmax + nc);
                std::vector<unsigned> sortedClusterNeighborsGPU(nc);
                for (unsigned nb = 0; nb < nc; ++nb)
                    sortedClusterNeighborsGPU[nb] = clusterNeighbors[clusterNeighborIndex(iCluster, nb, ncmax)];
                std::sort(sortedClusterNeighborsGPU.begin(), sortedClusterNeighborsGPU.end());
                for (unsigned nb = 0; nb < nc; ++nb)
                    if (clusterNeighborsCPU[iCluster * ncmax + nb] != sortedClusterNeighborsGPU[nb])
                    {
                        std::cout << iCluster << ":" << nb << " " << clusterNeighborsCPU[iCluster * ncmax + nb] << " "
                                  << sortedClusterNeighborsGPU[nb] << "\n";
                        fail = true;
                    }
            }
            if (fail) { std::cout << "Cluster neighbor search failed" << std::endl; }
            else { std::cout << "Cluster neighbor search passed" << std::endl; }
        }

        double r                  = 2 * h[0];
        double rho                = lastBody - firstBody;
        double expected_neighbors = 4.0 / 3.0 * M_PI * r * r * r * rho;
        auto average_neighbors = std::accumulate(clusterNeighborsCountCPU.begin(), clusterNeighborsCountCPU.end(), 0) /
                                 double(clusterNeighborsCountCPU.size()) * ClusterConfig::jSize;
        std::cout << "Interactions: " << (average_neighbors / expected_neighbors) << std::endl;
    }

    auto countNeighbors = [=] __device__(unsigned i, auto iPos, auto hi, unsigned j, auto jPos, auto distanceSq)
    {
        if (i == j) return 0u;

        unsigned nb = atomicAdd(&nc[i], 1);
        if (nb < ngmax)
            nidx[(i / TravConfig::targetSize) * TravConfig::targetSize * ngmax + TravConfig::targetSize * nb +
                 i % TravConfig::targetSize] = j;
        return 1u;
    };

    cudaMemset(nc + firstBody, 0, numBodies * sizeof(unsigned));
    resetTraversalCounters<<<1, 1>>>();
    t0 = std::chrono::high_resolution_clock::now();
    findNeighborsClustered<<<numBlocks, TravConfig::numThreads>>>(firstBody, lastBody, x, y, z, h, box,
                                                                  rawPtr(clusterNeighborsCount),
                                                                  rawPtr(clusterNeighbors), ncmax, countNeighbors, nc);
    kernelSuccess("findClusterNeighbors");
    t1 = std::chrono::high_resolution_clock::now();
    dt = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Clustered NB search time " << dt << " s" << std::endl;
}

template<class T, class StrongKeyType, class FindNeighborsGpuF, class NeighborIndexF>
void benchmarkGpu(FindNeighborsGpuF findNeighborsGpu, NeighborIndexF neighborIndex)
{
    using KeyType = typename StrongKeyType::ValueType;

    Box<T> box{0, 1, BoundaryType::periodic};
    int n = 2000000;

    RandomCoordinates<T, StrongKeyType> coords(n, box);
    std::vector<T> h(n, 0.012);

    // RandomGaussianCoordinates<T, StrongKeyType> coords(n, box);
    // adjustSmoothingLength<KeyType>(n, 100, 200, coords.x(), coords.y(), coords.z(), h, box);

    int ngmax = 200;

    std::vector<LocalIndex> neighborsCPU(ngmax * n);
    std::vector<unsigned> neighborsCountCPU(n);

    const T* x        = coords.x().data();
    const T* y        = coords.y().data();
    const T* z        = coords.z().data();
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

    std::vector<Vec3<T>> centers(octree.numNodes), sizes(octree.numNodes);
    gsl::span<const KeyType> nodeKeys(octree.prefixes.data(), octree.numNodes);
    nodeFpCenters<KeyType>(nodeKeys, centers.data(), sizes.data(), box);

    OctreeNsView<T, KeyType> nsView{octree.prefixes.data(),
                                    octree.childOffsets.data(),
                                    octree.internalToLeaf.data(),
                                    octree.levelRange.data(),
                                    layout.data(),
                                    centers.data(),
                                    sizes.data()};

    auto findNeighborsCpu = [&]()
    {
#pragma omp parallel for
        for (LocalIndex i = 0; i < n; ++i)
        {
            neighborsCountCPU[i] =
                findNeighbors(i, x, y, z, h.data(), nsView, box, ngmax, neighborsCPU.data() + i * ngmax);
        }
    };

    float cpuTime = timeCpu(findNeighborsCpu);

    std::cout << "CPU time " << cpuTime << " s" << std::endl;
    std::copy(neighborsCountCPU.data(), neighborsCountCPU.data() + std::min(n, 64),
              std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    std::vector<cstone::LocalIndex> neighborsGPU(ngmax * (n + TravConfig::targetSize));
    std::vector<unsigned> neighborsCountGPU(n);

    thrust::universal_vector<T> d_x(coords.x().begin(), coords.x().end());
    thrust::universal_vector<T> d_y(coords.y().begin(), coords.y().end());
    thrust::universal_vector<T> d_z(coords.z().begin(), coords.z().end());
    thrust::universal_vector<T> d_h = h;

    thrust::universal_vector<KeyType> d_prefixes             = octree.prefixes;
    thrust::universal_vector<TreeNodeIndex> d_childOffsets   = octree.childOffsets;
    thrust::universal_vector<TreeNodeIndex> d_internalToLeaf = octree.internalToLeaf;
    thrust::universal_vector<TreeNodeIndex> d_levelRange     = octree.levelRange;
    thrust::universal_vector<LocalIndex> d_layout            = layout;
    thrust::universal_vector<Vec3<T>> d_centers              = centers;
    thrust::universal_vector<Vec3<T>> d_sizes                = sizes;

    OctreeNsView<T, KeyType> nsViewGpu{rawPtr(d_prefixes),   rawPtr(d_childOffsets), rawPtr(d_internalToLeaf),
                                       rawPtr(d_levelRange), rawPtr(d_layout),       rawPtr(d_centers),
                                       rawPtr(d_sizes)};

    thrust::universal_vector<LocalIndex> d_neighbors(neighborsGPU.size());
    thrust::universal_vector<unsigned> d_neighborsCount(neighborsCountGPU.size());

    thrust::universal_vector<KeyType> d_codes(coords.particleKeys().begin(), coords.particleKeys().end());
    const auto* deviceKeys = (const KeyType*)(rawPtr(d_codes));

    auto findNeighborsLambda = [&]()
    {
        findNeighborsGpu(0, n, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_h), nsViewGpu, box,
                         rawPtr(d_neighborsCount), rawPtr(d_neighbors), ngmax);
    };

    findNeighborsLambda();
    float gpuTime = timeGpu(findNeighborsLambda);

    thrust::copy(d_neighborsCount.begin(), d_neighborsCount.end(), neighborsCountGPU.begin());
    thrust::copy(d_neighbors.begin(), d_neighbors.end(), neighborsGPU.begin());

    std::cout << "GPU time " << gpuTime / 1000 << " s" << std::endl;
    std::copy(neighborsCountGPU.data(), neighborsCountGPU.data() + std::min(n, 64),
              std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    int numFails     = 0;
    int numFailsList = 0;
    for (int i = 0; i < n; ++i)
    {
        std::sort(neighborsCPU.data() + i * ngmax, neighborsCPU.data() + i * ngmax + neighborsCountCPU[i]);

        std::vector<cstone::LocalIndex> nilist(neighborsCountGPU[i]);
        for (unsigned j = 0; j < neighborsCountGPU[i]; ++j)
        {
            nilist[j] = neighborsGPU[neighborIndex(i, j, ngmax)];
        }
        std::sort(nilist.begin(), nilist.end());

        if (neighborsCountGPU[i] != neighborsCountCPU[i])
        {
            std::cout << i << " " << neighborsCountGPU[i] << " " << neighborsCountCPU[i] << std::endl;
            numFails++;
        }

        if (!std::equal(begin(nilist), end(nilist), neighborsCPU.begin() + i * ngmax)) { numFailsList++; }
    }

    bool allEqual = std::equal(begin(neighborsCountGPU), end(neighborsCountGPU), begin(neighborsCountCPU));
    if (allEqual)
        std::cout << "Neighbor counts: PASS\n";
    else
        std::cout << "Neighbor counts: FAIL " << numFails << std::endl;

    std::cout << "numFailsList " << numFailsList << std::endl;
}

int main()
{
    using Tc      = double;
    using KeyType = HilbertKey<uint64_t>;

    std::cout << "--- NAIVE ---" << std::endl;
    auto naive = [](std::size_t firstBody, std::size_t lastBody, const auto* x, const auto* y, const auto* z,
                    const auto* h, auto tree, const auto& box, unsigned* nc, unsigned* nidx, unsigned ngmax)
    {
        findNeighborsKernel<<<iceil(lastBody - firstBody, 128), 128>>>(x, y, z, h, firstBody, lastBody, box, tree,
                                                                       ngmax, nidx, nc);
    };
    auto neighborIndexNaive = [](unsigned i, unsigned j, unsigned ngmax) { return i * ngmax + j; };
    benchmarkGpu<Tc, KeyType>(naive, neighborIndexNaive);

    std::cout << "--- BATCHED ---" << std::endl;
    auto batched = [](std::size_t firstBody, std::size_t lastBody, const auto* x, const auto* y, const auto* z,
                      const auto* h, auto tree, const auto& box, unsigned* nc, unsigned* nidx, unsigned ngmax)
    { findNeighborsBT(firstBody, lastBody, x, y, z, h, tree, box, nc, nidx, ngmax); };
    auto neighborIndexBatched = [](unsigned i, unsigned j, unsigned ngmax)
    {
        auto warpOffset = (i / TravConfig::targetSize) * TravConfig::targetSize * ngmax;
        auto laneOffset = i % TravConfig::targetSize;
        auto index      = warpOffset + TravConfig::targetSize * j + laneOffset;
        return index;
    };
    benchmarkGpu<Tc, KeyType>(batched, neighborIndexBatched);

    std::cout << "--- CLUSTERED ---" << std::endl;
    auto clustered = [&](std::size_t firstBody, std::size_t lastBody, const auto* x, const auto* y, const auto* z,
                         const auto* h, auto tree, const auto& box, unsigned* nc, unsigned* nidx, unsigned ngmax)
    { findNeighborsC(firstBody, lastBody, x, y, z, h, tree, box, nc, nidx, ngmax, 512); };
    benchmarkGpu<Tc, KeyType>(clustered, neighborIndexBatched);
}
