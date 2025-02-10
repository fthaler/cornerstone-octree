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
 * @brief Neighbor loop tests
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#include <algorithm>
#include <functional>
#include <random>

#include <thrust/universal_vector.h>

#include "cstone/traversal/ijloop/cpu.hpp"
#include "cstone/traversal/ijloop/gpu_alwaystraverse.cuh"
#include "cstone/traversal/ijloop/gpu_clusternblist.cuh"
#include "cstone/traversal/ijloop/gpu_fullnblist.cuh"

#include "../../coord_samples/random.hpp"

#include "gtest/gtest.h"

using namespace cstone;

constexpr inline unsigned ngmax = 160;

using StrongKeyT = HilbertKey<std::uint64_t>;
using KeyT       = StrongKeyT::ValueType;

struct NeighborFun
{
    template<class ParticleData>
    constexpr auto
    operator()(ParticleData const& iData, ParticleData const& jData, Vec3<double> ijPosDiff, double distSq) const
    {
        const auto [i, iPos, hi, vi] = iData;
        const auto [j, jPos, hj, vj] = jData;
        return std::make_tuple(i, j, iPos, jPos, ijPosDiff, distSq, hi, hj, vi, vj, 1u);
    }
};

using Result                      = std::tuple<thrust::universal_vector<LocalIndex>,   // iSum
                                               thrust::universal_vector<LocalIndex>,   // jSum
                                               thrust::universal_vector<Vec3<double>>, // iPosSum
                                               thrust::universal_vector<Vec3<double>>, // jPosSum
                                               thrust::universal_vector<Vec3<double>>, // ijPosDiffSum
                                               thrust::universal_vector<double>,       // distSqSum
                                               thrust::universal_vector<double>,       // hiSum
                                               thrust::universal_vector<double>,       // hjSum
                                               thrust::universal_vector<double>,       // viSum
                                               thrust::universal_vector<double>,       // vjSum
                                               thrust::universal_vector<unsigned>      // neighborsCount
                                               >;
constexpr static auto resultNames = std::make_tuple("iSum",
                                                    "jSum",
                                                    "iPosSum",
                                                    "jPosSum",
                                                    "ijPosDiffSum",
                                                    "distSqSum",
                                                    "hiSum",
                                                    "hjSum",
                                                    "viSum",
                                                    "vjSum",
                                                    "neighborsCount");

Result reference(const Box<double>& box,
                 const double* const x,
                 const double* const y,
                 const double* const z,
                 const double* const h,
                 const double* const v,
                 const LocalIndex numParticles,
                 const LocalIndex firstIParticle,
                 const LocalIndex lastIParticle)
{
    thrust::universal_vector<LocalIndex> iSum(lastIParticle), jSum(lastIParticle);
    thrust::universal_vector<Vec3<double>> iPosSum(lastIParticle), jPosSum(lastIParticle), ijPosDiffSum(lastIParticle);
    thrust::universal_vector<double> d2Sum(lastIParticle), hiSum(lastIParticle), hjSum(lastIParticle),
        viSum(lastIParticle), vjSum(lastIParticle);
    thrust::universal_vector<unsigned> neighborsCount(lastIParticle);

    for (unsigned i = firstIParticle; i < lastIParticle; ++i)
    {
        for (unsigned j = 0; j < numParticles; ++j)
        {
            const double xi = x[i];
            const double yi = y[i];
            const double zi = z[i];
            const double xj = x[j];
            const double yj = y[j];
            const double zj = z[j];

            double xij = xi - xj;
            double yij = yi - yj;
            double zij = zi - zj;

            if (box.boundaryX() == BoundaryType::periodic)
            {
                if (xij < -0.5 * box.lx())
                    xij += box.lx();
                else if (xij > 0.5 * box.lx())
                    xij -= box.lx();
            }
            if (box.boundaryY() == BoundaryType::periodic)
            {
                if (yij < -0.5 * box.ly())
                    yij += box.ly();
                else if (yij > 0.5 * box.ly())
                    yij -= box.ly();
            }
            if (box.boundaryZ() == BoundaryType::periodic)
            {
                if (zij < -0.5 * box.lz())
                    zij += box.lz();
                else if (zij > 0.5 * box.lz())
                    zij -= box.lz();
            }

            const double d2 = xij * xij + yij * yij + zij * zij;

            if (d2 < 4 * h[i] * h[i])
            {
                iSum[i] += i;
                jSum[i] += j;
                iPosSum[i] += Vec3<double>{xi, yi, zi};
                jPosSum[i] += Vec3<double>{xj, yj, zj};
                ijPosDiffSum[i] += Vec3<double>{xij, yij, zij};
                d2Sum[i] += d2;
                hiSum[i] += h[i];
                hjSum[i] += h[j];
                viSum[i] += v[i];
                vjSum[i] += v[j];
                neighborsCount[i] += 1;
            }
        }
    }

    return Result(std::move(iSum), std::move(jSum), std::move(iPosSum), std::move(jPosSum), std::move(ijPosDiffSum),
                  std::move(d2Sum), std::move(hiSum), std::move(hjSum), std::move(viSum), std::move(vjSum),
                  std::move(neighborsCount));
}

void validate(const Result& expected, const Result& actual)
{
    auto validateElem = [](auto ei, auto ai, const char* name, std::size_t i)
    {
        if constexpr (std::is_same_v<decltype(ei), double>)
        {
            if (std::abs(ei - ai) > 1e-8)
                return testing::AssertionFailure() << name << "[" << i << "]: " << ai << " != " << ei;
        }
        else if constexpr (std::is_same_v<decltype(ei), Vec3<double>>)
        {
            for (unsigned d = 0; d < 3; ++d)
            {
                if (std::abs(ei[d] - ai[d]) > 1e-8)
                {
                    return testing::AssertionFailure()
                           << name << "[" << i << "]: {" << ai[0] << ", " << ai[1] << ", " << ai[2] << "} != {" << ei[0]
                           << ", " << ei[1] << ", " << ei[2] << "}";
                }
            }
        }
        else
        {
            if (ei != ai)
                return testing::AssertionFailure()
                       << name << "[" << i << "]: " << ai << " (actual) != " << ei << " (expected)";
        }
        return testing::AssertionSuccess();
    };

    util::for_each_tuple(
        [&validateElem](auto const& e, auto const& a, const char* name)
        {
            ASSERT_EQ(e.size(), a.size());
            for (std::size_t i = 0; i < e.size(); ++i)
                EXPECT_TRUE(validateElem(e[i], a[i], name, i));
        },
        expected, actual, resultNames);
}

auto initialData()
{
    const unsigned totalParticles = 997;
    const unsigned firstIParticle = 241;
    const unsigned lastIParticle  = 701;
    Box<double> box{0, 1, BoundaryType::periodic};
    RandomCoordinates<double, StrongKeyT> coords(totalParticles, box);

    thrust::universal_vector<double> x   = coords.x();
    thrust::universal_vector<double> y   = coords.y();
    thrust::universal_vector<double> z   = coords.z();
    thrust::universal_vector<KeyT> codes = coords.particleKeys();

    thrust::universal_vector<double> h(totalParticles), v(totalParticles);
    std::mt19937 gen(42);
    std::generate(h.begin(), h.end(), std::bind(std::uniform_real_distribution<double>(0.03, 0.15), std::ref(gen)));
    std::generate(v.begin(), v.end(), std::bind(std::uniform_real_distribution<double>(-100, 100), std::ref(gen)));

    auto [csTree, counts] = computeOctree(rawPtr(codes), rawPtr(codes) + totalParticles, 8);
    OctreeData<KeyT, CpuTag> octree;
    octree.resize(nNodes(csTree));
    updateInternalTree<KeyT>(csTree, octree.data());

    thrust::universal_vector<LocalIndex> layout(nNodes(csTree) + 1);
    std::exclusive_scan(counts.begin(), counts.end() + 1, layout.begin(), 0);

    thrust::universal_vector<Vec3<double>> centers(octree.numNodes), sizes(octree.numNodes);
    gsl::span<const KeyT> nodeKeys(rawPtr(octree.prefixes), octree.numNodes);
    nodeFpCenters(nodeKeys, rawPtr(centers), rawPtr(sizes), box);

    Result ref = reference(box, rawPtr(x), rawPtr(y), rawPtr(z), rawPtr(h), rawPtr(v), totalParticles, firstIParticle,
                           lastIParticle);

    OctreeNsView<double, KeyT> view{octree.numLeafNodes,
                                    rawPtr(octree.prefixes),
                                    rawPtr(octree.childOffsets),
                                    rawPtr(octree.internalToLeaf),
                                    rawPtr(octree.levelRange),
                                    nullptr,
                                    rawPtr(layout),
                                    rawPtr(centers),
                                    rawPtr(sizes)};

    auto treeData =
        std::make_tuple(std::move(codes), std::move(octree), std::move(layout), std::move(centers), std::move(sizes));

    return std::make_tuple(box, totalParticles, firstIParticle, lastIParticle, std::move(x), std::move(y), std::move(z),
                           std::move(h), std::move(v), std::move(treeData), std::move(view), std::move(ref));
}

template<ijloop::Neighborhood Neighborhood>
auto run(Neighborhood const& nb)
{
    auto [box, totalParticles, firstIParticle, lastIParticle, x, y, z, h, v, treeData, nsView, ref] = initialData();

    Result actual;
    const auto built = nb.build(nsView, box, totalParticles, firstIParticle, lastIParticle, rawPtr(x), rawPtr(y),
                                rawPtr(z), rawPtr(h));

    built.ijLoop(std::make_tuple(rawPtr(v)),
                 util::tupleMap(
                     [lastIParticle = lastIParticle](auto& vec)
                     {
                         vec.resize(lastIParticle);
                         return rawPtr(vec);
                     },
                     actual),
                 NeighborFun{}, ijloop::symmetry::asymmetric);
    checkGpuErrors(cudaDeviceSynchronize());

    validate(ref, actual);
}

TEST(IjLoop, Cpu) { run(ijloop::CpuDirectNeighborhood{ngmax}); }
TEST(IjLoop, GpuAlwaysTraverse) { run(ijloop::GpuAlwaysTraverseNeighborhood{ngmax}); }
TEST(IjLoop, GpuFullNbList) { run(ijloop::GpuFullNbListNeighborhood{ngmax}); }
TEST(IjLoop, GpuClusterNbList4x4WithoutSymmetryWithoutCompression)
{
    run(ijloop::GpuClusterNbListNeighborhood<>::withNcMax<ngmax>::withClusterSize<
        4, 4>::withoutSymmetry::withoutCompression{});
}
TEST(IjLoop, GpuClusterNbList4x4WithSymmetryWithoutCompression)
{
    run(ijloop::GpuClusterNbListNeighborhood<>::withNcMax<ngmax>::withClusterSize<
        4, 4>::withSymmetry::withoutCompression{});
}
TEST(IjLoop, GpuClusterNbList4x4WithoutSymmetryWithCompression)
{
    run(ijloop::GpuClusterNbListNeighborhood<>::withNcMax<ngmax>::withClusterSize<
        4, 4>::withoutSymmetry::withCompression<8>{});
}
TEST(IjLoop, GpuClusterNbList4x4WithSymmetryWithCompression)
{
    run(ijloop::GpuClusterNbListNeighborhood<>::withNcMax<ngmax>::withClusterSize<4, 4>::withSymmetry::withCompression<
        8>{});
}
TEST(IjLoop, GpuClusterNbList8x4WithoutSymmetryWithoutCompression)
{
    run(ijloop::GpuClusterNbListNeighborhood<>::withNcMax<ngmax>::withClusterSize<
        8, 4>::withoutSymmetry::withoutCompression{});
}
TEST(IjLoop, GpuClusterNbList8x4WithSymmetryWithoutCompression)
{
    run(ijloop::GpuClusterNbListNeighborhood<>::withNcMax<ngmax>::withClusterSize<
        8, 4>::withSymmetry::withoutCompression{});
}
TEST(IjLoop, GpuClusterNbList8x4WithoutSymmetryWithCompression)
{
    run(ijloop::GpuClusterNbListNeighborhood<>::withNcMax<ngmax>::withClusterSize<
        8, 4>::withoutSymmetry::withCompression<8>{});
}
TEST(IjLoop, GpuClusterNbList8x4WithSymmetryWithCompression)
{
    run(ijloop::GpuClusterNbListNeighborhood<>::withNcMax<ngmax>::withClusterSize<8, 4>::withSymmetry::withCompression<
        8>{});
}
