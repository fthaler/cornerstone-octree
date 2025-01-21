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
#include <vector>

#include "cstone/traversal/ijloop/cpu.hpp"

#include "../../coord_samples/random.hpp"

#include "gtest/gtest.h"

using namespace cstone;

constexpr unsigned ngmax = 160;

using StrongKeyT = HilbertKey<std::uint64_t>;
using KeyT       = StrongKeyT::ValueType;

struct NeighborFun
{
    template<class ParticleData>
    constexpr __host__ __device__ auto
    operator()(ParticleData const& iData, ParticleData const& jData, Vec3<double> ijPosDiff, double distSq) const
    {
        const auto [i, iPos, hi, vi] = iData;
        const auto [j, jPos, hj, vj] = jData;
        return std::make_tuple(i, j, iPos, jPos, ijPosDiff, distSq, hi, hj, vi, vj, 1u);
    }
};

using Result                      = std::tuple<std::vector<LocalIndex>,   // iSum
                                               std::vector<LocalIndex>,   // jSum
                                               std::vector<Vec3<double>>, // iPosSum
                                               std::vector<Vec3<double>>, // jPosSum
                                               std::vector<Vec3<double>>, // ijPosDiffSum
                                               std::vector<double>,       // distSqSum
                                               std::vector<double>,       // hiSum
                                               std::vector<double>,       // hjSum
                                               std::vector<double>,       // viSum
                                               std::vector<double>,       // vjSum
                                               std::vector<unsigned>      // neighborsCount
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
                 const LocalIndex firstBody,
                 const LocalIndex lastBody)
{
    std::vector<LocalIndex> iSum(lastBody), jSum(lastBody);
    std::vector<Vec3<double>> iPosSum(lastBody), jPosSum(lastBody), ijPosDiffSum(lastBody);
    std::vector<double> d2Sum(lastBody), hiSum(lastBody), hjSum(lastBody), viSum(lastBody), vjSum(lastBody);
    std::vector<unsigned> neighborsCount(lastBody);

    for (unsigned i = firstBody; i < lastBody; ++i)
    {
        for (unsigned j = 0; j < lastBody; ++j)
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
            if (ei != ai) return testing::AssertionFailure() << name << "[" << i << "]: " << ai << " != " << ei;
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
    const unsigned n = 1000;
    Box<double> box{0, 1, BoundaryType::periodic};
    RandomCoordinates<double, StrongKeyT> coords(n, box);

    auto x     = std::move(coords.x());
    auto y     = std::move(coords.y());
    auto z     = std::move(coords.z());
    auto codes = std::move(coords.particleKeys());

    std::vector<double> h(n), v(n);
    std::mt19937 gen(42);
    std::generate(h.begin(), h.end(), std::bind(std::uniform_real_distribution<double>(0.03, 0.15), std::ref(gen)));
    std::generate(v.begin(), v.end(), std::bind(std::uniform_real_distribution<double>(-100, 100), std::ref(gen)));

    auto [csTree, counts] = computeOctree(codes.data(), codes.data() + n, 64);
    OctreeData<KeyT, CpuTag> octree;
    octree.resize(nNodes(csTree));
    updateInternalTree<KeyT>(csTree, octree.data());

    std::vector<LocalIndex> layout(nNodes(csTree) + 1);
    std::exclusive_scan(counts.begin(), counts.end() + 1, layout.begin(), 0);

    std::vector<Vec3<double>> centers(octree.numNodes), sizes(octree.numNodes);
    gsl::span<const KeyT> nodeKeys(octree.prefixes.data(), octree.numNodes);
    nodeFpCenters(nodeKeys, centers.data(), sizes.data(), box);

    Result ref = reference(box, coords.x().data(), coords.y().data(), coords.z().data(), h.data(), v.data(), n / 4, n);

    return std::make_tuple(box, n / 4, n, x, y, z, h, v, codes, octree, layout, centers, sizes, ref);
}

OctreeNsView<double, KeyT> cpuView(OctreeData<KeyT, CpuTag> const& octree,
                                   std::vector<LocalIndex> const& layout,
                                   std::vector<Vec3<double>> const& centers,
                                   std::vector<Vec3<double>> const& sizes)
{
    return {octree.numLeafNodes,
            octree.prefixes.data(),
            octree.childOffsets.data(),
            octree.internalToLeaf.data(),
            octree.levelRange.data(),
            nullptr,
            layout.data(),
            centers.data(),
            sizes.data()};
}

TEST(IjLoop, CPU)
{
    auto [box, firstBody, lastBody, x, y, z, h, v, codes, octree, layout, centers, sizes, ref] = initialData();
    auto nsView = cpuView(octree, layout, centers, sizes);

    Result actual;
    ijloop::CpuDirectNeighborhood{ngmax}
        .build(nsView, box, firstBody, lastBody, x.data(), y.data(), z.data(), h.data())
        .ijLoop(std::make_tuple(v.data()),
                util::tupleMap(
                    [lastBody = lastBody](auto& vec)
                    {
                        vec.resize(lastBody);
                        return vec.data();
                    },
                    actual),
                NeighborFun{}, ijloop::symmetry::asymmetric);

    validate(ref, actual);
}
