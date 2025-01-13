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
 * @brief Neighbor search on CPU
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <algorithm>
#include <utility>
#include <vector>

#include "cstone/traversal/ijloop/ijloop.hpp"
#include "cstone/tree/octree.hpp"

namespace cstone::ijloop
{

namespace detail
{
template<class Tc, class KeyType, class Th>
struct CpuDirectNeighborhoodImpl
{
    OctreeNsView<Tc, KeyType> tree;
    Box<Tc> box;
    LocalIndex firstBody, lastBody;
    const Tc *x, *y, *z;
    const Th* h;
    unsigned ngmax;

    template<class... In, class... Out, class Interaction, class Symmetry = symmetry::Asymmetric>
    void ijLoop(std::tuple<const In*...> const& input,
                std::tuple<Out*...> const& output,
                Interaction&& interaction,
                Symmetry = symmetry::asymmetric)
    {
#pragma omp parallel
        {
            std::vector<LocalIndex> neighbors(ngmax);
            const bool anyPbc = box.boundaryX() == BoundaryType::periodic | box.boundaryY() == BoundaryType::periodic |
                                box.boundaryZ() == BoundaryType::periodic;

#pragma omp for
            for (LocalIndex i = firstBody; i < lastBody; ++i)
            {
                const auto iData = loadParticleData(x, y, z, h, input, i);

                const unsigned nbs = std::min(findNeighbors(i, x, y, z, h, tree, box, ngmax, neighbors.data()), ngmax);
                auto result        = interaction(iData, iData, Tc(0));
                for (unsigned nb = 0; nb < nbs; ++nb)
                {
                    const LocalIndex j = neighbors[nb];
                    const auto jData   = loadParticleData(x, y, z, h, input, j);

                    const Tc distSq =
                        anyPbc ? distanceSq<true>(std::get<1>(jData), std::get<2>(jData), std::get<3>(jData),
                                                  std::get<1>(iData), std::get<2>(iData), std::get<3>(iData), box)
                               : distanceSq<false>(std::get<1>(jData), std::get<2>(jData), std::get<3>(jData),
                                                   std::get<1>(iData), std::get<2>(iData), std::get<3>(iData), box);

                    updateResult(result, interaction(iData, jData, distSq));
                }

                storeParticleData(output, i, result);
            }
        }
    }
};
} // namespace detail

struct CpuDirectNeighborhood
{
    unsigned ngmax;

    template<class Tc, class KeyType, class Th>
    detail::CpuDirectNeighborhoodImpl<Tc, KeyType, Th> build(OctreeNsView<Tc, KeyType> tree,
                                                             Box<Tc> box,
                                                             LocalIndex firstBody,
                                                             LocalIndex lastBody,
                                                             const Tc* x,
                                                             const Tc* y,
                                                             const Tc* z,
                                                             const Th* h)
    {
        return {std::move(tree), std::move(box), firstBody, lastBody, x, y, z, h, ngmax};
    }
};

} // namespace cstone::ijloop
