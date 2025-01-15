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
 * @brief Basic functionality for looping over particles and neighbors
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <tuple>
#include <type_traits>

#include "cstone/tree/definitions.h"
#include "cstone/sfc/box.hpp"
#include "cstone/util/tuple_util.hpp"

namespace cstone::ijloop
{

namespace symmetry
{
struct Asymmetric
{
};
struct Even
{
};
struct Odd
{
};

inline constexpr Asymmetric asymmetric = {};
inline constexpr Even even             = {};
inline constexpr Odd odd               = {};
} // namespace symmetry

template<class... Ts>
inline constexpr std::tuple<const Ts*...> makeConstRestrict(std::tuple<Ts*...> input)
{
    // TODO: __restrict__
    return {input};
}

template<class Tc, class Th, class... Ts>
inline constexpr std::tuple<LocalIndex, Vec3<Tc>, Th, Ts...> loadParticleData(
    const Tc* x, const Tc* y, const Tc* z, const Th* h, std::tuple<const Ts*...> const& input, LocalIndex index)
{
    const Vec3<Tc> pos = {x[index], y[index], z[index]};
    return std::tuple_cat(std::make_tuple(index, pos, h[index]),
                          util::tupleMap([index](auto const* ptr) { return ptr[index]; }, input));
}

template<class Tc, class Th, class... Ts>
inline constexpr bool requiresPbcHandling(Box<Tc> const& box, std::tuple<LocalIndex, Vec3<Tc>, Th, Ts...> const& iData)
{
    if (box.boundaryX() != BoundaryType::periodic & box.boundaryY() != BoundaryType::periodic &
        box.boundaryZ() != BoundaryType::periodic)
        return false;
    const Vec3<Tc>& iPos = std::get<1>(iData);
    const Tc twoHi       = Tc(2) * std::get<2>(iData);
    return !insideBox(iPos, {twoHi, twoHi, twoHi}, box);
}

template<class Tc, class Th, class... Ts>
inline constexpr std::tuple<Vec3<Tc>, Tc> posDiffAndDistSq(bool usePbc,
                                                           Box<Tc> const& box,
                                                           std::tuple<LocalIndex, Vec3<Tc>, Th, Ts...> const& iData,
                                                           std::tuple<LocalIndex, Vec3<Tc>, Th, Ts...> const& jData)
{
    const Vec3<Tc>& iPos = std::get<1>(iData);
    const Vec3<Tc>& jPos = std::get<1>(jData);
    Vec3<Tc> ijPosDiff   = iPos - jPos;
    if (usePbc)
    {
        ijPosDiff[0] -= (box.boundaryX() == BoundaryType::periodic) * box.lx() * std::rint(ijPosDiff[0] * box.ilx());
        ijPosDiff[1] -= (box.boundaryY() == BoundaryType::periodic) * box.ly() * std::rint(ijPosDiff[1] * box.ily());
        ijPosDiff[2] -= (box.boundaryZ() == BoundaryType::periodic) * box.lz() * std::rint(ijPosDiff[2] * box.ilz());
    }
    return {ijPosDiff, norm2(ijPosDiff)};
}

template<class... Ts>
inline constexpr void updateResult(std::tuple<Ts...>& result, std::tuple<Ts...> const& value)
{
    util::for_each_tuple([](auto& r, auto const& v) { r += v; }, result, value);
}

template<class... Ts>
inline constexpr void
storeParticleData(std::tuple<Ts*...> const& output, LocalIndex index, std::tuple<Ts...> const& value)
{
    util::for_each_tuple([index](auto* ptr, auto const& v) { ptr[index] = v; }, output, value);
}

} // namespace cstone::ijloop
