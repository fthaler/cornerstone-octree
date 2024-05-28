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

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstdlib>

#include "cstone/cuda/annotation.hpp"

namespace cstone
{

class NeighborListCompressor
{
public:
    HOST_DEVICE_INLINE NeighborListCompressor(void* buffer, std::size_t maxSize)
        : buffer_(reinterpret_cast<std::uint8_t*>(buffer))
        , maxNibbles_(2 * maxSize)
    {
    }

    NeighborListCompressor(const NeighborListCompressor&)            = delete;
    NeighborListCompressor& operator=(const NeighborListCompressor&) = delete;
    NeighborListCompressor(NeighborListCompressor&&)                 = default;
    NeighborListCompressor& operator=(NeighborListCompressor&&)      = default;

    HOST_DEVICE_INLINE bool add(std::uint32_t nbIndex)
    {
        const std::uint32_t value = nbIndex - prevNbIndex_;
        prevNbIndex_              = nbIndex;

        if (value == 1)
        {
            if (ones_) { writeNibble(nNibbles_ - 1, readNibble(nNibbles_ - 1) + 1); }
            else
            {
                if (!pushBackNibble(9)) return false;
            }
            assert(readNibble(nNibbles_ - 1) == ones_ + 9);
            if (++ones_ >= 7) ones_ = 0;
        }
        else
        {
            const int n = std::max((std::bit_width(value) + 3) / 4, 1);
            if (!pushBackNibble(n)) return false;
            for (int i = 0; i < n; ++i)
                if (!pushBackNibble((value >> (4 * i)) & 0xf)) return false;
            ones_ = 0;
        }
        ++nNeighbors_;
        return true;
    }

    HOST_DEVICE_INLINE void decompress(std::uint32_t* buffer, std::size_t maxNeighbors) const
    {
        std::size_t i          = 0;
        std::uint32_t previous = 0;
        while (i < nNibbles_)
        {
            const int n = readNibble(i++);
            if (n > 8)
            {
                for (int j = 0; j < n - 8; ++j)
                {
                    if (!maxNeighbors--) return;
                    *(buffer++) = ++previous;
                }
            }
            else
            {
                std::uint32_t value = 0;
                for (int j = 0; j < n; ++j)
                {
                    if (i >= nNibbles_) return;
                    const std::uint32_t d = readNibble(i++);
                    value |= d << (4 * j);
                }
                previous += value;
                if (!maxNeighbors--) return;
                *(buffer++) = previous;
            }
        }
    }

    HOST_DEVICE_INLINE std::size_t size() const { return nNeighbors_; }
    HOST_DEVICE_INLINE std::size_t nbytes() const { return (nNibbles_ + 1) / 2; }

private:
    HOST_DEVICE_INLINE void writeNibble(std::size_t index, std::uint8_t value)
    {
        assert(index < nNibbles_);
        assert(nNibbles_ <= maxNibbles_);
        assert(value < 16);
        const auto [byte, offset] = std::div(index, 2);
        const std::uint8_t mask   = offset ? 0x0f : 0xf0;
        buffer_[byte]             = (buffer_[byte] & mask) | (value << (4 * offset));
    }

    HOST_DEVICE_INLINE std::uint8_t readNibble(std::size_t index) const
    {
        assert(index < nNibbles_);
        assert(nNibbles_ <= maxNibbles_);
        const auto [byte, offset] = std::div(index, 2);
        return (buffer_[byte] >> (4 * offset)) & 0xf;
    }

    HOST_DEVICE_INLINE bool pushBackNibble(std::uint8_t value)
    {
        if (nNibbles_ >= maxNibbles_) return false;
        writeNibble(nNibbles_++, value);
        return true;
    }

    std::uint8_t* buffer_;
    std::size_t maxNibbles_, nNibbles_ = 0, nNeighbors_ = 0, ones_ = 0;
    std::uint32_t prevNbIndex_ = 0;
};

} // namespace cstone
