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

class NibbleBuffer
{
public:
    NibbleBuffer(void* buffer, unsigned bufferSize, bool clear = true)
        : buffer_(reinterpret_cast<std::uint8_t*>(buffer) + sizeof(unsigned))
        , maxSize_(2 * (bufferSize - sizeof(unsigned)))
    {
        if (clear) sizeRef() = 0;
    }

    HOST_DEVICE_INLINE void push_back(std::uint8_t value)
    {
        assert(value < 16);
        const auto [byte, offset] = std::div(sizeRef()++, 2);
        const std::uint8_t mask   = offset ? 0x0f : 0xf0;
        buffer_[byte]             = (buffer_[byte] & mask) | (value << (4 * offset));
    }

    HOST_DEVICE_INLINE std::uint8_t pop_back()
    {
        std::uint8_t value = (*this)[size() - 1];
        --sizeRef();
        return value;
    }

    HOST_DEVICE_INLINE std::uint8_t operator[](unsigned index) const
    {
        assert(index < size());
        const auto [byte, offset] = std::div(index, 2);
        return (buffer_[byte] >> (4 * offset)) & 0xf;
    }

    HOST_DEVICE_INLINE unsigned size() const { return *(reinterpret_cast<const unsigned*>(buffer_) - 1); }
    HOST_DEVICE_INLINE unsigned max_size() const { return maxSize_; }

private:
    HOST_DEVICE_INLINE unsigned& sizeRef() { return *(reinterpret_cast<unsigned*>(buffer_) - 1); }

    std::uint8_t* buffer_;
    unsigned maxSize_;
};

class NeighborListCompressorIterator;

class NeighborListCompressor
{
public:
    HOST_DEVICE_INLINE NeighborListCompressor(void* buffer, unsigned bufferSize)
        : nibbles_(buffer, bufferSize, true)
    {
    }

    NeighborListCompressor(const NeighborListCompressor&)            = delete;
    NeighborListCompressor& operator=(const NeighborListCompressor&) = delete;
    NeighborListCompressor(NeighborListCompressor&&)                 = default;
    NeighborListCompressor& operator=(NeighborListCompressor&&)      = default;

    HOST_DEVICE_INLINE bool push_back(std::uint32_t nbIndex)
    {
        const std::uint32_t value = nbIndex - prevNbIndex_;
        prevNbIndex_              = nbIndex;

        if (value == 1)
        {
            if (ones_) { nibbles_.push_back(nibbles_.pop_back() + 1); }
            else
            {
                if (nibbles_.size() >= nibbles_.max_size()) return false;
                nibbles_.push_back(9);
            }
            assert(nibbles_[nibbles_.size() - 1] == ones_ + 9);
            if (++ones_ >= 7) ones_ = 0;
        }
        else
        {
            const int n = std::max((std::bit_width(value) + 3) / 4, 1);
            if (nibbles_.size() + 1 + n >= nibbles_.max_size()) return false;
            nibbles_.push_back(n);
            for (int i = 0; i < n; ++i)
                nibbles_.push_back((value >> (4 * i)) & 0xf);
            ones_ = 0;
        }
        ++nNeighbors_;
        return true;
    }

    HOST_DEVICE_INLINE unsigned size() const { return nNeighbors_; }
    HOST_DEVICE_INLINE unsigned nbytes() const { return (nibbles_.size() + 1) / 2; }

    NeighborListCompressorIterator begin() const;
    NeighborListCompressorIterator end() const;

private:
    friend class NeighborListCompressorIterator;

    NibbleBuffer nibbles_;
    unsigned nNeighbors_ = 0, ones_ = 0;
    std::uint32_t prevNbIndex_ = 0;
};

class NeighborListCompressorIterator
{
public:
    using difference_type   = int;
    using value_type        = std::uint32_t;
    using pointer           = void;
    using reference         = void;
    using iterator_category = std::input_iterator_tag;

    NeighborListCompressorIterator(const NeighborListCompressorIterator& other)            = default;
    NeighborListCompressorIterator(NeighborListCompressorIterator&& other)                 = default;
    NeighborListCompressorIterator& operator=(const NeighborListCompressorIterator& other) = default;
    NeighborListCompressorIterator& operator=(NeighborListCompressorIterator&& other)      = default;

    HOST_DEVICE_INLINE static NeighborListCompressorIterator begin(const NeighborListCompressor& compressor)
    {
        return {&compressor, 0};
    }
    HOST_DEVICE_INLINE static NeighborListCompressorIterator end(const NeighborListCompressor& compressor)
    {
        return {&compressor, compressor.nibbles_.size()};
    }

    HOST_DEVICE_INLINE bool operator==(const NeighborListCompressorIterator& other) const
    {
        return i_ == other.i_ && onesLeft_ == other.onesLeft_;
    }

    HOST_DEVICE_INLINE bool operator!=(const NeighborListCompressorIterator& other) const { return !(*this == other); }

    HOST_DEVICE_INLINE std::uint32_t operator*() const { return value_; }

    HOST_DEVICE_INLINE NeighborListCompressorIterator& operator++()
    {
        if (onesLeft_ > 0)
        {
            --onesLeft_;
            ++value_;
        }
        else
        {
            i_ = iNext_;
            if (i_ < compressor_->nibbles_.size()) readValue();
        }
        return *this;
    }

    HOST_DEVICE_INLINE std::uint32_t operator++(int)
    {
        std::uint32_t value = **this;
        ++*this;
        return value;
    }

private:
    HOST_DEVICE_INLINE NeighborListCompressorIterator(const NeighborListCompressor* compressor, unsigned i)
        : compressor_(compressor)
        , i_(i)
    {
        if (i == 0) readValue();
    }

    HOST_DEVICE_INLINE void readValue()
    {
        iNext_                 = i_;
        const int n            = compressor_->nibbles_[iNext_++];
        std::uint32_t previous = value_;
        if (n > 8)
        {
            value_    = ++previous;
            onesLeft_ = n - 9;
        }
        else
        {
            value_ = 0;
            for (int j = 0; j < n; ++j)
            {
                assert(iNext_ < compressor_->nibbles_.size());
                const std::uint32_t d = compressor_->nibbles_[iNext_++];
                value_ |= d << (4 * j);
            }
            value_ += previous;
            onesLeft_ = 0;
        }
    }

    const NeighborListCompressor* compressor_ = nullptr;
    unsigned i_ = 0, iNext_, onesLeft_ = 0;
    std::uint32_t value_ = 0;
};

HOST_DEVICE_INLINE NeighborListCompressorIterator NeighborListCompressor::begin() const
{
    return NeighborListCompressorIterator::begin(*this);
}

HOST_DEVICE_INLINE NeighborListCompressorIterator NeighborListCompressor::end() const
{
    return NeighborListCompressorIterator::end(*this);
}

} // namespace cstone
