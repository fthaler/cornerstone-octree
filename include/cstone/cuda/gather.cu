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

/*! \file
 * \brief  Exposes gather functionality to reorder arrays by a map
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/sort.h>

#include "gather.cuh"

template<class T, class I>
class DeviceMemory
{
public:
    template<class TD>
    using DeviceIterator = typename thrust::device_vector<TD>::iterator;
    template<class TD>
    using DeviceConstIterator = typename thrust::device_vector<TD>::const_iterator;

    DeviceMemory(const I* map_first, const I* map_last)
        : d_ordering_(typename std::vector<I>::const_iterator(map_first),
                      typename std::vector<I>::const_iterator(map_last)),
          d_source_(map_last - map_first),
          d_destination_(map_last - map_first)
    {}

    DeviceConstIterator<I> ordering() const { return d_ordering_.begin(); }

    DeviceIterator<T> source()      { return d_source_.begin(); }
    DeviceIterator<T> destination() { return d_destination_.begin(); }

private:
    thrust::device_vector<I> d_ordering_;
    thrust::device_vector<T> d_source_;
    thrust::device_vector<T> d_destination_;
};


template<class T, class I>
DeviceGather<T, I>::DeviceGather()
{}

template<class T, class I>
void DeviceGather<T, I>::setReorderMap(const I* map_first, const I* map_last)
{
    mapSize_      = map_last - map_first;
    deviceMemory_ = std::make_unique<DeviceMemory<T, I>>(map_first, map_last);
}

template<class T, class I>
DeviceGather<T, I>::~DeviceGather() {}

template<class T, class I>
void DeviceGather<T, I>::operator()(T* values)
{
    // upload to device
    thrust::copy(values, values + mapSize_, deviceMemory_->source());

    // reorder on device
    thrust::gather(thrust::device, deviceMemory_->ordering(), deviceMemory_->ordering() + mapSize_,
                   deviceMemory_->source(), deviceMemory_->destination());

    // download to host
    thrust::copy(deviceMemory_->destination(), deviceMemory_->destination() + mapSize_, values);
}

template class DeviceGather<float,  unsigned>;
template class DeviceGather<float,  uint64_t>;
template class DeviceGather<double, unsigned>;
template class DeviceGather<double, uint64_t>;
