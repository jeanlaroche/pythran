#ifndef PYTHONIC_NUMPY_SHAPE_HPP
#define PYTHONIC_NUMPY_SHAPE_HPP

#include "pythonic/include/numpy/shape.hpp"

#include "pythonic/utils/proxy.hpp"
#include "pythonic/types/ndarray.hpp"

namespace pythonic {

    namespace numpy {

        template <class E>
            auto shape(E const& e) -> decltype(e.shape())
            {
                return e.shape();
            }

        PROXY_IMPL(pythonic::numpy, shape)
    }
}

#endif
