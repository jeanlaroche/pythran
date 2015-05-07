#ifndef PYTHONIC_NUMPY_COUNT_NONZERO_HPP
#define PYTHONIC_NUMPY_COUNT_NONZERO_HPP

#include "pythonic/include/numpy/count_nonzero.hpp"

#include "pythonic/utils/proxy.hpp"
#include "pythonic/types/ndarray.hpp"

namespace pythonic {

    namespace numpy {

        template<class dtype, class E>
            auto _count_nonzero(E begin, E end, size_t& count, utils::int_<1>)
            -> typename std::enable_if<std::is_same<dtype, bool>::value>::type
            {
                for(; begin != end; ++begin)
                    // Behaviour defined in the standard
                    count += *begin;
            }

        template<class dtype, class E>
            auto _count_nonzero(E begin, E end, size_t& count, utils::int_<1>)
            -> typename std::enable_if<!std::is_same<dtype, bool>::value>::type
            {
                for(; begin != end; ++begin)
                    if (*begin != static_cast<dtype>(0))
                        ++count;
            }

        template<class dtype, class E, size_t N>
            void _count_nonzero(E begin, E end, size_t& count, utils::int_<N>)
            {
                for(; begin != end; ++begin)
                    _count_nonzero<dtype>((*begin).begin(), (*begin).end(), count, utils::int_<N - 1>());
            }

        template<class E>
            size_t count_nonzero(E const& array)
            {
                size_t count(0);
                _count_nonzero<typename E::dtype>(array.begin(),
                                                  array.end(),
                                                  count,
                                                  utils::int_<types::numpy_expr_to_ndarray<E>::N>());
                return count;
            }

        PROXY_IMPL(pythonic::numpy, count_nonzero);

    }

}

#endif
