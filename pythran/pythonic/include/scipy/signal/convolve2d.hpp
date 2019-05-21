#ifndef PYTHONIC_INCLUDE_SCIPY_SIGNAL_CONVOLVE2D_HPP
#define PYTHONIC_INCLUDE_SCIPY_SIGNAL_CONVOLVE2D_HPP

#include "pythonic/include/utils/functor.hpp"
#include "pythonic/include/types/ndarray.hpp"

PYTHONIC_NS_BEGIN

namespace scipy
{
  namespace signal
  {
      template <class A, class B>
      types::ndarray<typename A::dtype, types::pshape<long,long>>
      convolve2d(A const &inA, B const &inB);

      NUMPY_EXPR_TO_NDARRAY0_DECL(convolve2d)
      DEFINE_FUNCTOR(pythonic::scipy::signal, convolve2d)
  }
}
PYTHONIC_NS_END

#endif
