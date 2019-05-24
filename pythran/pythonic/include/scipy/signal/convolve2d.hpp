#ifndef PYTHONIC_INCLUDE_SCIPY_SIGNAL_CONVOLVE2D_HPP
#define PYTHONIC_INCLUDE_SCIPY_SIGNAL_CONVOLVE2D_HPP

#include "pythonic/include/utils/functor.hpp"
#include "pythonic/include/types/ndarray.hpp"

PYTHONIC_NS_BEGIN

namespace scipy
{
  namespace signal
  {
      // This is the equivalent of Conv2D in tensorflow. A is batch x size1 x size2 x nchannels_in
      // B is nchannel_out x ksize1 x ksize2 x nchannels_in
      // increments must be 1,inc1,inc2,1
      template <class A, class B>
      types::ndarray<typename A::dtype, types::pshape<long,long,long,long>>
      convolve2d(A const &inA, B const &inB, types::ndarray<long, types::pshape<long>> const &increments);

      template <class A, class B>
      types::ndarray<typename A::dtype, types::pshape<long,long>>
      convolve2d(A const &inA, B const &inB);

//      template <class A, class B, typename U>
//      types::ndarray<typename A::dtype, types::pshape<long,long>>
//      convolve2d(A const &inA, B const &inB, U mode);
//
//      template <class A, class B, typename U, typename V>
//      types::ndarray<typename A::dtype, types::pshape<long,long>>
//      convolve2d(A const &inA, B const &inB, U mode, V boundary);
//
//      template <class A, class B, typename U, typename V, typename W>
//      types::ndarray<typename A::dtype, types::pshape<long,long>>
//      convolve2d(A const &inA, B const &inB, U mode, V boundary, W fillvalue);

      NUMPY_EXPR_TO_NDARRAY0_DECL(convolve2d)
      DEFINE_FUNCTOR(pythonic::scipy::signal, convolve2d)
  }
}
PYTHONIC_NS_END

#endif
