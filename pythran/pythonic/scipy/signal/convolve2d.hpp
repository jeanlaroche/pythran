#ifndef PYTHONIC_SCIPY_SIGNAL_CONVOLVE2D_HPP
#define PYTHONIC_SCIPY_SIGNAL_CONVOLVE2D_HPP

#include "pythonic/include/scipy/signal/convolve2d.hpp"
#include "pythonic/numpy/dot.hpp"
#include "pythonic/numpy/conjugate.hpp"
#include "pythonic/numpy/asarray.hpp"
#include "pythonic/types/ndarray.hpp"
#if defined(PYTHRAN_BLAS_ATLAS) || defined(PYTHRAN_BLAS_SATLAS)
extern "C" {
#endif
#include <cblas.h>
#if defined(PYTHRAN_BLAS_ATLAS) || defined(PYTHRAN_BLAS_SATLAS)
}
#endif


#define min(A, B) ((A < B) ? (A) : (B))
#define max(A, B) ((A > B) ? (A) : (B))

PYTHONIC_NS_BEGIN

namespace scipy
{
  namespace signal
  {

    // n,m input (and output) dim
    // r,q kernel dim
    // Use this if r and q are both small
    void convol_loop(double *im, double *kernel, double *out, unsigned n,
                     unsigned m, unsigned r, unsigned q)
        __attribute__((noinline))
    {
      for (unsigned k = 0; k < r; ++k)                 // loop over kernel cols
        for (unsigned l = 0; l < q; ++l)               // loop over kernel rows
          for (unsigned i = r / 2; i < n - r / 2; ++i) // loop over in cols
            for (unsigned j = q / 2; j < m - q / 2; ++j) // loop over in rows
              out[i * m + j] +=
                  im[(i + k - r / 2) * m + (j + l - q / 2)] * kernel[k * q + l];
    }

    // Use this if r<q
    void convol_dot_rows(double *im, double *kernel, double *out, unsigned n,
                         unsigned m, unsigned r, unsigned q)
        __attribute__((noinline))
    {
      for (unsigned k = 0; k < r; ++k)                 // loop over kernel cols
        for (unsigned i = r / 2; i < n - r / 2; ++i)   // loop over in cols
          for (unsigned j = q / 2; j < m - q / 2; ++j) // loop over in rows
            out[i * m + j] +=
                cblas_ddot(q, im + (i + k - r / 2) * m + j - q / 2, 1,
                           kernel + k * q, 1); // <- kernel rows
    }

    // Use this if r>q
    void convol_dot_cols(double *im, double *kernel, double *out, unsigned n,
                         unsigned m, unsigned r, unsigned q)
        __attribute__((noinline))
    {
      for (unsigned l = 0; l < q; ++l)                 // loop over kernel rows
        for (unsigned i = r / 2; i < n - r / 2; ++i)   // loop over in cols
          for (unsigned j = q / 2; j < m - q / 2; ++j) // loop over in rows
            out[i * m + j] +=
                cblas_ddot(r, im + (i - r / 2) * m + j + l - q / 2, m,
                           kernel + l, q); // <- kernel cols
    }

    // This handles the convolution edges
    void convol_edge(double *im, double *kernel, double *out, int n, int m,
                     int r, int q) __attribute__((noinline))
    {
      for (int i = 0; i < n; ++i)   // loop over in cols
        for (int j = 0; j < m; ++j) // loop over in rows
        {
          if (j >= q / 2 && i >= r / 2 && i < n - r / 2 && j < m - q / 2)
            j = m - q / 2;

          // std::cout << "i " << i << " J " << j <<" ";
          // std::cout << "k " << max(0, r / 2 - i) << " -- " << min(r, n - i +
          // r / 2) <<" ";
          // std::cout << "l " << max(0, q / 2 - j) << " -- " << min(q, m - j +
          // q / 2) <<"\n";

          for (int k = max(0, r / 2 - i); k < min(r, n - i + r / 2);
               ++k) // loop over kernel cols
            for (int l = max(0, q / 2 - j); l < min(q, m - j + q / 2);
                 ++l) // loop over kernel rows
              out[i * m + j] +=
                  im[(i + k - r / 2) * m + (j + l - q / 2)] * kernel[k * q + l];
        }
    }

    template <class A, class B>
    types::ndarray<typename A::dtype, types::pshape<long, long>>
    convolve2d(A const &inA, B const &inB)
    {
      auto shapeA = sutils::array(inA.shape());
      auto shapeB = sutils::array(inB.shape());

      long NA = shapeA[0];
      long NB = shapeB[0];
      types::ndarray<typename A::dtype, types::pshape<long, long>> out = {
          shapeA, (typename A::dtype)(0)};

      auto inA_ = numpy::functor::asarray{}(inA);
      auto inB_ = numpy::functor::asarray{}(inB);
      auto a = out.buffer;
      auto b = inA_.buffer;
      auto c = inB_.buffer;

      // std::cout << "CONV0\n";
      convol_edge(inA_.buffer, inB_.buffer, out.buffer, shapeA[0], shapeA[1],
                  shapeB[0], shapeB[1]);
      // std::cout << "CONV1\n";

      // Using cblas for the dot product is only good if the kernel size is
      // large enough in at least 1 dim
      int startBlas = 15;
      if (shapeB[0] < startBlas && shapeB[1] < startBlas)
        convol_loop(inA_.buffer, inB_.buffer, out.buffer, shapeA[0], shapeA[1],
                    shapeB[0], shapeB[1]);
      else if (shapeB[0] < shapeB[1])
        convol_dot_rows(inA_.buffer, inB_.buffer, out.buffer, shapeA[0],
                        shapeA[1], shapeB[0], shapeB[1]);
      else
        convol_dot_cols(inA_.buffer, inB_.buffer, out.buffer, shapeA[0],
                        shapeA[1], shapeB[0], shapeB[1]);
      // std::cout << "DONE\n";

      return out;
    }

    NUMPY_EXPR_TO_NDARRAY0_IMPL(convolve2d)
  }
}
PYTHONIC_NS_END

#endif
