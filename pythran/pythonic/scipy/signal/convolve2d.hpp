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
    // input of size a,sI,sJ,d (b,c)=image, d = channel, a = batch
    // kernel of size s,si,sj,d (si,sj)=image, d=channel_in, s=channel_out
    // output of size a,soi,soj,s
    // This is a direct implementation of the Conv2D function from tfdeploy. It
    // uses cblas_ddot to collapse the two inner loops (over the output
    // channels, and
    // the kernel inner dimension).
    void convol_loop4D(double *im, double *kernel, double *out, int a, int sI,
                       int sJ, int d, int s, int si, int sj, int inci, int incj,
                       int soi, int soj) __attribute__((noinline))
    {
      int ipad = ((soi - 1) * inci + si - sI) / 2;
      if (ipad < 0)
        ipad = 0;
      int jpad = ((soj - 1) * incj + sj - sJ) / 2;
      if (jpad < 0)
        jpad = 0;

      for (int b = 0; b < a; b++)
        for (int i = 0; i < soi; i++) {
          int starti = 0;
          int ui = inci * i - ipad;
          if (ui < 0) {
            starti = ipad - inci * i;
            ui = 0;
          }
          int endi = si < sI - ui + starti ? si : sI - ui + starti;
          for (int j = 0; j < soj; j++) {
            int startj = 0;
            int uj = incj * j - jpad;
            if (uj < 0) {
              startj = jpad - incj * j;
              uj = 0;
            }
            int endj = sj < sJ - uj + startj ? sj : sJ - uj + startj;
            for (int k = 0; k < s; k++)
              for (int iii = ui; iii < ui + endi - starti; iii++) {
                out[b * soi * soj * s + i * soj * s + j * s + k] +=
                    cblas_ddot(d * (endj - startj),
                               im + b * sI * sJ * d + iii * sJ * d + uj * d, 1,
                               kernel + k * si * sj * d +
                                   (starti + iii - ui) * sj * d + startj * d,
                               1);
              }
          }
        }
    }

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

    template <class A, class B, typename U, typename V>
    types::ndarray<typename A::dtype, types::pshape<long, long>>
    convolve2d(A const &inA, B const &inB, U mode, V boundary)
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

    template <class A, class B>
    types::ndarray<typename A::dtype, types::pshape<long, long, long, long>>
    convolve4d(A const &inA, B const &inB,
               types::ndarray<long, types::pshape<long>> const &increments)
    {
      auto shapeA = sutils::array(inA.shape());
      auto shapeB = sutils::array(inB.shape());

      long NA = shapeA[0];
      long NB = shapeB[0];
      long outX = shapeA[1] / increments[1] + (shapeA[1] % increments[1] != 0);
      long outY = shapeA[2] / increments[2] + (shapeA[2] % increments[2] != 0);
      // outX = shapeA[1];
      // outY = shapeA[2];
      auto shapeOut = types::pshape<long, long, long, long>(shapeA[0], outX,
                                                            outY, shapeB[0]);
      types::ndarray<typename A::dtype, types::pshape<long, long, long, long>>
          out = {shapeOut, (typename A::dtype)(0)};

      auto inA_ = numpy::functor::asarray{}(inA);
      auto inB_ = numpy::functor::asarray{}(inB);

      convol_loop4D(inA_.buffer, inB_.buffer, out.buffer, shapeA[0], shapeA[1],
                    shapeA[2], shapeA[3], shapeB[0], shapeB[1], shapeB[2],
                    increments[1], increments[2], outX, outY);
      // std::cout << "DONE\n";

      return out;
    }

    ///////////////////////////////////////////////////////////////////

    template <class A, class B>
    types::ndarray<typename A::dtype, types::pshape<long, long>>
    convolve2d(A const &inA, B const &inB)
    {
      return convolve2d(inA, inB, "same", "fill");
    }

    template <class A, class B>
    types::ndarray<typename A::dtype, types::pshape<long, long, long, long>>
    convolve2d(A const &inA, B const &inB,
               types::ndarray<long, types::pshape<long>> const &increments)
    {
      return convolve4d(inA, inB, increments);
    }

    //    template <class A, class B, typename U>
    //    types::ndarray<typename A::dtype, types::pshape<long, long>>
    //    convolve2d(A const &inA, B const &inB, U mode)
    //    {
    //      return convolve2d(inA, inB, mode, "fill");
    //    }
    //
    //    template <class A, class B, typename U, typename V, typename W>
    //    types::ndarray<typename A::dtype, types::pshape<long, long>>
    //    convolve2d(A const &inA, B const &inB, U mode, V boundary, W
    //    fillvalue)
    //    {
    //      return convolve2d(inA, inB, mode, boundary);
    //    }

    NUMPY_EXPR_TO_NDARRAY0_IMPL(convolve2d)
  }
}
PYTHONIC_NS_END

#endif
