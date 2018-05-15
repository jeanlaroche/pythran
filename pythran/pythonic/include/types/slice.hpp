#ifndef PYTHONIC_INCLUDE_TYPES_SLICE_HPP
#define PYTHONIC_INCLUDE_TYPES_SLICE_HPP

#include "pythonic/include/types/NoneType.hpp"

#include <limits>

PYTHONIC_NS_BEGIN

namespace types
{

  template <class T>
  class bound
  {
    T value_;

    // use a sentinel to store none<long>, it takes less space
    static constexpr long sentinel = std::numeric_limits<T>::min();

  public:
    bound() = default;
    bound(none_type) : value_(sentinel)
    {
    }
    bound(none<T> v) : value_(v.is_none ? sentinel : (T)v)
    {
    }
    bound(T v) : value_(v)
    {
    }

    operator T() const
    {
      return value_;
    }
    operator none<T>() const
    {
      if (value_ == sentinel)
        return none_type();
      else
        return value_;
    }
    bool is_none() const
    {
      return value_ == sentinel;
    }
  };

  struct slice;
  struct contiguous_slice;
  struct contiguous_normalized_slice;

  struct normalized_slice {
    long lower, upper, step;
    normalized_slice();
    normalized_slice(long lower, long upper, long step = 1);

    normalized_slice operator*(normalized_slice const &other) const;
    normalized_slice operator*(contiguous_normalized_slice const &other) const;
    normalized_slice operator*(slice const &other) const;
    normalized_slice operator*(contiguous_slice const &other) const;

    long size() const;
    inline long get(long i) const;
  };

  struct slice {
    using normalized_type = normalized_slice;

    bound<long> lower, upper;
    long step;
    slice(none<long> lower, none<long> upper, none<long> step = 1);
    slice();

    slice operator*(slice const &other) const;

    slice operator*(contiguous_slice const &other) const;

    /*
       Normalize change a[:-1] to a[:len(a)-1] to have positif index.
       It also check for value bigger than len(a) to fit the size of the
       container
       */
    normalized_slice normalize(long max_size) const;

    /*
     * An assert is raised when we can't compute the size without more
     * informations.
     */
    long size() const;

    long get(long i) const;
  };

  struct contiguous_normalized_slice {
    long lower, upper;
    static constexpr long step = 1;
    contiguous_normalized_slice();
    contiguous_normalized_slice(long lower, long upper);

    contiguous_normalized_slice
    operator*(contiguous_normalized_slice const &other) const;
    contiguous_normalized_slice operator*(contiguous_slice const &other) const;
    normalized_slice operator*(normalized_slice const &other) const;
    normalized_slice operator*(slice const &other) const;

    long size() const;

    inline long get(long i) const;
  };

  struct contiguous_slice {
    using normalized_type = contiguous_normalized_slice;
    long lower;
    bound<long> upper;
    static constexpr long step = 1;
    contiguous_slice(none<long> lower, none<long> upper);
    contiguous_slice();

    contiguous_slice operator*(contiguous_slice const &other) const;

    slice operator*(slice const &other) const;

    /*
       Normalize change a[:-1] to a[:len(a)-1] to have positif index.
       It also check for value bigger than len(a) to fit the size of the
       container
       */
    contiguous_normalized_slice normalize(long max_size) const;

    long size() const;

    inline long get(long i) const;
  };

  template <class T>
  struct normalized {
    using type = T;
  };

  template <>
  struct normalized<slice> {
    using type = normalized_slice;
  };

  template <>
  struct normalized<contiguous_slice> {
    using type = contiguous_normalized_slice;
  };

  template <class S>
  using normalize_t = typename normalized<S>::type;

  template <class S>
  S normalize(S s, long n)
  {
    if (s < 0)
      s += n;
    return s;
  }
  none_type normalize(none_type s, long n)
  {
    return {};
  }
  normalized_slice normalize(slice s, long n)
  {
    return s.normalize(n);
  }
  contiguous_normalized_slice normalize(contiguous_slice s, long n)
  {
    return s.normalize(n);
  }
}
PYTHONIC_NS_END

#endif
