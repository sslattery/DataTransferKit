/****************************************************************************
 * Copyright (c) 2012-2018 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef DTK_MULTIVARIATE_POLYNOMIAL_BASIS_HPP
#define DTK_MULTIVARIATE_POLYNOMIAL_BASIS_HPP

// FIXME: KOKKOS_INLINE_FUNCTION is undefined in Kokkos_Array.hpp, had to
// separate out the two includes below so that Kokkos_Macros.hpp is included
// 1st.  Headers can be regrouped after
// [kokkos/kokkos#1579](https://github.com/kokkos/kokkos/pull/1579) makes it
// into Trilinos.
#include <Kokkos_Macros.hpp>

#include <Kokkos_Array.hpp>

namespace DataTransferKit
{

struct Constant
{
};

struct Linear
{
};

struct Quadratic
{
};

namespace Details
{

template <typename Basis, int DIM>
struct Traits
{
    static KOKKOS_INLINE_FUNCTION int constexpr size();
};

template <int DIM>
struct Traits<Constant, DIM>
{
    static KOKKOS_INLINE_FUNCTION int constexpr size() { return 1; }
};

template <>
struct Traits<Linear, 3>
{
    static KOKKOS_INLINE_FUNCTION int constexpr size() { return 4; }
};

template <>
struct Traits<Quadratic, 3>
{
    static KOKKOS_INLINE_FUNCTION int constexpr size() { return 10; }
};

template <>
struct Traits<Linear, 2>
{
    static KOKKOS_INLINE_FUNCTION int constexpr size() { return 3; }
};

template <>
struct Traits<Quadratic, 2>
{
    static KOKKOS_INLINE_FUNCTION int constexpr size() { return 6; }
};

} // namespace Details

template <typename Basis, int DIM>
struct MultivariatePolynomialBasis
{
    static KOKKOS_INLINE_FUNCTION int constexpr size()
    {
        return Details::Traits<Basis, DIM>::size();
    }
    template <typename Point>
    KOKKOS_INLINE_FUNCTION Kokkos::Array<double, size()>
    operator()( Point const &p ) const;
};

// NOTE: For now relying on Point::operator[]( int i ) to access the coordinates
// which make it possible to use various types such as DTK::Point or
// Kokkos::Array<double, DIM>

template <>
template <typename Point>
KOKKOS_INLINE_FUNCTION
    Kokkos::Array<double, MultivariatePolynomialBasis<Constant, 3>::size()>
    MultivariatePolynomialBasis<Constant, 3>::operator()( Point const & ) const
{
    return {{1.}};
}

template <>
template <typename Point>
KOKKOS_INLINE_FUNCTION
    Kokkos::Array<double, MultivariatePolynomialBasis<Linear, 3>::size()>
    MultivariatePolynomialBasis<Linear, 3>::operator()( Point const &p ) const
{
    return {{1., p[0], p[1], p[2]}};
}

template <>
template <typename Point>
KOKKOS_INLINE_FUNCTION
    Kokkos::Array<double, MultivariatePolynomialBasis<Quadratic, 3>::size()>
    MultivariatePolynomialBasis<Quadratic, 3>::
    operator()( Point const &p ) const
{
    return {{1., p[0], p[1], p[2], p[0] * p[0], p[0] * p[1], p[0] * p[2],
             p[1] * p[1], p[1] * p[2], p[2] * p[2]}};
}

template <>
template <typename Point>
KOKKOS_INLINE_FUNCTION
    Kokkos::Array<double, MultivariatePolynomialBasis<Constant, 2>::size()>
    MultivariatePolynomialBasis<Constant, 2>::operator()( Point const & ) const
{
    return {{1.}};
}

template <>
template <typename Point>
KOKKOS_INLINE_FUNCTION
    Kokkos::Array<double, MultivariatePolynomialBasis<Linear, 2>::size()>
    MultivariatePolynomialBasis<Linear, 2>::operator()( Point const &p ) const
{
    return {{1., p[0], p[1]}};
}

template <>
template <typename Point>
KOKKOS_INLINE_FUNCTION
    Kokkos::Array<double, MultivariatePolynomialBasis<Quadratic, 2>::size()>
    MultivariatePolynomialBasis<Quadratic, 2>::
    operator()( Point const &p ) const
{
    return {{1., p[0], p[1], p[0] * p[0], p[0] * p[1], p[1] * p[1]}};
}

} // namespace DataTransferKit

#endif
