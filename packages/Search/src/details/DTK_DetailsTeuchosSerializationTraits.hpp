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
#ifndef DTK_DETAILS_TEUCHOS_SERIALIZATION_TRAITS_HPP
#define DTK_DETAILS_TEUCHOS_SERIALIZATION_TRAITS_HPP

#include <DTK_Box.hpp>
#include <DTK_Point.hpp>
#include <DTK_Predicates.hpp>
#include <DTK_Sphere.hpp>

#include <Teuchos_SerializationTraits.hpp>

namespace Teuchos
{

template <typename Ordinal>
class SerializationTraits<Ordinal, DataTransferKit::Point>
    : public DirectSerializationTraits<Ordinal, DataTransferKit::Point>
{
};

template <typename Ordinal>
class SerializationTraits<Ordinal, DataTransferKit::Box>
    : public DirectSerializationTraits<Ordinal, DataTransferKit::Box>
{
};

template <typename Ordinal>
class SerializationTraits<Ordinal, DataTransferKit::Sphere>
    : public DirectSerializationTraits<Ordinal, DataTransferKit::Sphere>
{
};

template <typename Ordinal, typename Geometry>
class SerializationTraits<Ordinal, DataTransferKit::Nearest<Geometry>>
    : public DirectSerializationTraits<Ordinal,
                                       DataTransferKit::Nearest<Geometry>>
{
};

template <typename Ordinal, typename Geometry>
class SerializationTraits<Ordinal, DataTransferKit::Intersects<Geometry>>
    : public DirectSerializationTraits<Ordinal,
                                       DataTransferKit::Intersects<Geometry>>
{
};

} // namespace Teuchos

#endif
