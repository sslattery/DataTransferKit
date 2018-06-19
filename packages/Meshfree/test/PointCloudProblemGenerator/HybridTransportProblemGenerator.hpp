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

#ifndef DTK_HYBRIDTRANSPORTPROBLEMGENERATOR_HPP
#define DTK_HYBRIDTRANSPORTPROBLEMGENERATOR_HPP

#include "DTK_ConfigDefs.hpp"
#include "DTK_Types.h"
#include "PointCloudProblemGenerator.hpp"

#include <Kokkos_View.hpp>

#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>

#include <functional>
#include <string>

namespace DataTransferKit
{
//---------------------------------------------------------------------------//
// Generate point cloud problem from a hybrid transport benchmark.
template <class Scalar, class SourceDevice, class TargetDevice>
class HybridTransportProblemGenerator
    : public PointCloudProblemGenerator<Scalar, SourceDevice, TargetDevice>
{
  public:
    // Constructor.
    HybridTransportProblemGenerator(
        const Teuchos::RCP<const Teuchos::Comm<int>> &comm,
        const std::string &input_file );

    // Create a problem where all points are uniquely owned (i.e. no
    // ghosting). Both source and target fields have one component and are
    // initialized to zero.
    //
    // This is not implemented for this particular problem generator as all
    // hybrid transport problems have some natural element of overlap due to
    // the nature of the monte carlo decomposition.
    void createUniquelyOwnedProblem(
        Kokkos::View<Coordinate **, Kokkos::LayoutLeft, SourceDevice>
            &src_coords,
        Kokkos::View<Scalar **, Kokkos::LayoutLeft, SourceDevice> &src_field,
        Kokkos::View<Coordinate **, Kokkos::LayoutLeft, TargetDevice>
            &tgt_coords,
        Kokkos::View<Scalar **, Kokkos::LayoutLeft, TargetDevice> &tgt_field )
        override;

    // Create a general problem where points may exist on multiple
    // processors. Both source and target fields have 1 component and are
    // initialized to zero.
    void createGhostedProblem(
        Kokkos::View<Coordinate **, Kokkos::LayoutLeft, SourceDevice>
            &src_coords,
        Kokkos::View<GlobalOrdinal *, Kokkos::LayoutLeft, SourceDevice>
            &src_gids,
        Kokkos::View<Scalar **, Kokkos::LayoutLeft, SourceDevice> &src_field,
        Kokkos::View<Coordinate **, Kokkos::LayoutLeft, TargetDevice>
            &tgt_coords,
        Kokkos::View<GlobalOrdinal *, Kokkos::LayoutLeft, TargetDevice>
            &tgt_gids,
        Kokkos::View<Scalar **, Kokkos::LayoutLeft, TargetDevice> &tgt_field )
        override;

  private:
    // Comm
    Teuchos::RCP<const Teuchos::Comm<int>> _comm;

    // Input file.
    std::string _input_file;
};

//---------------------------------------------------------------------------//

} // namespace DataTransferKit

//---------------------------------------------------------------------------//
// Template includes
//---------------------------------------------------------------------------//

#include "HybridTransportProblemGenerator_def.hpp"

//---------------------------------------------------------------------------//

#endif // end  DTK_HYBRIDTRANSPORTPROBLEMGENERATOR_HPP
