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

#ifndef DTK_HYBRIDTRANSPORTPROBLEMGENERATOR_DEF_HPP
#define DTK_HYBRIDTRANSPORTPROBLEMGENERATOR_DEF_HPP

#include "DTK_Benchmark_DeterministicMesh.hpp"
#include "DTK_Benchmark_MonteCarloMesh.hpp"

#include <netcdf.h>

#include <Teuchos_ParameterXMLFileReader.hpp>

#include <Tpetra_Distributor.hpp>

#include <DTK_DBC.hpp>

namespace DataTransferKit
{
//---------------------------------------------------------------------------//
template <class Scalar, class SourceDevice, class TargetDevice>
HybridTransportProblemGenerator<Scalar, SourceDevice, TargetDevice>::
    HybridTransportProblemGenerator(
        const Teuchos::RCP<const Teuchos::Comm<int>> &comm,
        const std::string &input_file )
    : _comm( comm )
    , _input_file( input_file )
{ /* ... */
}

//---------------------------------------------------------------------------//
// Create a problem where all points are uniquely owned (i.e. no ghosting)
template <class Scalar, class SourceDevice, class TargetDevice>
void HybridTransportProblemGenerator<Scalar, SourceDevice, TargetDevice>::
    createUniquelyOwnedProblem(
        Kokkos::View<Coordinate **, Kokkos::LayoutLeft, SourceDevice>
            &src_coords,
        Kokkos::View<Scalar **, Kokkos::LayoutLeft, SourceDevice> &src_field,
        Kokkos::View<Coordinate **, Kokkos::LayoutLeft, TargetDevice>
            &tgt_coords,
        Kokkos::View<Scalar **, Kokkos::LayoutLeft, TargetDevice> &tgt_field )
{
    // The monte carlo mesh of the hybrid problem is overlapping in nature and
    // therefore we can't make a uniquely owned problem.
    throw DataTransferKitNotImplementedException();
}

//---------------------------------------------------------------------------//
// Create a general problem where points may exist on multiple
// processors. Points have a unique global id.
template <class Scalar, class SourceDevice, class TargetDevice>
void HybridTransportProblemGenerator<Scalar, SourceDevice, TargetDevice>::
    createGhostedProblem(
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
{
    // Read the input.
    Teuchos::ParameterXMLFileReader xml_reader( _input_file );
    auto parameters = xml_reader.getParameters();

    // Create the deterministic mesh. This is the source mesh.
    auto det_params = parameters.sublist( "Deterministic" );
    int det_ni = det_params.get<int>( "Num Cells I" );
    int det_nj = det_params.get<int>( "Num Cells J" );
    int det_nk = det_params.get<int>( "Num Cells K" );
    double det_dx = det_params.get<double>( "Delta X" );
    double det_dy = det_params.get<double>( "Delta Y" );
    double det_dz = det_params.get<double>( "Delta Z" );
    Benchmark::DeterministicMesh det_mesh( _comm, det_ni, det_nj, det_nk,
                                           det_dx, det_dy, det_dz );

    // Get the Monte Carlo parameters. This is the target mesh.
    auto mc_params = parameters.sublist( "Monte Carlo" );
    int num_sets = mc_params.get<int>( "Num Sets" );
    int mc_ni = mc_params.get<int>( "Num Cells I" );
    int mc_nj = mc_params.get<int>( "Num Cells J" );
    int mc_nk = mc_params.get<int>( "Num Cells K" );
    double mc_dx = mc_params.get<double>( "Delta X" );
    double mc_dy = mc_params.get<double>( "Delta Y" );
    double mc_dz = mc_params.get<double>( "Delta Z" );
    Teuchos::Array<double> mc_bnd_mesh_x =
        mc_params.get<Teuchos::Array<double>>( "Boundary Mesh X" );
    Teuchos::Array<double> mc_bnd_mesh_y =
        mc_params.get<Teuchos::Array<double>>( "Boundary Mesh Y" );
    Teuchos::Array<double> mc_bnd_mesh_z =
        mc_params.get<Teuchos::Array<double>>( "Boundary Mesh Z" );
    Benchmark::MonteCarloMesh mc_mesh(
        _comm, num_sets, mc_ni, mc_nj, mc_nk, mc_dx, mc_dy, mc_dz,
        mc_bnd_mesh_x.toVector(), mc_bnd_mesh_y.toVector(),
        mc_bnd_mesh_z.toVector() );

    // Extract the source mesh data. The source data is cell centered.
    auto num_src =
        det_mesh.cartesianMesh()->localCellCenterCoordinates().extent( 0 );
    Kokkos::resize( src_coords, num_src, 3 );
    Kokkos::deep_copy( src_coords,
                       det_mesh.cartesianMesh()->localCellCenterCoordinates() );
    Kokkos::resize( src_gids, num_src );
    Kokkos::deep_copy( src_gids,
                       det_mesh.cartesianMesh()->localCellGlobalIds() );

    // Extract the target mesh data. The target data is cell centered.
    auto num_tgt =
        mc_mesh.cartesianMesh()->localCellCenterCoordinates().extent( 0 );
    Kokkos::resize( tgt_coords, num_tgt, 3 );
    Kokkos::deep_copy( tgt_coords,
                       mc_mesh.cartesianMesh()->localCellCenterCoordinates() );
    Kokkos::resize( tgt_gids, num_tgt );
    Kokkos::deep_copy( tgt_gids,
                       mc_mesh.cartesianMesh()->localCellGlobalIds() );

    // Allocate the fields and initialize to zero.
    src_field = Kokkos::View<Scalar **, Kokkos::LayoutLeft, SourceDevice>(
        "src_field", num_src, 1 );
    Kokkos::deep_copy( src_field, 0.0 );
    tgt_field = Kokkos::View<Scalar **, Kokkos::LayoutLeft, TargetDevice>(
        "tgt_field", num_tgt, 1 );
    Kokkos::deep_copy( tgt_field, 0.0 );
}

//---------------------------------------------------------------------------//

} // namespace DataTransferKit

#endif // end  DTK_HYBRIDTRANSPORTPROBLEMGENERATOR_DEF_HPP
