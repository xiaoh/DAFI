/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "kEpsilonNNQuadraticTrain.H"
#include "bound.H"
#include "wallFvPatch.H"
#include "nutkWallFunctionFvPatchScalarField.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace incompressible
{
namespace RASModels
{

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

defineTypeNameAndDebug(kEpsilonNNQuadraticTrain, 0);
addToRunTimeSelectionTable(RASModel, kEpsilonNNQuadraticTrain, dictionary);

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

void kEpsilonNNQuadraticTrain::correctNut()
{
    correctNonlinearStress(fvc::grad(U_));
}


void kEpsilonNNQuadraticTrain::correctNonlinearStress(const volTensorField& gradU)
{
    timeScale_=k_/epsilon_;

    // Linear (nut)
    nut_ = -g1_*k_*timeScale_;
    nut_.correctBoundaryConditions();
    
    // Quadratic (tau_NL)
    volSymmTensorField S(timeScale_*symm(gradU));
    volTensorField W(timeScale_*skew(gradU));

    nonlinearStress_ = 
        2*k_
       *(
           g2_ * twoSymm(S&W) 
         + g3_ * dev(innerSqr(S))
         + g4_ * dev(symm(W&W))
        );
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

kEpsilonNNQuadraticTrain::kEpsilonNNQuadraticTrain
(
    const geometricOneField& alpha,
    const geometricOneField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    nonlinearEddyViscosity<incompressible::RASModel>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),

    Ceps1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Ceps1",
            coeffDict_,
            1.44
        )
    ),

    Ceps2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Ceps2",
            coeffDict_,
            1.92
        )
    ),

    sigmak_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "sigmak",
            coeffDict_,
            1.0
        )
    ),

    sigmaEps_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "sigmaEps",
            coeffDict_,
            1.3
        )
    ),

    k_
    (
        IOobject
        (
            IOobject::groupName("k", alphaRhoPhi.group()),
            runTime_.timeName(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_
    ),

    epsilon_
    (
        IOobject
        (
            IOobject::groupName("epsilon", alphaRhoPhi.group()),
            runTime_.timeName(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_
    ),

    g1_
    (
        IOobject
        (
            "g1",
            runTime_.timeName(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_
    ),

    g2_
    (
        IOobject
        (
            "g2",
            runTime_.timeName(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_
    ),

    g3_
    (
        IOobject
        (
            "g3",
            runTime_.timeName(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_
    ),

    g4_
    (
        IOobject
        (
            "g4",
            runTime_.timeName(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_
    ),

    timeScale_
    (
        IOobject
        (
            "timeScale",
            runTime_.timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_,
        dimensionedScalar("timeScale", dimTime, scalar(0.0))
    )

{
    bound(k_, kMin_);
    bound(epsilon_, epsilonMin_);

    if (type == typeName)
    {
        printCoeffs(type);
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool kEpsilonNNQuadraticTrain::read()
{
    if (nonlinearEddyViscosity<incompressible::RASModel>::read())
    {
        Ceps1_.readIfPresent(coeffDict());
        Ceps2_.readIfPresent(coeffDict());
        sigmak_.readIfPresent(coeffDict());
        sigmaEps_.readIfPresent(coeffDict());

        return true;
    }

    return false;
}


void kEpsilonNNQuadraticTrain::correct()
{
    if (!turbulence_)
    {
        return;
    }

    nonlinearEddyViscosity<incompressible::RASModel>::correct();

    tmp<volTensorField> tgradU = fvc::grad(U_);
    const volTensorField& gradU = tgradU();

    volScalarField G
    (
        GName(),
        (nut_*twoSymm(gradU) - nonlinearStress_) && gradU
    );


    // Update epsilon and G at the wall
    epsilon_.boundaryFieldRef().updateCoeffs();

    // Dissipation equation
    tmp<fvScalarMatrix> epsEqn
    (
        fvm::ddt(epsilon_)
      + fvm::div(phi_, epsilon_)
      - fvm::laplacian(DepsilonEff(), epsilon_)
      ==
        Ceps1_*G*epsilon_/k_
      - fvm::Sp(Ceps2_*epsilon_/k_, epsilon_)
    );

    epsEqn.ref().relax();
    epsEqn.ref().boundaryManipulate(epsilon_.boundaryFieldRef());
    solve(epsEqn);
    bound(epsilon_, epsilonMin_);


    // Turbulent kinetic energy equation
    tmp<fvScalarMatrix> kEqn
    (
        fvm::ddt(k_)
      + fvm::div(phi_, k_)
      - fvm::laplacian(DkEff(), k_)
      ==
        G
      - fvm::Sp(epsilon_/k_, k_)
    );

    kEqn.ref().relax();
    solve(kEqn);
    bound(k_, kMin_);


    // Re-calculate viscosity and non-linear stress
    correctNonlinearStress(gradU);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace incompressible
} // End namespace Foam

// ************************************************************************* //
