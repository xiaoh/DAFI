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

#include "kOmegaQuadratic.H"
// #include "fvOptions.H"
#include "bound.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace incompressible
{
namespace RASModels
{

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

defineTypeNameAndDebug(kOmegaQuadratic, 0);
addToRunTimeSelectionTable(RASModel, kOmegaQuadratic, dictionary);

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

void kOmegaQuadratic::correctNut()
{
    correctNonlinearStress(fvc::grad(U_));
}

void kOmegaQuadratic::correctNonlinearStress(const volTensorField& gradU)
{
    timeScale_=1.0/(omega_*Cmu_);

    // Linear (nut)
    nut_ = -g1_*k_*timeScale_;
    nut_.correctBoundaryConditions();
    // fv::options::New(this->mesh_).correct(this->nut_);
    // BasicTurbulenceModel::correctNut(); 

    // Quadratic (tau_NL
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

kOmegaQuadratic::kOmegaQuadratic
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

    Cmu_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "betaStar",
            coeffDict_,
            0.09
        )
    ),

    beta_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "beta",
            coeffDict_,
            0.072
        )
    ),

    gamma_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "gamma",
            coeffDict_,
            0.52
        )
    ),

    alphaK_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaK",
            coeffDict_,
            0.5
        )
    ),

    alphaOmega_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaOmega",
            coeffDict_,
            0.5
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

    omega_
    (
        IOobject
        (
            IOobject::groupName("omega", alphaRhoPhi.group()),
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
    bound(omega_, omegaMin_);

    if (type == typeName)
    {
        printCoeffs(type);
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool kOmegaQuadratic::read()
{
    if (nonlinearEddyViscosity<incompressible::RASModel>::read())
    {
        Cmu_.readIfPresent(coeffDict());
        beta_.readIfPresent(coeffDict());
        gamma_.readIfPresent(coeffDict());
        alphaK_.readIfPresent(coeffDict());
        alphaOmega_.readIfPresent(coeffDict());

        return true;
    }

    return false;
}


void kOmegaQuadratic::correct()
{
    if (!turbulence_)
    {
        return;
    }

    // Local references
    const alphaField& alpha = alpha_;
    const rhoField& rho = rho_;
    const surfaceScalarField& alphaRhoPhi = alphaRhoPhi_;
    const volVectorField& U = U_;
    const volScalarField& nut = nut_;
    const volSymmTensorField& nonlinearStress = nonlinearStress_;
    // fv::options& fvOptions(fv::options::New(this->mesh_));

    nonlinearEddyViscosity<incompressible::RASModel>::correct();

    const volScalarField::Internal divU
    (
        fvc::div(fvc::absolute(phi(), U))().v()
    );

    tmp<volTensorField> tgradU = fvc::grad(U);
    const volTensorField& gradU = tgradU();
    
    volScalarField G
    (
        GName(),
        (nut*twoSymm(gradU) - nonlinearStress) && gradU
    );
    // tgradU.clear();

    // Update omega and G at the wall
    omega_.boundaryFieldRef().updateCoeffs();

     // Turbulence specific dissipation rate equation
    tmp<fvScalarMatrix> omegaEqn
    (
        fvm::ddt(alpha, rho, omega_)
      + fvm::div(alphaRhoPhi, omega_)
      - fvm::laplacian(alpha*rho*DomegaEff(), omega_)
     ==
        gamma_*alpha()*rho()*G*omega_/k_
      - fvm::SuSp(((2.0/3.0)*gamma_)*alpha()*rho()*divU, omega_)
      - fvm::Sp(beta_*alpha()*rho()*omega_(), omega_)
    //   + fvOptions(alpha, rho, omega_)
    );

    omegaEqn.ref().relax();
    // fvOptions.constrain(omegaEqn.ref());
    omegaEqn.ref().boundaryManipulate(omega_.boundaryFieldRef());
    solve(omegaEqn);
    // fvOptions.correct(omega_);
    bound(omega_, omegaMin_);


    // Turbulent kinetic energy equation
    tmp<fvScalarMatrix> kEqn
    (
        fvm::ddt(alpha, rho, k_)
      + fvm::div(alphaRhoPhi, k_)
      - fvm::laplacian(alpha*rho*DkEff(), k_)
     ==
        alpha()*rho()*G
      - fvm::SuSp((2.0/3.0)*alpha()*rho()*divU, k_)
      - fvm::Sp(Cmu_*alpha()*rho()*omega_(), k_)
    //   + fvOptions(alpha, rho, k_)
    );

    kEqn.ref().relax();
    // fvOptions.constrain(kEqn.ref());
    solve(kEqn);
    // fvOptions.correct(k_);
    bound(k_, kMin_);

    correctNonlinearStress(gradU);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace incompressible
} // End namespace Foam

// ************************************************************************* //}
