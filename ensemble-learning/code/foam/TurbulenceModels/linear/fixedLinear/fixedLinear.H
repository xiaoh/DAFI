/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
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

Class
    Foam::RASModels::fixedLinear

Group
    grpRASTurbulence

Description
    Standard high Reynolds-number k-omega turbulence model for
    incompressible and compressible flows.

    References:
    \verbatim
        Wilcox, D. C. (1998).
        Turbulence modeling for CFD
        (Vol. 2, pp. 103-217). La Canada, CA: DCW industries.
    \endverbatim

    The default model coefficients are
    \verbatim
        fixedLinearCoeffs
        {
            Cmu         0.09;  // Equivalent to betaStar
            alpha       0.52;
            beta        0.072;
            alphak      0.5;
            alphaOmega  0.5;
        }
    \endverbatim

SourceFiles
    fixedLinear.C

\*---------------------------------------------------------------------------*/

#ifndef fixedLinear_H
#define fixedLinear_H

#include "RASModel.H"
#include "eddyViscosity.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

/*---------------------------------------------------------------------------*\
                           Class fixedLinear Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class fixedLinear
:
    public eddyViscosity<RASModel<BasicTurbulenceModel>>
{

protected:

        volScalarField nut_;
        volScalarField k_;
        volScalarField epsilon_;
        virtual void correctNut();


public:

    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;


    //- Runtime type information
    TypeName("fixedLinear");


    // Constructors

        //- Construct from components
        fixedLinear
        (
            const alphaField& alpha,
            const rhoField& rho,
            const volVectorField& U,
            const surfaceScalarField& alphaRhoPhi,
            const surfaceScalarField& phi,
            const transportModel& transport,
            const word& propertiesName = turbulenceModel::propertiesName,
            const word& type = typeName
        );


    //- Destructor
    virtual ~fixedLinear() = default;


    // Member Functions

        //- Read RASProperties dictionary
        virtual bool read();

        // virtual tmp<volScalarField> epsilon() const = 0;

        // virtual tmp<volScalarField> k() const = 0;

        // virtual tmp<volScalarField> omega() const = 0;

        // //- Return the turbulence kinetic energy
        virtual tmp<volScalarField> k() const
        {
            return k_;
        }

        virtual tmp<volScalarField> epsilon() const
        {
            return epsilon_;
        }

        // //- Return the turbulence specific dissipation rate
        // virtual tmp<volScalarField> omega() const
        // {
        //     return omega_;
        // }

        // //- Return the turbulence kinetic energy dissipation rate
        // virtual tmp<volScalarField> epsilon() const
        // {
        //     return tmp<volScalarField>
        //     (
        //         new volScalarField
        //         (
        //             IOobject
        //             (
        //                 "epsilon",
        //                 this->mesh_.time().timeName(),
        //                 this->mesh_
        //             ),
        //             Cmu_*k_*omega_,
        //             omega_.boundaryField().types()
        //         )
        //     );
        // }

        //- Solve the turbulence equations and correct the turbulence viscosity
        virtual void correct();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
#ifdef NoRepository
    #include "fixedLinear.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
