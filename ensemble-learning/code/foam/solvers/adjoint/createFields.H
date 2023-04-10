
Info<< "Reading field pa\n" << endl;
volScalarField pa
(
    IOobject
    (
        "pa",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


Info<< "Reading field Ua\n" << endl;
volVectorField Ua
(
    IOobject
    (
        "Ua",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

#include "createPhia.H"

label paRefCell = 0;
scalar paRefValue = 0.0;
setRefCell(pa, simple.dict(), paRefCell, paRefValue);
mesh.setFluxRequired(pa.name());


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

#include "createPhi.H"


Info<< "Reading field UForcing\n" << endl;
volVectorField UForcing
(
    IOobject
    (
        "UForcing",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


Info<< "Reading field pForcing\n" << endl;
volScalarField pForcing
(
    IOobject
    (
        "pForcing",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


Info<< "Reading field sensitivity\n" << endl;
volTensorField sensitivity
(
    IOobject
    (
        "sensitivity",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


IOdictionary transportProperties
(
IOobject
(
"transportProperties",
runTime.constant(),
mesh,
IOobject::MUST_READ,
IOobject::NO_WRITE
)
);

dimensionedScalar nu
(
    "nu",
    transportProperties
);


scalar ATC
(
    mesh.solutionDict().subDict("adjoint").lookupOrDefault<scalar>("ATC", 1.0)
);
