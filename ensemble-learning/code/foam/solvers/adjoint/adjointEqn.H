// solve Ua equation (Momentum predictor) * * * * * * * * * * * * * * * * * * //
volVectorField adjointTransposeConvection((fvc::grad(U) & Ua));

tmp<fvVectorMatrix> tUaEqn
(
    fvm::div(-phi,Ua)
  - adjointTransposeConvection*ATC
  - fvm::laplacian(nu, Ua)
  + UForcing
);
fvVectorMatrix& UaEqn = tUaEqn.ref();
UaEqn.relax();
solve(UaEqn == -fvc::grad(pa));


// solve pa equation * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
volScalarField rAU(1.0/UaEqn.A());
volVectorField HbyA(constrainHbyA(rAU*UaEqn.H(), Ua, pa));
surfaceScalarField phiHbyA("phiHbyA", fvc::flux(HbyA));
adjustPhi(phiHbyA, Ua, pa);
tmp<volScalarField> rAtU(rAU);
tUaEqn.clear();
constrainPressure(pa, Ua, phiHbyA, rAtU());
while (simple.correctNonOrthogonal())
{
    fvScalarMatrix paEqn
    (
        fvm::laplacian(rAtU(), pa) == fvc::div(phiHbyA) + pForcing
    );
    paEqn.setReference(paRefCell, paRefValue);
    paEqn.solve();
    if (simple.finalNonOrthogonalIter())
    {
        phia = phiHbyA - paEqn.flux();
    }
}
#include "adjointContinuityErrs.H"
pa.relax();


// update U (Momentum corrector) * * * * * * * * * * * * * * * * * * * * * * //
Ua = HbyA - rAtU()*fvc::grad(pa);
Ua.correctBoundaryConditions();
