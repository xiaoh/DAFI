// Momentum predictor

nut = Foam::max(nut, nutMin);
tmp<fvVectorMatrix> tUEqn
(
    fvm::div(phi, U)
  - fvm::laplacian(nut,U)
  - fvm::laplacian(nu,U)
  - fvc::div(nut*dev(T(fvc::grad(U))))
  - fvc::div(nu*dev(T(fvc::grad(U))))
  ==
    fvOptions(U)
);

fvVectorMatrix& UEqn = tUEqn.ref();
UEqn.relax();

fvOptions.constrain(UEqn);

solve(UEqn == -fvc::grad(p));

fvOptions.correct(U);
