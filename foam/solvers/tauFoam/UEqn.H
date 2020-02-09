    // Momentum predictor

    tmp<fvVectorMatrix> UEqn
    (
        fvm::div(phi, U)
      + fvc::div(Tau-1.0/3.0*I*tr(Tau))
      + fvc::laplacian(nu*9,U)
      - fvm::laplacian(nu*10,U)
      ==
        fvOptions(U)
    );

    UEqn().relax();

    fvOptions.constrain(UEqn());

    solve(UEqn() == -fvc::grad(p));

    fvOptions.correct(U);
