    tmp<fvScalarMatrix> TEqn
    (
        fvm::div(phi, T)
      - fvm::laplacian(nut+nu,T)
    );

    #include "diagnosisT.H"
