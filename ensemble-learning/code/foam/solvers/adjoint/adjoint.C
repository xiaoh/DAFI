/*---------------------------------------------------------------------------*\

Application
    adjoint

Description
    Solves the adjoint equations with specified, fixed forcing functions,
    based on the cost function.
    Uses the SIMPLE algorithm.

Location: applications/solvers/incompressible/adjoint

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "simpleControl.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Steady-state adjoint solver for incompressible flows."
    );

    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createFields.H"
    #include "initContinuityErrs.H"


    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;
    while (simple.loop())
    {
      Info<< "Time = " << runTime.timeName() << nl << endl;
      // --- Pressure-velocity SIMPLE corrector
      {
          #include "adjointEqn.H"
      }
      sensitivity = -fvc::grad(Ua);
      runTime.write();
      runTime.printExecutionTime(Info);
    }
    Info<< "End\n" << endl;
    return(0);
}

// ************************************************************************* //
