
Info<< "Reading/calculating face flux field phia\n" << endl;
surfaceScalarField phia
(
    IOobject
    (
        "phia",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    fvc::flux(Ua)
);
