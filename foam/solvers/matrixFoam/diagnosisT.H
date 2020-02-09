    // Initialization of the matrix
    label nCell = T.internalField().size();

    List<List<scalar> > A(nCell);
    
    for(int i = 0; i<nCell; i++)
    {
        A[i].setSize(nCell);
        for(int j = 0; j<nCell; j++)
        {
            A[i][j] = 0;
        }
    }
    
    // Print the matrix and the source term
    if (runTime.value() == runTime.endTime().value())
    {
        OFstream TMatrix("TMatrix");

        forAll(T.internalField(), iCell)
        {
            A[iCell][iCell] += TEqn().DD()()[iCell];
            // A[iCell][iCell] += UEqn().D()()[iCell];
            TMatrix<< iCell << " " 
                   << iCell << " " 
                   << A[iCell][iCell] << nl;
        }

        forAll(TEqn().lduAddr().upperAddr(), kOffDiag)
        {
            label iOff = TEqn().lduAddr().lowerAddr()[kOffDiag];
            label jOff = TEqn().lduAddr().upperAddr()[kOffDiag];
            // May need to transpose
            A[iOff][jOff] += TEqn().upper()[kOffDiag];
            A[jOff][iOff] += TEqn().lower()[kOffDiag];

            TMatrix<< iOff << " "
                   << jOff << " "
                   << A[iOff][jOff] << nl;
            TMatrix<< jOff << " "
                   << iOff << " "
                   << A[jOff][iOff] << nl;
        }
    }
