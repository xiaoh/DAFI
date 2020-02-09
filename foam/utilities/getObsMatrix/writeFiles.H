
if(obsMatrix.good())
  {
    forAll(supportCellsIdx, i)
      {
	forAll(supportCellsIdx[i], j)
	  {
	    obsMatrix << i << "  " << supportCellsIdx[i][j] << "  " << invDistCoeffs[i][j] << endl;
	  }
      }
  }
 else
   {
     FatalErrorIn(args.executable())
       << "Cannot open file "
       << runTime.path()/"constant"/"obsMatrix"
       << exit(FatalError);
   };
