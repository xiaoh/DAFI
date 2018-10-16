if(indexHFile.good())
  {
    forAll(supportCellsIdx, i)
      {
	forAll(supportCellsIdx[i], j)
	  {
	    indexHFile << i << "  " << supportCellsIdx[i][j] << endl;
	  }
      }
  }
 else
   {
     FatalErrorIn(args.executable())
       << "Cannot open file "
       << runTime.path()/"constant"/"indexH.txt"
       << exit(FatalError);
   };


if(weightHFile.good())
  {
    forAll(invDistCoeffs, i)
      {
	forAll(invDistCoeffs[i], j)
	  {
	    weightHFile << invDistCoeffs[i][j] << endl;
	  }
      }
  }
 else
   {
     FatalErrorIn(args.executable())
       << "Cannot open file "
       << runTime.path()/"constant"/"weightH.txt"
       << exit(FatalError);
   };

