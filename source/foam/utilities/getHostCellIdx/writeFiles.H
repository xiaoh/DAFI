if(indexHFile.good())
  {
    forAll(hostCellIndex, i)
      {
	    indexHFile << i << "  " << hostCellIndex[i] << endl;
      }
  }
 else
   {
     FatalErrorIn(args.executable())
       << "Cannot open file "
       << runTime.path()/"constant"/"indexHost.txt"
       << exit(FatalError);
   };
