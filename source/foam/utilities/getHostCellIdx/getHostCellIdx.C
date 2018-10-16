/*---------------------------------------------------------------------------* \
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 1991-2009 OpenCFD Ltd.
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software; you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation; either version 2 of the License, or (at your
    option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM; if not, write to the Free Software Foundation,
    Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA

Application
    getObsMatrix

Description
    Compute observation matrix H from a mesh and a list of given points.


\*---------------------------------------------------------------------------*/


#include "fvCFD.H"
#include "vectorIOList.H"
#include "scalarList.H"
#include "OFstream.H"
#include "boundBox.H"

int main(int argc, char *argv[])
{
#   include "setRootCase.H"

#   include "createTime.H"
#   include "createMesh.H"
#   include "createFiles.H"
  
  // vectorIOList pts(0.5, 0.5, 1);  // observation points    

  vectorIOList obsPts(
		   IOobject
		   (
		    "obsLocations",
		    runTime.constant(),
		    mesh,
		    IOobject::MUST_READ,
		    IOobject::AUTO_WRITE
		    )
		   );

  labelList hostCellIndex(obsPts.size(), label(-1));


  Info << "# of obs pts: " << obsPts.size() << endl;

  
  forAll(obsPts, pointI)
    {
      vector pt = obsPts[pointI];
      Info << "finding host for point: " << pt << endl;

      label hostCellI = mesh.findCell(pt);
      
      // bound box of host mesh
      const boundBox & meshBB = mesh.bounds();

      if(hostCellI >= 0)
      {
          hostCellIndex[pointI] = hostCellI;

          // debug
          Info << "hostCellI = " << hostCellI << endl;
      }
      else if(meshBB.contains(pt))
          // at least the point is  in the bound box of host host mesh
      {
          label nearCellI = mesh.findNearestCell(pt);
          hostCellIndex[pointI] = nearCellI;

          // JX TODO later: please print the cell center location as well for diagnosis purposes
          WarningIn("getHostCellIdx")
              << "Point outside any cell but in bound box. Nearest cell found instead!" << endl
                  << "Point: " << pt << endl
                  <<" Nearst cell ID: " << nearCellI << endl
                  <<"Cell center: "
                  << endl;

          // debug
          Info << "nearCellI = " << nearCellI << endl;
      }
      else
      {
          WarningIn("getHostCellIdx")
              << "Point outside host mesh domain!" << endl
                  << abort(FatalError);
      }
      



    }

#include  "writeFiles.H"

  return(0);
}



// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //



// ************************************************************************* //
