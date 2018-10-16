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

int main(int argc, char *argv[])
{
#   include "setRootCase.H"

#   include "createTime.H"
#   include "createMesh.H"
#   include "createFiles.H"
  
  // vectorIOList pts(0.5, 0.5, 1);  // observation points    

  const scalar directHitTol = 1e-6;

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

  
  scalarListList* inverseDistanceWeightsPtr = new scalarListList(obsPts.size());
  scalarListList& invDistCoeffs = *inverseDistanceWeightsPtr;

  labelListList* supportCellIndexPtr = new labelListList(obsPts.size());
  labelListList& supportCellsIdx = *supportCellIndexPtr;


  Info << "# of obs pts: " << obsPts.size() << endl;
  const labelListList& cc = mesh.cellCells();
  
  forAll(obsPts, pointI)
    {
      vector pt = obsPts[pointI];
      Info << "calculating weights for point: " << pt << endl;

      label hostCellI = mesh.findCell(pt);
      const labelList& neighbours = cc[hostCellI];
      const vectorField& centre = mesh.C().internalField();

      scalar m = mag(pt - centre[hostCellI]);

      // Debug

      Info << "hostCellI = " << hostCellI << endl;
      Info << "m = " << m << endl;

      if
	(
	 m < directHitTol                            // Direct hit
	 || neighbours.empty()
	 )
	{
	  invDistCoeffs[pointI].setSize(1);
	  invDistCoeffs[pointI][0] = 1.0;

	  supportCellsIdx[pointI].setSize(1);
	  supportCellsIdx[pointI][0] = hostCellI;

	  Info << "direct hit!\n" << endl;
	}
      else
	{
	  Info << "nbs: " << neighbours << endl;
	  invDistCoeffs[pointI].setSize(neighbours.size() + 1);

	  supportCellsIdx[pointI].setSize(neighbours.size() + 1);;
	  supportCellsIdx[pointI][0] = hostCellI;
	  
	  // The first coefficient corresponds to the centre cell.
	  // The rest is ordered in the same way as the cellCells list.
	  scalar invDist = 1.0/m;
	  invDistCoeffs[pointI][0] = invDist;
	  scalar sumInvDist = invDist;

	  // now add the neighbours
	  forAll(neighbours, ni)
	    {
	      invDist = 1.0/mag(pt - centre[neighbours[ni]]);
	      invDistCoeffs[pointI][ni + 1] = invDist;
	      supportCellsIdx[pointI][ni + 1] = neighbours[ni];
	      sumInvDist += invDist;
	    }

	  // divide by the total inverse-distance
	  forAll(invDistCoeffs[pointI], i)
	    {
	      invDistCoeffs[pointI][i] /= sumInvDist;
	    }

	  Info << "weight is: " <<  invDistCoeffs[pointI] << "\n"  << endl;
	}
    }

  Info << "computed weights: " << invDistCoeffs << endl;
  Info << "corresponding cell indices: " << supportCellsIdx << endl;

#include  "writeFiles.H"

  return(0);
}



// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //



// ************************************************************************* //
