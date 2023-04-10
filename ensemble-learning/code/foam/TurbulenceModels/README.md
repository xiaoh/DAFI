# Turbulence Models
Turbulence models in OpenFOAM are templated and therefor you cannot compile turbulence models individually. 
Instead, make a local copy of the entire TurbulenceModels directory and add the new turbulence models. 
OpenFOAM will use the user-defined turbulence model library over the installation one. 

Make a user copy of the TurbulenceModels directory and compile it.
```sh
# copy TurbulenceModels to user directory
cd $WM_PROJECT_DIR
cp -r --parents src/TurbulenceModels $WM_PROJECT_USER_DIR
# change the location of the compiled files in all Make/files
cd $WM_PROJECT_USER_DIR/src/TurbulenceModels
sed -i s/FOAM_LIBBIN/FOAM_USER_LIBBIN/g ./*/Make/files
# compile 
./Allwmake
```

Copy each linear turbulence model *myLinearModel* and include it in compile list.
```sh
# copy the new turbulence model
cp -r $DDTM/code/foam/TurbulenceModels/myLinearModel turbulenceModels/RAS
# manually add the new model to the compile list
# add the following lines to the file 'incompressible/turbulentTransportModels/turbulentTransportModels.C':
      #include "myLinearModel.H"
      makeRASModel(myLinearModel);
```
List of linear models: 
* __kOmegaNNLinear__: linear k-omega for prediction, calculates 'g1' using trained neural nework
* __kOmegaNNLinearTrain__: linear k-omega for training, reads fixed 'g1' file
* __fixedLinear__: provide a fixed eddy viscosity 'nut' file

Copy each non-linear turbulence model *myNonLinearModel* and include it in the compile list.
```sh
# copy the new turbulence models
cp -r $DDTM/code/foam/TurbulenceModels/myNonLinearModel incompressible/turbulentTransportModels/RAS
# manually add the new model to the compile list
# add the following line to the file 'incompressible/Make/files':
      turbulentTransportModels/RAS/myNonLinearModel/myNonLinearModel.C
```
List of non-linear models:
* __kEpsilonNNQuadraticTrain__: quadratic k-epsilon for training, reads fixed 'g1'-'g4' files
* __kEpsilonNNQuadratic__: NOT IMPLEMENTED
* __kOmegaNNQuadraticTrain__: quadratic k-omega for training, reads fixed 'g1'-'g4' files
* __kOmegaNNQuadratic__: NOT IMPLEMENTED


Finally, re-compile:
```sh
# compile again
wmakeLnInclude -u turbulenceModels
./Allwmake
```
