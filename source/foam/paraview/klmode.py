
from paraview.simple import *
from os import sep

# inputs
vars = ['K', 'Eta', 'Xi']
Nmodes = 20
bar_flag = 1
bar_fixed_flag = 1
bar_range = [-0.5, 0.5]
save_dir = '/home/grad3/cmich/Desktop/pvfigs'
initialField = 'p'

# run
# field names
Nvars = len(vars)
fieldnames = []
for j in range(Nvars):
    for k in range(Nmodes):
        fieldnames += ['KLMode_' + vars[j] + '_' + str(k+1)]
Nfields = len(fieldnames)

# view setting
paraview.simple._DisableFirstRenderCameraReset()
casefoam = GetActiveSource()
renderView1 = GetActiveViewOrCreate('RenderView')
casefoamDisplay = GetDisplayProperties(casefoam, view=renderView1)
cameraPosition = renderView1.CameraPosition

pLUT = GetColorTransferFunction(initialField)
barorg = GetScalarBar(pLUT, renderView1)
orient = barorg.Orientation
pos = barorg.Position
barlength = barorg.ScalarBarLength

# create the images
for field in fieldnames:
    renderView1 = GetActiveViewOrCreate('RenderView')
    ColorBy(casefoamDisplay, ('POINTS', field))
    HideScalarBarIfNotNeeded(pLUT, renderView1)
    pLUT = GetColorTransferFunction(field)
    PWF = GetOpacityTransferFunction(field)
    if bar_fixed_flag:
        pLUT.RescaleTransferFunction(bar_range[0], bar_range[1])
        PWF.RescaleTransferFunction(bar_range[0], bar_range[1])
    else:
        casefoamDisplay.RescaleTransferFunctionToDataRange(True, True)
    casefoamDisplay.SetScalarBarVisibility(renderView1, bar_flag)
    bar = GetScalarBar(pLUT, renderView1)
    bar.ScalarBarLength = barlength
    bar.Orientation = orient
    bar.Position = pos
    SaveScreenshot(save_dir + sep + field + '.png',
                   magnification=1, quality=100, view=renderView1)
