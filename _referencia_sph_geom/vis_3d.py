from paraview.simple import *
import numpy as np

def sphere_stl(inputs, out):
    print(inputs, out)
    # create a new 'XML Unstructured Grid Reader'
    in_vtu = XMLUnstructuredGridReader(FileName=[inputs[1]])

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    # uncomment following to set a specific view size
    renderView1.ViewSize = [720, 480]

    # get layout
    layout1 = GetLayout()

    # show data in view
    case_3d_bounded_sphere_sphere_01869vtuDisplay = Show(in_vtu, renderView1, 'UnstructuredGridRepresentation')

    # get color transfer function/color map for 'pz'
    pzLUT = GetColorTransferFunction('pz')

    # get opacity transfer function/opacity map for 'pz'
    pzPWF = GetOpacityTransferFunction('pz')

    case_3d_bounded_sphere_sphere_01869vtuDisplay.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # create a new 'Glyph'
    glyph1 = Glyph(Input=in_vtu,
        GlyphType='Arrow')
    glyph1.GlyphType = 'Sphere'
    glyph1.ScaleArray = ['POINTS', 'No scale array']
    glyph1.GlyphMode = 'All Points'
    glyph1.ScaleFactor = 0.05

    # show data in view
    glyph1Display = Show(glyph1, renderView1, 'GeometryRepresentation')

    # show color bar/color legend
    glyph1Display.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # set scalar coloring
    ColorBy(glyph1Display, ('POINTS', 'rho'))

    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(pzLUT, renderView1)

    # rescale color and/or opacity maps used to include current data range
    glyph1Display.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    glyph1Display.SetScalarBarVisibility(renderView1, True)

    # get color transfer function/color map for 'rho'
    rhoLUT = GetColorTransferFunction('rho')
    rhoLUT.RescaleTransferFunction(0.95, 1.05)

    # get opacity transfer function/opacity map for 'rho'
    rhoPWF = GetOpacityTransferFunction('rho')
    rhoPWF.RescaleTransferFunction(0.95, 1.05)

    # hide data in view
    Hide(in_vtu, renderView1)

    # get color legend/bar for rhoLUT in view renderView1

    # Properties modified on rhoLUTColorBar
    rhoLUT.ApplyPreset('Viridis (matplotlib)', True)
    rhoLUTColorBar = GetScalarBar(rhoLUT, renderView1)
    rhoLUTColorBar.Title = r'$\rho$' 
    rhoLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    rhoLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    rhoLUTColorBar.Orientation = 'Vertical'
    rhoLUTColorBar.ScalarBarLength = 0.5
    rhoLUTColorBar.WindowLocation = 'LowerRightCorner'
    rhoLUTColorBar.AddRangeLabels = 0

    # create a new 'STL Reader'
    in_stl = STLReader(FileNames=[inputs[0]])

    # show data in view
    spherestlDisplay = Show(in_stl, renderView1, 'GeometryRepresentation')

    # get color transfer function/color map for 'STLSolidLabeling'
    sTLSolidLabelingLUT = GetColorTransferFunction('STLSolidLabeling')

    # show color bar/color legend
    spherestlDisplay.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # get opacity transfer function/opacity map for 'STLSolidLabeling'
    sTLSolidLabelingPWF = GetOpacityTransferFunction('STLSolidLabeling')

    # create a new 'Transform'
    transform1 = Transform(Input=in_stl)
    transform1.Transform = 'Transform'

    # Properties modified on transform1.Transform
    transform1.Transform.Scale = [0.95, 0.9, 0.925]

    # show data in view
    transform1Display = Show(transform1, renderView1, 'GeometryRepresentation')

    # hide data in view
    Hide(in_stl, renderView1)

    # show color bar/color legend
    transform1Display.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Background = [1.0, 1.0, 1.0]
    renderView1.Update()

    # toggle 3D widget visibility (only when running from the GUI)
    Hide3DWidgets(proxy=transform1.Transform)

    # turn off scalar coloring
    ColorBy(transform1Display, None)

    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(sTLSolidLabelingLUT, renderView1)

    # set active source
    SetActiveSource(glyph1)

    renderView1.OrientationAxesVisibility = 0

    # current camera placement for renderView1
    renderView1.CameraPosition = [-0.0001929394680598162, -0.00019338270543811653, 4.329103810074073]
    renderView1.CameraFocalPoint = [-0.0001929394680598162, -0.00019338270543811653, 0.0001589839541406013]
    renderView1.CameraParallelScale = 1.6403972094502917

    #### uncomment the following to render all views
    # RenderAllViews()
    # alternatively, if you want to write images, you can use SaveScreenshot(...).
    SaveScreenshot(out, magnification=1, quality=100, view=renderView1)


def bunny_stl(inputs, out, vu=0):
    print(inputs, out)
    # create a new 'XML Unstructured Grid Reader'
    in_vtu = XMLUnstructuredGridReader(FileName=[inputs[1]])

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    # uncomment following to set a specific view size
    renderView1.ViewSize = [720, 480]

    # get layout
    layout1 = GetLayout()

    # show data in view
    case_3d_bounded_sphere_sphere_01869vtuDisplay = Show(in_vtu, renderView1, 'UnstructuredGridRepresentation')

    # get color transfer function/color map for 'pz'
    pzLUT = GetColorTransferFunction('pz')

    # get opacity transfer function/opacity map for 'pz'
    pzPWF = GetOpacityTransferFunction('pz')

    case_3d_bounded_sphere_sphere_01869vtuDisplay.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # create a new 'Glyph'
    glyph1 = Glyph(Input=in_vtu,
        GlyphType='Arrow')
    glyph1.GlyphType = 'Sphere'
    glyph1.ScaleArray = ['POINTS', 'No scale array']
    glyph1.GlyphMode = 'All Points'
    glyph1.ScaleFactor = 0.01

    # show data in view
    glyph1Display = Show(glyph1, renderView1, 'GeometryRepresentation')

    # show color bar/color legend
    glyph1Display.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # set scalar coloring
    ColorBy(glyph1Display, ('POINTS', 'rho'))

    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(pzLUT, renderView1)

    # rescale color and/or opacity maps used to include current data range
    glyph1Display.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    glyph1Display.SetScalarBarVisibility(renderView1, True)

    # get color transfer function/color map for 'rho'
    rhoLUT = GetColorTransferFunction('rho')

    # get opacity transfer function/opacity map for 'rho'
    rhoPWF = GetOpacityTransferFunction('rho')

    # hide data in view
    Hide(in_vtu, renderView1)

    # get color legend/bar for rhoLUT in view renderView1

    # Properties modified on rhoLUTColorBar
    rhoLUT.ApplyPreset('Viridis (matplotlib)', True)
    rhoLUTColorBar = GetScalarBar(rhoLUT, renderView1)
    rhoLUTColorBar.Title = r'$\rho$' 
    rhoLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    rhoLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    rhoLUTColorBar.Orientation = 'Vertical'
    rhoLUTColorBar.ScalarBarLength = 0.5
    rhoLUTColorBar.WindowLocation = 'LowerRightCorner'
    rhoLUTColorBar.AddRangeLabels = 0

    # create a new 'STL Reader'
    in_stl = STLReader(FileNames=[inputs[0]])

    # show data in view
    spherestlDisplay = Show(in_stl, renderView1, 'GeometryRepresentation')

    # get color transfer function/color map for 'STLSolidLabeling'
    sTLSolidLabelingLUT = GetColorTransferFunction('STLSolidLabeling')

    # show color bar/color legend
    spherestlDisplay.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # get opacity transfer function/opacity map for 'STLSolidLabeling'
    sTLSolidLabelingPWF = GetOpacityTransferFunction('STLSolidLabeling')

    # create a new 'Transform'
    transform1 = Transform(Input=in_stl)
    transform1.Transform = 'Transform'

    # Properties modified on transform1.Transform
    # transform1.Transform.Scale = [0.95, 0.95, 0.95]
    transform1.Transform.Scale = [10.0, 10.0, 10.0]
    transform1Display = Show(transform1, renderView1)
    # current camera placement for renderView1
    bound = transform1.GetDataInformation().GetBounds()
    centerx = -0.5*(bound[0] + bound[1])
    centery = -0.5*(bound[2] + bound[3])
    centerz = -0.5*(bound[4] + bound[5])


    transform2 = Transform(Input=transform1)
    transform2.Transform = 'Transform'
    transform2.Transform.Translate = [centerx, centery, centerz]
    transform1Display = Show(transform2, renderView1, 'GeometryRepresentation')

    # hide data in view
    Hide(in_stl, renderView1)
    Hide(transform1, renderView1)

    # show color bar/color legend
    transform1Display.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Background = [1.0, 1.0, 1.0]
    renderView1.Update()

    # toggle 3D widget visibility (only when running from the GUI)
    Hide3DWidgets(proxy=transform1.Transform)

    # turn off scalar coloring
    ColorBy(transform1Display, None)

    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(sTLSolidLabelingLUT, renderView1)

    # set active source
    SetActiveSource(glyph1)

    renderView1.OrientationAxesVisibility = 0

    # current camera placement for renderView1
    if vu==0:
        renderView1.CameraPosition = [0.5504952015387378, -1.7346002295849994, -0.08798732251565834]
    elif vu==1:
        renderView1.CameraPosition = [0, 1.2883370468004447, 1.2883370468004447]
    elif vu==2:
        renderView1.CameraPosition = [-1.4883370468004447, -1.1883370468004447, 0]
    renderView1.CameraFocalPoint = [3.067755905986069e-05, 9.778040686134193e-05, -0.0006076850864953354]
    renderView1.CameraViewUp = [0.05116432263671329, -0.03404070299386218, 0.998109935141821]
    renderView1.CameraParallelScale = 0.6904376628640279 

    #### uncomment the following to render all views
    # RenderAllViews()
    # alternatively, if you want to write images, you can use SaveScreenshot(...).
    SaveScreenshot(out, magnification=1, quality=100, view=renderView1)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="script to convert stl and point data to png")
    parser.add_argument('--infiles')
    parser.add_argument(
        '--in-files', nargs='*', action="store",
        dest="in_files", type=str,
        help="All input files"
    )
    parser.add_argument(
        '-o', '--outfile', action="store",
        dest="outfile", type=str,
        help="output file name"
    )
    parser.add_argument(
        '-m', '--mode', action="store",
        dest="mode", type=str,
        help="The mode of operation"
    )
    args = parser.parse_args()

    filenames = args.in_files
    outfile = args.outfile
    if args.mode == 'stl_with_body':
        sphere_stl(filenames, outfile)
    elif args.mode == 'stl_with_bunny':
        bunny_stl(filenames, outfile)
    elif args.mode == 'stl_with_bunny1':
        bunny_stl(filenames, outfile, vu=1)
    elif args.mode == 'stl_with_bunny2':
        bunny_stl(filenames, outfile, vu=2)