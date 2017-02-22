"""********************************************************************************
                             mayavi_plot3d.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the 
terms of the GNU General Public License version 3 as published by the Free Software 
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
********************************************************************************"""
import os, sys, math, time, numpy
from PyQt4 import QtCore, QtGui
from daetools.pyDAE import *

python_major = sys.version_info[0]
from .choose_variable import daeChooseVariable, daeTableDialog

try:
    import enthought.mayavi as mayavi
    from enthought.mayavi import mlab
    from enthought.mayavi.sources.vtk_file_reader import VTKFileReader
except ImportError:
    import mayavi
    from mayavi import mlab
    from mayavi.sources.vtk_file_reader import VTKFileReader
    
@mlab.animate(delay=50, ui=True)
def animateVTKFiles(figure, vtkSource, vtkFiles):
    for vtk_file in vtkFiles[1:-1]:
        vtkSource.initialize(vtk_file)                
        figure.scene.render()
        yield

class daeMayavi3DPlot:
    def __init__(self, tcpipServer):
        self.tcpipServer = tcpipServer
        self._cv_dlg     = None
                
    def newSurface(self):
        processes = [dataReceiver.Process for dataReceiver in self.tcpipServer.DataReceivers]
        processes.sort(key=lambda process: process.Name)

        if not self._cv_dlg:
            self._cv_dlg = daeChooseVariable(daeChooseVariable.plot3D)
        self._cv_dlg.updateProcessesList(processes)
        self._cv_dlg.setWindowTitle('Choose variable for 3D plot')
        if self._cv_dlg.exec_() != QtGui.QDialog.Accepted:
            return False
            
        variable, domainIndexes, domainPoints, xAxisLabel, yAxisLabel, zAxisLabel, xPoints, yPoints, zPoints, currentTime = self._cv_dlg.getPlot3DData()
        xPoints = numpy.array(xPoints)
        yPoints = numpy.array(yPoints)

        xmax=numpy.max(xPoints)
        ymax=numpy.max(yPoints)
        zmax=numpy.max(zPoints)

        xmin=numpy.min(xPoints)
        ymin=numpy.min(yPoints)
        zmin=numpy.min(zPoints)

        warp = 'auto'
        #if((xmax == xmin) or (ymax == ymin) or (zmax == zmin)):
        #    warp = 'auto'
        #else:
        #    warp = math.sqrt( (xmax-xmin)*(ymax-ymin) ) / (zmax-zmin)
               
        # colormap='gist_earth', 'RdBu'
        stype = 'surface'
        mlab.figure()
        if(stype == 'surface'):
            #print "warp=", warp
            #print "[xmin, xmax, ymin, ymax, zmin, zmax]=", [xmin, xmax, ymin, ymax, zmin, zmax]
            mlab.surf(xPoints, yPoints, zPoints, warp_scale=warp, representation='surface')
            mlab.colorbar(orientation='vertical')
            #mlab.title('polar mesh')
            #mlab.outline()
            mlab.axes(ranges=[xmin, xmax, ymin, ymax, zmin, zmax], nb_labels=3)

            mlab.xlabel(xAxisLabel)
            mlab.ylabel(yAxisLabel)
            mlab.zlabel(zAxisLabel)
        elif(stype == 'map'):
            mlab.imshow(zPoints)
            mlab.colorbar(orientation='vertical')
            #mlab.title('polar mesh')
            #mlab.outline()
            mlab.axes(ranges=[xmin, xmax, ymin, ymax], nb_labels=3)

            mlab.xlabel(xAxisLabel)
            mlab.ylabel(yAxisLabel)
            mlab.zlabel(zAxisLabel)
            
        mlab.show()
       
    @staticmethod
    def showVTKFile_2D(filename):
        figure    = mlab.figure(size=(800, 600))
        vtkSource = VTKFileReader()
        vtkSource.initialize(filename)
        surface   = mlab.pipeline.surface(vtkSource)
        axes      = mlab.axes()
        colorbar  = mlab.colorbar(object = surface, orientation='horizontal')
        mlab.view(0, 0)
        mlab.show()
        
    @staticmethod
    def saveVTKFilesAsImages_2D(sourceFolder, destinationFolder):
        if not os.path.isdir(sourceFolder) or not os.path.isdir(destinationFolder):
            return
        
        vtkFiles = []
        for f in sorted(os.listdir(sourceFolder)):
            if f.endswith(".vtk"):
                vtkFiles.append(f)

        if len(vtkFiles) == 0:
            return
        
        figure    = mlab.figure(size=(800, 600))
        figure.scene.disable_render = True
        vtkSource = VTKFileReader()
        vtk_file  = os.path.join(sourceFolder, vtkFiles[0])
        vtkSource.initialize(vtk_file)                
        surface   = mlab.pipeline.surface(vtkSource)
        axes      = mlab.axes()
        colorbar  = mlab.colorbar(object = surface, orientation='horizontal')
        mlab.view(0, 0)
        figure.scene.disable_render = False
        mlab.draw()
        png_file = os.path.join(destinationFolder, vtkFiles[0]).replace('.vtk', '.png')
        mlab.savefig(png_file)
        
        for f in vtkFiles[1:-1]:
            vtk_file = os.path.join(sourceFolder, f)
            vtkSource.initialize(vtk_file)   
            png_file = os.path.join(destinationFolder, f).replace('.vtk', '.png')
            mlab.savefig(png_file)
            app = QtCore.QCoreApplication.instance()
            if app:
                app.processEvents()
        
    @staticmethod
    def animateVTKFiles_2D(folder):
        if not os.path.isdir(folder):
            return
        
        vtkFiles = []
        for f in sorted(os.listdir(folder)):
            if f.endswith(".vtk"):
                vtkFiles.append(os.path.join(folder, f))

        if len(vtkFiles) == 0:
            return
        
        figure    = mlab.figure(size=(800, 600))
        figure.scene.disable_render = True
        vtkSource = VTKFileReader()
        vtk_file  = vtkFiles[0]
        vtkSource.initialize(vtk_file)                
        surface   = mlab.pipeline.surface(vtkSource)
        axes      = mlab.axes()
        colorbar  = mlab.colorbar(object = surface, orientation='horizontal')
        mlab.view(0, 0)
        figure.scene.disable_render = False
        mlab.draw()
        
        a = animateVTKFiles(figure, vtkSource, vtkFiles)

                
"""
for f in vtkFiles[1:-1]:
    vtk_file = os.path.join(folder, f)
    vtkSource.initialize(vtk_file)                
    mlab.draw()
    time.sleep(0.5)
    app = QtCore.QCoreApplication.instance()
    if app:
        app.processEvents()
        
#figure.render()
#mlab.view(0, 0)
#mlab.savefig(vtk_file + '.png', figure=figure)
#figure.scene.disable_render = False
#mlab.show()
#mayavi.tools.figure.clf(figure)
                
if counter == 0:
    figure = mlab.figure(size=(800, 600))
    surface = mlab.pipeline.surface(src)
    axes = mlab.axes()
    colorbar = mlab.colorbar(object=surface, orientation='horizontal')
    print colorbar
    mlab.view(0, 0)
else:
    figure = mlab.gcf()
    figure.scene.disable_render = True
    mayavi.tools.figure.clf(figure)
    surface = mlab.pipeline.surface(src)
    #surface.actor.property.representation = "wireframe"
    axes = mlab.axes()
    colorbar = mlab.colorbar(object=surface, orientation='horizontal')
    #surface.remove()
    #surface = surface_new
    figure.scene.disable_render = False
"""
