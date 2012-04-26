"""********************************************************************************
                             daeMayavi3DPlot.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the 
terms of the GNU General Public License version 3 as published by the Free Software 
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
********************************************************************************"""
import sys, math
import numpy
from PyQt4 import QtCore, QtGui
from daetools.pyDAE import *
from daeChooseVariable import daeChooseVariable, daeTableDialog
try:
    from enthought.mayavi import mlab
except ImportError:
    from mayavi import mlab
    
class daeMayavi3DPlot:
    def __init__(self, tcpipServer):
        self.tcpipServer = tcpipServer        
                
    def newSurface(self):
        NoOfProcesses = self.tcpipServer.NumberOfProcesses
        processes = []
        for i in range(0, NoOfProcesses):
            processes.append(self.tcpipServer.GetProcess(i))
        cv = daeChooseVariable(processes, daeChooseVariable.plot3D)
        cv.setWindowTitle('Choose variable for 3D plot')
        if cv.exec_() != QtGui.QDialog.Accepted:
            return False
            
        variable, domainIndexes, domainPoints, xAxisLabel, yAxisLabel, zAxisLabel, xPoints, yPoints, zPoints, currentTime = cv.getPlot3DData()
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
       

