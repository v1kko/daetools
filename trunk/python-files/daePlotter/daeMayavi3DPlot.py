import sys, math

try:
    import numpy
except ImportError, e:
    print '[daeMayavi3DPlot]: Cannot load numpy module', str(e)

try:
    from PyQt4 import QtCore, QtGui
except ImportError, e:
    print '[daeMayavi3DPlot]: Cannot load pyQt4 modules', str(e)

try:
    from daetools.pyDAE import *
    from daeChooseVariable import daeChooseVariable, daeTableDialog
except ImportError, e:
    print '[daeMayavi3DPlot]: Cannot load daetools modules', str(e)

try:
    from enthought.mayavi import mlab, contourf
except ImportError, e:
    print '[daeMayavi3DPlot]: Cannot load mayavi module', str(e)


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
            
        domainIndexes, xAxisLabel, yAxisLabel, zAxisLabel, xPoints, yPoints, zPoints = cv.getPlot3DData()
        nx = len(xPoints) # no of points in x domain
        ny = len(yPoints) # no of points in y domain
        xPoints = numpy.array(xPoints)
        yPoints = numpy.array(yPoints)

        xmax=numpy.max(xPoints)
        ymax=numpy.max(yPoints)
        zmax=numpy.max(zPoints)

        xmin=numpy.min(xPoints)
        ymin=numpy.min(yPoints)
        zmin=numpy.min(zPoints)

        warp = 1
        if((xmax == xmin) or (ymax == ymin) or (zmax == zmin)):
            warp = 'auto'
        else:
            warp = math.sqrt( (xmax-xmin)*(ymax-ymin) ) / (zmax-zmin)
               
        # colormap='gist_earth', 'RdBu'
        stype = 'surface'
        if(stype == 'surface'):
            #print "warp=", warp
            #print "[xmin, xmax, ymin, ymax, zmin, zmax]=", [xmin, xmax, ymin, ymax, zmin, zmax]
            mlab.surf(xPoints, yPoints, zPoints, warp_scale=warp, representation='surface')
            mlab.colorbar(orientation='vertical')
            #mlab.title('polar mesh')
            #mlab.outline()
            mlab.axes(ranges=[xmin, xmax, ymin, ymax, zmin, zmax], nb_labels=4)

            mlab.xlabel(xAxisLabel)
            mlab.ylabel(yAxisLabel)
            mlab.zlabel(zAxisLabel)
        elif(stype == 'map'):
            mlab.imshow(zPoints)
            mlab.colorbar(orientation='vertical')
            #mlab.title('polar mesh')
            #mlab.outline()
            mlab.axes(ranges=[xmin, xmax, ymin, ymax], nb_labels=4)

            mlab.xlabel(xAxisLabel)
            mlab.ylabel(yAxisLabel)
            mlab.zlabel(zAxisLabel)
            
        mlab.show()
       

