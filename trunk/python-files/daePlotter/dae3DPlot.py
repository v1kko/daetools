"""********************************************************************************
                             dae3DPlot.py
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

try:
    import numpy
except ImportError, e:
    print '[dae3DPlot]: Cannot load numpy module', str(e)

try:
    from daePlotOptions import *
    from daetools.pyDAE import *
    from daeChooseVariable import daeChooseVariable, daeTableDialog
except ImportError, e:
    print '[dae3DPlot]: Cannot load daetools modules', str(e)

try:
    from PyQt4 import QtCore, QtGui
except ImportError, e:
    print '[dae3DPlot]: Cannot load pyQt4 modules', str(e)

try:
    import matplotlib
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import LogNorm
except ImportError, e:
    print '[dae3DPlot]: Cannot load matplotlib modules', str(e)


class dae3DPlot(QtGui.QDialog):
    def __init__(self, parent, tcpipServer):
        QtGui.QDialog.__init__(self, parent, QtCore.Qt.Window)
        
        self.tcpipServer = tcpipServer        
        self.setWindowTitle("3D plot")
        self.setWindowIcon(QtGui.QIcon('images/line-chart.png'))

        exit = QtGui.QAction(QtGui.QIcon('images/close.png'), 'Exit', self)
        exit.setShortcut('Ctrl+Q')
        exit.setStatusTip('Exit application')
        self.connect(exit, QtCore.SIGNAL('triggered()'), self, QtCore.SLOT('close()'))

        properties = QtGui.QAction(QtGui.QIcon('images/preferences.png'), 'Options', self)
        properties.setShortcut('Ctrl+P')
        properties.setStatusTip('Options')
        self.connect(properties, QtCore.SIGNAL('triggered()'), self, QtCore.SLOT('slotProperties()'))

        self.toolbar_widget = QtGui.QWidget(self)
        layoutToolbar = QtGui.QVBoxLayout(self.toolbar_widget)
        layoutToolbar.setContentsMargins(0,0,0,0)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        self.toolbar_widget.setSizePolicy(sizePolicy)

        layoutPlot = QtGui.QVBoxLayout(self)
        layoutPlot.setContentsMargins(2,2,2,2)

        self.figure = Figure((6.0, 4.0), dpi=100, facecolor="#E5E5E5") #(0.9, 0.9, 0.9))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)        
        self.canvas.axes = Axes3D(self.figure)

        self.mpl_toolbar = NavigationToolbar(self.canvas, self.toolbar_widget, False)
        actions = self.mpl_toolbar.actions()
        for i in xrange(8): # All actions until Save
            self.mpl_toolbar.removeAction(actions[i])
        
        self.mpl_toolbar.addSeparator()
        self.mpl_toolbar.addAction(properties)
        
        self.fp9  = matplotlib.font_manager.FontProperties(family='sans-serif', style='normal', variant='normal', weight='normal', size=9)
        self.fp10 = matplotlib.font_manager.FontProperties(family='sans-serif', style='normal', variant='normal', weight='normal', size=10)
        self.fp11 = matplotlib.font_manager.FontProperties(family='sans-serif', style='italic', variant='normal', weight='normal', size=11)
        
        self.canvas.axes.grid(True)
        self.canvas.axes.legend(loc = 0, prop=self.fp9)
        
        layoutToolbar.addWidget(self.mpl_toolbar)
        layoutPlot.addWidget(self.canvas)
        layoutPlot.addWidget(self.toolbar_widget)
                
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
        
        if nx == ny:
            # Colormaps: autumn, bone, cool, copper, flag, gray, hot, hsv, jet, pink, prism, spring, summer, winte, spectral
            # ACHTUNG ACHTUNG!!! Check!!
            # meshgrid creates a matrix of fortran type 
            xPoints, yPoints = numpy.meshgrid(xPoints, yPoints)
            xPoints, yPoints = numpy.transpose(xPoints), numpy.transpose(yPoints)
            #print xPoints
            #print yPoints
            #print zPoints
            self.canvas.axes.plot_surface(xPoints, yPoints, zPoints, rstride=1, cstride=1, cmap=matplotlib.cm.jet)
            self.canvas.axes.plot_wireframe(xPoints, yPoints, zPoints, rstride=1, cstride=1, linewidth=0, color='blue')
            
        else:
            # I need equal number of points in x and ydomains
            # Therefore, I have to reshape arrays to get equal number of points
            #   in each of domains (both x, y, and z)
            xs = 1 # x stride/step
            ys = 1 # y stride/step
            np = 1 # no of points (final)
            
            if nx > ny:
                xs = (nx-1.)/(ny-1.)
                ys = 1
                np = ny
            else:
                ys = (ny-1.)/(nx-1.)
                xs = 1
                np = nx
            
            npx = int(np*xs) # new number of points in x domain
            npy = int(np*ys) # new number of points in y domain

            X = numpy.zeros(np)      # new x domain points
            Y = numpy.zeros(np)      # new y domain points
            Z = numpy.zeros((np,np)) # new z domain values
            
            xind = []
            yind = []
            for i in range(np):
                xind.append( int( math.ceil(i*xs) ) )
                yind.append( int( math.ceil(i*ys) ) ) 
                
            #print "xind:"
            #print xind
            #print "yind:"
            #print yind
                
            for i in range(np):
                X[i] = xPoints[xind[i]]
                Y[i] = yPoints[yind[i]]
                for k in range(np):
                    Z[i,k] = zPoints[xind[i], yind[k]]
                    
            #print "Initial no of points:", nx, ny
            #print "New no of points:", np, npx, npy
            #print "New no of points:", len(X), len(Y)
            #print "Steps:", xs, ys
            #print "New x points:"
            #print X
            #print "New y points:"
            #print Y
            #print "New z values:"
            #print Z
                    
            #minx = min(X) 
            #maxx = max(X)
            #miny = min(Y)
            #maxy = max(Y)
            #minz = 390 #1.1 * min(Z)
            #maxz = 410 #0.9 * max(Z)

            # ACHTUNG ACHTUNG!!! Check
            # meshgrid creates a matrix of fortran type 
            X, Y = numpy.meshgrid(X, Y)
            X, Y = numpy.transpose(X), numpy.transpose(Y)
            self.canvas.axes.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.jet)
            self.canvas.axes.plot_wireframe(X, Y, Z, rstride=1, cstride=1, linewidth=0, color='blue')
            
        self.canvas.axes.set_xlabel(xAxisLabel, fontproperties=self.fp11)
        self.canvas.axes.set_ylabel(yAxisLabel, fontproperties=self.fp11)
        self.canvas.axes.set_zlabel(zAxisLabel, fontproperties=self.fp11)

        for xlabel in self.canvas.axes.get_xticklabels():
            xlabel.set_fontproperties(self.fp10)
        for ylabel in self.canvas.axes.get_yticklabels():
            ylabel.set_fontproperties(self.fp10)
            
        #print "xAxisLabel ", xAxisLabel
        #print "yAxisLabel ", yAxisLabel
        #print "zAxisLabel ", zAxisLabel
        #print zAxisLabel, str(domainIndexes)
        #print "xPoints "
        #print xPoints
        #print "yPoints "
        #print yPoints
        #print "zPoints "
        #print zPoints

        title = "("
        for i in range(0, len(domainIndexes)):
            if i != 0:
                title += ", "
            title += domainIndexes[i]
        title += ")"
            
        self.setWindowTitle(zAxisLabel + title)
        self.canvas.draw()
        
        return True
          
    #@QtCore.pyqtSlot()
    def slotProperties(self):
        surface_edit(self.canvas, self)

        

