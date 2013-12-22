#!/usr/bin/env python
"""********************************************************************************
                             daePlotter.py
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
import os, sys, distutils.sysconfig, types, json
from os.path import join, realpath, dirname

python_major = sys.version_info[0]
python_minor = sys.version_info[1]
python_build = sys.version_info[2]

try:
    from PyQt4 import QtCore, QtGui
except Exception as e:
    print(('[daePlotter]: Cannot load pyQt4 modules\n Error: ', str(e)))
    sys.exit()

try:
    import numpy
except Exception as e:
    print(('[daePlotter]: Cannot load numpy module\n Error: ', str(e)))
    sys.exit()

try:
    from daetools.pyDAE import *
except Exception as e:
    print(('[daePlotter]: Cannot load daetools.pyDAE module\n Error: ', str(e)))
    sys.exit()

try:
    from .about import daeAboutDialog
    from daetools.pyDAE.web_view_dialog import daeWebView
except Exception as e:
    print(('[daePlotter]: Cannot load daeAbout/daeWebView module\n Error: ', str(e)))

try:
    images_dir = join(dirname(__file__), 'images')
except:
    # In case we are running the module on its own (i.e. as __main__)
    images_dir = join(realpath(dirname(sys.argv[0])), 'images')
    
class daeMainWindow(QtGui.QMainWindow):
    def __init__(self, tcpipServer):
        QtGui.QMainWindow.__init__(self)
        
        self.tcpipServer = tcpipServer

        self.move(0, 0)
        self.resize(400, 200)
        self.setWindowIcon(QtGui.QIcon(join(images_dir, 'daetools-48x48.png')))
        self.setWindowTitle("DAE Plotter v%s [py%d.%d]" % (daeVersion(True), python_major, python_minor))

        exit = QtGui.QAction(QtGui.QIcon(join(images_dir, 'close.png')), 'Exit', self)
        exit.setShortcut('Ctrl+Q')
        exit.setStatusTip('Exit application')
        self.connect(exit, QtCore.SIGNAL('triggered()'), self.close)

        plot2D = QtGui.QAction(QtGui.QIcon(join(images_dir, 'add-2d.png')), 'New 2D plot...', self)
        plot2D.setShortcut('Ctrl+2')
        plot2D.setStatusTip('New 2D plot')
        self.connect(plot2D, QtCore.SIGNAL('triggered()'), self.slotPlot2D)

        animatedPlot2D = QtGui.QAction(QtGui.QIcon(join(images_dir, 'add-ani-2d.png')), 'New animated 2D plot...', self)
        animatedPlot2D.setShortcut('Ctrl+4')
        animatedPlot2D.setStatusTip('New animated 2D plot')
        self.connect(animatedPlot2D, QtCore.SIGNAL('triggered()'), self.slotAnimatedPlot2D)

        plot3D = QtGui.QAction(QtGui.QIcon(join(images_dir, 'add-3d.png')), 'New 3D plot...', self)
        plot3D.setShortcut('Ctrl+3')
        plot3D.setStatusTip('New 3D plot')
        self.connect(plot3D, QtCore.SIGNAL('triggered()'), self.slotPlot3D)

        plotVTK_2D = QtGui.QAction(QtGui.QIcon(join(images_dir, 'vtk.png')), 'Open 2D VTK plot from file...', self)
        plotVTK_2D.setShortcut('Ctrl+4')
        plotVTK_2D.setStatusTip('Open 2D VTK plot from file')
        self.connect(plotVTK_2D, QtCore.SIGNAL('triggered()'), self.slotOpenVTK_2D)

        saveVTKasImages_2D = QtGui.QAction(QtGui.QIcon(join(images_dir, 'vtk.png')), 'Export 2D VTK files to PNG images...', self)
        saveVTKasImages_2D.setShortcut('Ctrl+5')
        saveVTKasImages_2D.setStatusTip('Export 2D VTK files to PNG images...')
        self.connect(saveVTKasImages_2D, QtCore.SIGNAL('triggered()'), self.slotSaveVTKFilesAsImages_2D)

        animateVTKFiles_2D = QtGui.QAction(QtGui.QIcon(join(images_dir, 'vtk.png')), 'Animate 2D VTK plots...', self)
        animateVTKFiles_2D.setShortcut('Ctrl+6')
        animateVTKFiles_2D.setStatusTip('Animate 2D VTK plots...')
        self.connect(animateVTKFiles_2D, QtCore.SIGNAL('triggered()'), self.slotAnimateVTKFiles_2D)

        openTemplate = QtGui.QAction(QtGui.QIcon(join(images_dir, 'template.png')), 'Open 2D template...', self)
        openTemplate.setShortcut('Ctrl+T')
        openTemplate.setStatusTip('Open 2D plot emplate')
        self.connect(openTemplate, QtCore.SIGNAL('triggered()'), self.slotOpenTemplate)

        about = QtGui.QAction('About', self)
        about.setStatusTip('About')
        self.connect(about, QtCore.SIGNAL('triggered()'), self.slotAbout)

        docs = QtGui.QAction('Documentation', self)
        docs.setStatusTip('Documentation')
        self.connect(docs, QtCore.SIGNAL('triggered()'), self.slotDocumentation)

        self.statusBar()

        menubar = self.menuBar()
        file = menubar.addMenu('&File')
        file.addAction(exit)

        plot = menubar.addMenu('&Plot')
        plot.addAction(plot2D)
        plot.addAction(animatedPlot2D)
        plot.addAction(plot3D)
        plot.addSeparator()
        plot.addAction(openTemplate)
        plot.addSeparator()
        plot.addAction(plotVTK_2D)
        plot.addAction(saveVTKasImages_2D)
        plot.addAction(animateVTKFiles_2D)

        help = menubar.addMenu('&Help')
        help.addAction(about)
        help.addAction(docs)

        self.toolbar = self.addToolBar('Main toolbar')

        self.toolbar.addAction(plot2D)
        self.toolbar.addAction(animatedPlot2D)
        self.toolbar.addAction(plot3D)
        self.toolbar.addSeparator()
        self.toolbar.addAction(openTemplate)
        self.toolbar.addSeparator()
        self.toolbar.addAction(plotVTK_2D)

    #@QtCore.pyqtSlot()
    def slotPlot2D(self):
        self.plot2D()

    #@QtCore.pyqtSlot()
    def slotAnimatedPlot2D(self):
        msecs, ok = QtGui.QInputDialog.getInteger(self, 'Insert the update interval', 'Interval (msecs):', 1000, 1, 1E5, 100)
        if ok:
            self.plot2D(msecs) # 1000 ms

    def plot2D(self, updateInterval = 0):
        try:
            from .plot2d import dae2DPlot
        except Exception as e:
            QtGui.QMessageBox.warning(None, "daePlotter", "Cannot load 2D Plot module.\nDid you forget to install Matplotlib?\nError: " + str(e))
            return
            
        plot2D = dae2DPlot(self, self.tcpipServer, updateInterval)
        if plot2D.newCurve() == False:
            plot2D.close()
            del plot2D
        else:
            plot2D.show()

    #@QtCore.pyqtSlot()
    def slotPlot3D(self):
        try:
            from .mayavi_plot3d import daeMayavi3DPlot
        except Exception as e:
            QtGui.QMessageBox.warning(self, "daePlotter", "Cannot load 3D Plot module.\nDid you forget to install Mayavi2?\nError: " + str(e))
            return
        
        plot3D = daeMayavi3DPlot(self.tcpipServer)
        if plot3D.newSurface() == False:
            del plot3D

    #@QtCore.pyqtSlot()
    def slotOpenVTK_2D(self):
        filename = str(QtGui.QFileDialog.getOpenFileName(self, "Choose VTK File", '', "VTK Files (*.vtk)"))
        if filename == '':
            return
        try:
            from .mayavi_plot3d import daeMayavi3DPlot
        except Exception as e:
            QtGui.QMessageBox.warning(self, "daePlotter", "Cannot load 3D Plot module.\nDid you forget to install Mayavi2?\nError: " + str(e))
            return

        daeMayavi3DPlot.showVTKFile_2D(filename)

    #@QtCore.pyqtSlot()
    def slotSaveVTKFilesAsImages_2D(self):
        sourceFolder = str(QtGui.QFileDialog.getExistingDirectory(self, "Choose the source folder with VTK files", '', QtGui.QFileDialog.ShowDirsOnly | QtGui.QFileDialog.DontResolveSymlinks))
        if sourceFolder == '':
            return
            
        destinationFolder = str(QtGui.QFileDialog.getExistingDirectory(self, "Choose the destination folder for generated PNG images", '', QtGui.QFileDialog.ShowDirsOnly | QtGui.QFileDialog.DontResolveSymlinks))
        if destinationFolder == '':
            return
            
        try:
            from .mayavi_plot3d import daeMayavi3DPlot
        except Exception as e:
            QtGui.QMessageBox.warning(self, "daePlotter", "Cannot load 3D Plot module.\nDid you forget to install Mayavi2?\nError: " + str(e))
            return

        daeMayavi3DPlot.saveVTKFilesAsImages_2D(sourceFolder, destinationFolder)
        
    #@QtCore.pyqtSlot()
    def slotAnimateVTKFiles_2D(self):
        folder = str(QtGui.QFileDialog.getExistingDirectory(self, "Choose the folder with VTK files", '', QtGui.QFileDialog.ShowDirsOnly | QtGui.QFileDialog.DontResolveSymlinks))
        if folder == '':
            return
        try:
            from .mayavi_plot3d import daeMayavi3DPlot
        except Exception as e:
            QtGui.QMessageBox.warning(self, "daePlotter", "Cannot load 3D Plot module.\nDid you forget to install Mayavi2?\nError: " + str(e))
            return

        daeMayavi3DPlot.animateVTKFiles_2D(folder)

    #@QtCore.pyqtSlot()
    def slotOpenTemplate(self):
        try:
            filename = QtGui.QFileDialog.getOpenFileName(self, "Open 2D plot template", "", "Templates (*.pt)")
            if not filename:
                return

            f = open(filename, 'r')
            s = f.read(-1)
            template = json.loads(s)
            
            updateInterval = template['updateInterval']
            curves         = template['curves']
            
            try:
                from .plot2d import dae2DPlot
            except Exception as e:
                QtGui.QMessageBox.warning(None, "daePlotter", "Cannot load 2D Plot module.\nDid you forget to install Matplotlib?\nError: " + str(e))
                return

            plot2D = dae2DPlot(self, self.tcpipServer, updateInterval)
            if plot2D.newFromTemplate(curves) == False:
                plot2D.close()
                del plot2D
            else:
                plot2D.show()

        except Exception as e:
            print((str(e)))

    #@QtCore.pyqtSlot()
    def slotAbout(self):
        about = daeAboutDialog(self)
        about.exec_()
    
    #@QtCore.pyqtSlot()
    def slotDocumentation(self):
        #site_packages = distutils.sysconfig.get_python_lib()
        docs_dir = os.path.join(os.path.dirname(__file__), '../docs/html/index.html')
        url = QtCore.QUrl(docs_dir)
        wv = daeWebView(url)
        wv.resize(800, 550)
        #wv.setWindowState(QtCore.Qt.WindowMaximized)
        wv.setWindowTitle("DAE Tools documentation")
        wv.exec_()
    
def daeStartPlotter(port = 0):
    try:
        if port == 0:
            cfg = daeGetConfig()
            port = cfg.GetInteger("daetools.datareporting.tcpipDataReceiverPort", 50000)

        tcpipServer = daeTCPIPDataReceiverServer(port)
        tcpipServer.Start()
        if(tcpipServer.IsConnected() == False):
            return

        app = QtGui.QApplication(sys.argv)
        main = daeMainWindow(tcpipServer)
        main.show()
        app.exec_()
        tcpipServer.Stop()
    
    except RuntimeError:
        return
    
if __name__ == "__main__":
    if (len(sys.argv) > 1):
        daeStartPlotter(int(sys.argv[1]))
    else:
        daeStartPlotter()