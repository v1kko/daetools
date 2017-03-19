#!/usr/bin/env python
"""********************************************************************************
                             plotter.py
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
import os, sys, distutils.sysconfig, types, json, numpy, webbrowser
from os.path import join, realpath, dirname
from PyQt5 import QtCore, QtGui, QtWidgets
from daetools.pyDAE import *
from .choose_variable import daeChooseVariable
from .custom_plots import daeCustomPlots
from .about import daeAboutDialog
from .plot2d import dae2DPlot
from .user_data import daeUserData

python_major = sys.version_info[0]
python_minor = sys.version_info[1]
python_build = sys.version_info[2]

try:
    images_dir = join(dirname(__file__), 'images')
except:
    # In case we are running the module on its own (i.e. as __main__)
    images_dir = join(realpath(dirname(sys.argv[0])), 'images')

class daeMainWindow(QtWidgets.QMainWindow):
    def __init__(self, tcpipServer):
        QtWidgets.QMainWindow.__init__(self)

        self.tcpipServer = tcpipServer

        self.move(0, 0)
        self.resize(400, 200)
        self.setWindowIcon(QtGui.QIcon(join(images_dir, 'daetools-48x48.png')))
        self.setWindowTitle("DAE Tools Plotter v%s [py%d.%d]" % (daeVersion(True), python_major, python_minor))

        exit = QtWidgets.QAction(QtGui.QIcon(join(images_dir, 'close.png')), 'Exit', self)
        exit.setShortcut('Ctrl+Q')
        exit.setStatusTip('Exit application')
        exit.triggered.connect(self.close)

        plot2D = QtWidgets.QAction(QtGui.QIcon(join(images_dir, 'add-2d.png')), 'New 2D plot...', self)
        plot2D.setShortcut('Ctrl+2')
        plot2D.setStatusTip('New 2D plot')
        plot2D.triggered.connect(self.slotPlot2D)

        autoupdatePlot2D = QtWidgets.QAction(QtGui.QIcon(join(images_dir, 'add-autoupdate-2d.png')), 'New auto update 2D plot...', self)
        autoupdatePlot2D.setShortcut('Ctrl+U')
        autoupdatePlot2D.setStatusTip('New auto update 2D plot')
        autoupdatePlot2D.triggered.connect(self.slotPlot2DAutoUpdate)

        animatedPlot2D = QtWidgets.QAction(QtGui.QIcon(join(images_dir, 'add-ani-2d.png')), 'New animated 2D plot...', self)
        animatedPlot2D.setShortcut('Ctrl+A')
        animatedPlot2D.setStatusTip('New animated 2D plot')
        animatedPlot2D.triggered.connect(self.plot2DAnimated)

        customPlots = QtWidgets.QAction('New user-defined plot...', self)
        customPlots.setShortcut('Ctrl+C')
        customPlots.setStatusTip('New user-defined plot')
        customPlots.triggered.connect(self.slotCustomPlots)

        fromUserData = QtWidgets.QAction('New plot from the user-provided data...', self)
        fromUserData.setShortcut('Ctrl+D')
        fromUserData.setStatusTip('New plot from the user-provided data')
        fromUserData.triggered.connect(self.slotFromUserData)

        plot3D = QtWidgets.QAction(QtGui.QIcon(join(images_dir, 'add-3d.png')), 'New Mayavi 3D plot...', self)
        plot3D.setShortcut('Ctrl+3')
        plot3D.setStatusTip('New Mayavi 3D plot')
        plot3D.triggered.connect(self.slotPlot3D)

        #matplotlibSurfacePlot = QtWidgets.QAction('New matplotlib surface plot...', self)
        #matplotlibSurfacePlot.setShortcut('Ctrl+S')
        #matplotlibSurfacePlot.setStatusTip('New matplotlib surface plot')
        #matplotlibSurfacePlot.triggered.connect(self.slotMatplotlibSurfacePlot)

        plotVTK_2D = QtWidgets.QAction(QtGui.QIcon(join(images_dir, 'vtk.png')), 'Open 2D VTK plot from file...', self)
        plotVTK_2D.setShortcut('Ctrl+V')
        plotVTK_2D.setStatusTip('Open 2D VTK plot from file')
        plotVTK_2D.triggered.connect(self.slotOpenVTK_2D)

        saveVTKasImages_2D = QtWidgets.QAction('Export 2D VTK files to PNG images...', self)
        saveVTKasImages_2D.setShortcut('Ctrl+P')
        saveVTKasImages_2D.setStatusTip('Export 2D VTK files to PNG images...')
        saveVTKasImages_2D.triggered.connect(self.slotSaveVTKFilesAsImages_2D)

        #animateVTKFiles_2D = QtWidgets.QAction(QtGui.QIcon(join(images_dir, 'vtk.png')), 'Animate 2D VTK plots...', self)
        #animateVTKFiles_2D.setShortcut('Ctrl+6')
        #animateVTKFiles_2D.setStatusTip('Animate 2D VTK plots...')
        #animateVTKFiles_2D.triggered.connect(self.slotAnimateVTKFiles_2D)

        openTemplate = QtWidgets.QAction(QtGui.QIcon(join(images_dir, 'template.png')), 'Open 2D template...', self)
        openTemplate.setShortcut('Ctrl+T')
        openTemplate.setStatusTip('Open 2D plot emplate')
        openTemplate.triggered.connect(self.slotOpenTemplate)

        about = QtWidgets.QAction('About', self)
        about.setStatusTip('About')
        about.triggered.connect(self.slotAbout)

        docs = QtWidgets.QAction('Documentation', self)
        docs.setStatusTip('Documentation')
        docs.triggered.connect(self.slotDocumentation)

        self.statusBar()

        menubar = self.menuBar()
        file = menubar.addMenu('&File')
        file.addAction(exit)

        plot = menubar.addMenu('&Plot')
        plot.addAction(plot2D)
        plot.addAction(autoupdatePlot2D)
        plot.addAction(animatedPlot2D)
        plot.addAction(plot3D)
        #plot.addAction(matplotlibSurfacePlot)
        plot.addAction(customPlots)
        plot.addAction(fromUserData)
        plot.addSeparator()
        plot.addAction(openTemplate)
        plot.addSeparator()
        plot.addAction(plotVTK_2D)
        plot.addAction(saveVTKasImages_2D)
        #plot.addAction(animateVTKFiles_2D)

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
    def slotCustomPlots(self):
        processes = [dataReceiver.Process for dataReceiver in self.tcpipServer.DataReceivers]
        processes.sort(key=lambda process: process.Name)

        dlg = daeCustomPlots()
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        exec(dlg.plot_source, globals())
        if 'make_custom_plot' in globals():
            make_custom_plot(processes)

    def slotFromUserData(self):
        dlg = daeUserData()
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        plot2D = dae2DPlot(self, self.tcpipServer, 0.0)

        if plot2D.newCurveFromUserData(dlg.xLabel, dlg.yLabel, dlg.lineLabel, dlg.xPoints, dlg.yPoints) == False:
            plot2D.close()
            del plot2D
            return

        plot2D.show()

    #@QtCore.pyqtSlot()
    def slotPlot2D(self):
        self.plot2D()

    #@QtCore.pyqtSlot()
    def slotPlot2DAutoUpdate(self):
        msecs, ok = QtWidgets.QInputDialog.getInt(self, 'Insert the update interval', 'Interval (msecs):', 1000, 1, 1E5, 100)
        if ok:
            self.plot2D(msecs) # 1000 ms

    def plot2D(self, updateInterval = 0):
        plot2D = dae2DPlot(self, self.tcpipServer, updateInterval)
        if plot2D.newCurve() == False:
            plot2D.close()
            del plot2D
            return

        plot2D.show()

    def plot2DAnimated(self):
        plot2D = dae2DPlot(self, self.tcpipServer, 100, True) # 100 is some default, just to mark as animated plot
        if plot2D.newAnimatedCurve() == False:
            plot2D.close()
            del plot2D
            return

        plot2D.show()

    #@QtCore.pyqtSlot()
    def slotPlot3D(self):
        try:
            from .mayavi_plot3d import daeMayavi3DPlot
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "daePlotter", "Cannot load mayavi_plot3d module.\nDid you forget to install Mayavi2?\nError: " + str(e))
            return

        plot3D = daeMayavi3DPlot(self.tcpipServer)
        if plot3D.newSurface() == False:
            del plot3D

    """
    def slotMatplotlibSurfacePlot(self, updateInterval = 0):
        try:
            from .mpl_plot3d import dae3DPlot
        except Exception as e:
            QtWidgets.QMessageBox.warning(None, "daePlotter", "Cannot load mpl_plot3d module.\nError: " + str(e))
            return

        plot3D = dae3DPlot(self, self.tcpipServer, updateInterval)
        if plot3D.newSurface() == False:
            del plot3D
            return
        plot3D.show()
    """

    #@QtCore.pyqtSlot()
    def slotOpenVTK_2D(self):
        filename, ok = QtWidgets.QFileDialog.getOpenFileName(self, "Choose VTK File", '', "VTK Files (*.vtk)")
        if not ok:
            return
        try:
            from .mayavi_plot3d import daeMayavi3DPlot
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "daePlotter", "Cannot load 3D Plot module.\nDid you forget to install Mayavi2?\nError: " + str(e))
            return

        daeMayavi3DPlot.showVTKFile_2D(str(filename))

    #@QtCore.pyqtSlot()
    def slotSaveVTKFilesAsImages_2D(self):
        sourceFolder, ok = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose the source folder with VTK files", '', QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks)
        if not ok:
            return

        destinationFolder, ok = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose the destination folder for generated PNG images", '', QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks)
        if not ok:
            return

        try:
            from .mayavi_plot3d import daeMayavi3DPlot
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "daePlotter", "Cannot load 3D Plot module.\nDid you forget to install Mayavi2?\nError: " + str(e))
            return

        daeMayavi3DPlot.saveVTKFilesAsImages_2D(str(sourceFolder), str(destinationFolder))

    #@QtCore.pyqtSlot()
    def slotAnimateVTKFiles_2D(self):
        folder, ok = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose the folder with VTK files", '', QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks)
        if not ok:
            return
        try:
            from .mayavi_plot3d import daeMayavi3DPlot
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "daePlotter", "Cannot load 3D Plot module.\nDid you forget to install Mayavi2?\nError: " + str(e))
            return

        daeMayavi3DPlot.animateVTKFiles_2D(str(folder))

    #@QtCore.pyqtSlot()
    def slotOpenTemplate(self):
        try:
            filename, ok = QtWidgets.QFileDialog.getOpenFileName(self, "Open 2D plot template", "", "Templates (*.pt)")
            if not ok:
                return

            f = open(filename, 'r')
            s = f.read(-1)
            template = json.loads(s)

            curves         = template['curves']
            plotType       = int(template['plotType'])
            updateInterval = float(template['updateInterval'])

            if plotType == daeChooseVariable.plot2D or plotType == daeChooseVariable.plot2DAutoUpdated:
                plot2D = dae2DPlot(self, self.tcpipServer, updateInterval, False)
            elif plotType == daeChooseVariable.plot2DAnimated:
                plot2D = dae2DPlot(self, self.tcpipServer, updateInterval, True)
            else:
                raise RuntimeError('Invalid plot type')

            if plot2D.newFromTemplate(template) == False:
                plot2D.close()
                del plot2D
            else:
                plot2D.show()

        except Exception as e:
            print((str(e)))

    #@QtCore.pyqtSlot()
    def slotAbout(self):
        dlg = daeAboutDialog()
        dlg.exec_()

    #@QtCore.pyqtSlot()
    def slotDocumentation(self):
        docs_index = os.path.join(os.path.dirname(__file__), '../docs/html/index.html')
        #docs_index = 'http://www.daetools.com/docs/index.html'
        webbrowser.open(docs_index)

def daeStartPlotter(port = 0):
    try:
        if port == 0:
            cfg = daeGetConfig()
            port = cfg.GetInteger("daetools.datareporting.tcpipDataReceiverPort", 50000)

        tcpipServer = daeTCPIPDataReceiverServer(port)
        tcpipServer.Start()
        if(tcpipServer.IsConnected() == False):
            return

        app = QtWidgets.QApplication(sys.argv)
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
