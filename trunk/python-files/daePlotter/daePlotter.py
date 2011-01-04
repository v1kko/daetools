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
import os, sys, distutils.sysconfig

try:
    from PyQt4 import QtCore, QtGui
except Exception, e:
    print '[daePlotter]: Cannot load pyQt4 modules\n Error: ', str(e)
    sys.exit()

try:
    import numpy
except Exception, e:
    print '[daePlotter]: Cannot load numpy module\n Error: ', str(e)
    sys.exit()

try:
    from daetools.pyDAE import *
except Exception, e:
    print '[daePlotter]: Cannot load daetools.pyDAE module\n Error: ', str(e)
    sys.exit()

try:
    from about import AboutDialog
    from daetools.pyDAE.WebViewDialog import WebView
except Exception, e:
    print '[daePlotter]: Cannot load about/WebView module\n Error: ', str(e)


class daeMainWindow(QtGui.QMainWindow):
    def __init__(self, tcpipServer):
        QtGui.QMainWindow.__init__(self)
        
        self.tcpipServer = tcpipServer

        self.move(0, 0)
        self.resize(400, 200)
        self.setWindowIcon(QtGui.QIcon('images/app.png'))
        self.setWindowTitle("DAE Plotter v" + daeVersion(True))

        exit = QtGui.QAction(QtGui.QIcon('images/close.png'), 'Exit', self)
        exit.setShortcut('Ctrl+Q')
        exit.setStatusTip('Exit application')
        self.connect(exit, QtCore.SIGNAL('triggered()'), self.close)

        plot2D = QtGui.QAction(QtGui.QIcon('images/add-2d.png'), 'New 2D plot', self)
        plot2D.setShortcut('Ctrl+2')
        plot2D.setStatusTip('New 2D plot')
        self.connect(plot2D, QtCore.SIGNAL('triggered()'), self.slotPlot2D)

        plot3D = QtGui.QAction(QtGui.QIcon('images/add-3d.png'), 'New 3D plot', self)
        plot3D.setShortcut('Ctrl+3')
        plot3D.setStatusTip('New 3D plot')
        self.connect(plot3D, QtCore.SIGNAL('triggered()'), self.slotPlot3D)

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
        plot.addAction(plot3D)
        
        help = menubar.addMenu('&Help')
        help.addAction(about)
        help.addAction(docs)

        self.toolbar = self.addToolBar('Main toolbar')

        self.toolbar.addAction(plot2D)
        self.toolbar.addAction(plot3D)

    #@QtCore.pyqtSlot()
    def slotPlot2D(self):
        try:
            from dae2DPlot import dae2DPlot
        except Exception, e:
            QtGui.QMessageBox.warning(None, "daePlotter", "Cannot load 2D Plot module.\nDid you forget to install Matplotlib?\nError: " + str(e))
            return
            
        plot2d = dae2DPlot(self, self.tcpipServer)
        if plot2d.newCurve() == False:
            plot2d.close()
            del plot2d
        else:
            plot2d.show()
        
    #@QtCore.pyqtSlot()
    def slotPlot3D(self):
        try:
            from daeMayavi3DPlot import daeMayavi3DPlot
        except Exception, e:
            QtGui.QMessageBox.warning(self, "daePlotter", "Cannot load 3D Plot module.\nDid you forget to install Mayavi2?\nError: " + str(e))
            return
        
        plot3d = daeMayavi3DPlot(self.tcpipServer)
        if plot3d.newSurface() == False:
            del plot3d

    #@QtCore.pyqtSlot()
    def slotAbout(self):
        about = AboutDialog(self)
        about.exec_()
    
    #@QtCore.pyqtSlot()
    def slotDocumentation(self):
        site_packages = distutils.sysconfig.get_python_lib()
        url = QtCore.QUrl(site_packages + '/daetools/docs/index.html')
        wv = WebView(url)
        wv.resize(800, 550)
        wv.setWindowState(QtCore.Qt.WindowMaximized)
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
    daeStartPlotter()