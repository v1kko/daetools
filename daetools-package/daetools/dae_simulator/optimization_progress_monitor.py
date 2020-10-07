"""
***********************************************************************************
                      optimization_progress_monitor.py
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
************************************************************************************
"""
__doc__ = """
"""

import sys
from daetools.pyDAE import *
from os.path import join, realpath, dirname
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

images_dir = join(dirname(__file__), 'images')

class daeOptimizationProgressMonitor(QtWidgets.QDialog):
    def __init__(self):
        QtWidgets.QDialog.__init__(self)
        
        self.setWindowIcon(QtGui.QIcon(join(images_dir, 'daetools-48x48.png')))
        self.move(QtCore.QPoint(0, 50))
        self.resize(700, 400)  # Resize window
        self.setWindowTitle('Monitor the optimization progress')
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowSystemMenuHint
                                               | QtCore.Qt.WindowMinMaxButtonsHint)

        self.verticalLayout = QtWidgets.QVBoxLayout(self)

        self.figure = Figure((7.0, 4.0), dpi=100, facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)

        self.verticalLayout.addWidget(self.canvas)

        self.fp = matplotlib.font_manager.FontProperties(family='sans-serif', style='normal',
                                                         variant='normal', weight='normal', size=8)
        self.subplots = []

    def addSubplot(self, n, m, p, y_label):
        subplot = self.figure.add_subplot(n, m, p)

        self.subplots.append(subplot)

        # Add an empty curve
        line, = subplot.plot([], [])

        subplot.set_xlabel('',      fontproperties=self.fp)
        subplot.set_ylabel(y_label, fontproperties=self.fp)
        subplot.set_title('',       fontproperties=self.fp)

        for xlabel in subplot.get_xticklabels():
            xlabel.set_fontproperties(self.fp)
        for ylabel in subplot.get_yticklabels():
            ylabel.set_fontproperties(self.fp)

        subplot.grid(True)

        return subplot, line

    def addIteration(self, subplot, line, y):
        x_data = list(line.get_xdata())
        x_data.append(len(x_data))
        line.set_xdata(x_data)

        y_data = list(line.get_ydata())
        y_data.append(y)
        line.set_ydata(y_data)

        subplot.relim()
        subplot.autoscale_view()

    def redraw(self):
        self.canvas.draw()
