"""********************************************************************************
                            plot2d.py
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
import os, sys, numpy, json
from os.path import join, realpath, dirname
from PyQt5 import QtCore, QtGui, QtWidgets

# Matplotlib imports
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams.update({'figure.autolayout': True})
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.animation as animation

# daetools imports
from daetools.pyDAE import *
from .choose_variable import daeChooseVariable, daeTableDialog
from .animation_parameters import daeAnimationParameters
from .save_video import daeSavePlot2DVideo
from .user_data import daeUserData
#from .plot_options import figure_edit, surface_edit

images_dir = join(dirname(__file__), 'images')

class daePlot2dDefaults:
    def __init__(self, color='black', linewidth=0.5, linestyle='solid', marker='o', markersize=6, markerfacecolor='black', markeredgecolor='black'):
        self.color = color
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.marker = marker
        self.markersize = markersize
        self.markerfacecolor = markerfacecolor
        self.markeredgecolor = markeredgecolor

    def to_dict(self):
        d = {}
        d['color']           = self.color
        d['linewidth']       = self.linewidth
        d['linestyle']       = self.linestyle
        d['marker']          = self.marker
        d['markersize']      = self.markersize
        d['markerfacecolor'] = self.markerfacecolor
        d['markeredgecolor'] = self.markeredgecolor
        return d

    @classmethod
    def from_dict(cls, d):
        pd = daePlot2dDefaults()
        pd.color            = d['color']
        pd.linewidth        = float(d['linewidth'])
        pd.linestyle        = d['linestyle']
        pd.marker           = d['marker']
        pd.markersize       = int(d['markersize'])
        pd.markerfacecolor  = d['markerfacecolor']
        pd.markeredgecolor  = d['markeredgecolor']
        return pd

class dae2DPlot(QtWidgets.QDialog):
    plotDefaults = [daePlot2dDefaults('black', 0.5, 'solid', 'o', 6, 'black', 'black'),
                    daePlot2dDefaults('blue',  0.5, 'solid', 's', 6, 'blue',  'black'),
                    daePlot2dDefaults('red',   0.5, 'solid', '^', 6, 'red',   'black'),
                    daePlot2dDefaults('green', 0.5, 'solid', 'p', 6, 'green', 'black'),
                    daePlot2dDefaults('c',     0.5, 'solid', 'h', 6, 'c',     'black'),
                    daePlot2dDefaults('m',     0.5, 'solid', '*', 6, 'm',     'black'),
                    daePlot2dDefaults('k',     0.5, 'solid', 'd', 6, 'k',     'black'),
                    daePlot2dDefaults('y',     0.5, 'solid', 'x', 6, 'y',     'black'),

                    daePlot2dDefaults('black', 0.5, 'dashed', 'o', 6, 'black', 'black'),
                    daePlot2dDefaults('blue',  0.5, 'dashed', 's', 6, 'blue',  'black'),
                    daePlot2dDefaults('red',   0.5, 'dashed', '^', 6, 'red',   'black'),
                    daePlot2dDefaults('green', 0.5, 'dashed', 'p', 6, 'green', 'black'),
                    daePlot2dDefaults('c',     0.5, 'dashed', 'h', 6, 'c',     'black'),
                    daePlot2dDefaults('m',     0.5, 'dashed', '*', 6, 'm',     'black'),
                    daePlot2dDefaults('k',     0.5, 'dashed', 'd', 6, 'k',     'black'),
                    daePlot2dDefaults('y',     0.5, 'dashed', 'x', 6, 'y',     'black'),

                    daePlot2dDefaults('black', 0.5, 'dotted', 'o', 6, 'black', 'black'),
                    daePlot2dDefaults('blue',  0.5, 'dotted', 's', 6, 'blue',  'black'),
                    daePlot2dDefaults('red',   0.5, 'dotted', '^', 6, 'red',   'black'),
                    daePlot2dDefaults('green', 0.5, 'dotted', 'p', 6, 'green', 'black'),
                    daePlot2dDefaults('c',     0.5, 'dotted', 'h', 6, 'c',     'black'),
                    daePlot2dDefaults('m',     0.5, 'dotted', '*', 6, 'm',     'black'),
                    daePlot2dDefaults('k',     0.5, 'dotted', 'd', 6, 'k',     'black'),
                    daePlot2dDefaults('y',     0.5, 'dotted', 'x', 6, 'y',     'black') ]

    def __init__(self, parent, updateInterval = 0, animated = False):
        QtWidgets.QDialog.__init__(self, parent, QtCore.Qt.Window)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.plotter = parent

        self.legendOn = True
        self.gridOn   = True
        self.curves   = []
        self.funcAnimation = None
        self._isAnimating  = False
        self._timer        = None
        self._cv_dlg       = None
        self.xmin_policy = 0
        self.xmax_policy = 0
        self.ymin_policy = 1
        self.ymax_policy = 1

        if animated == True:
            self.updateInterval = updateInterval
            self.plotType = daeChooseVariable.plot2DAnimated
        elif updateInterval == 0:
            self.updateInterval = 0
            self.plotType = daeChooseVariable.plot2D
        else:
            self.updateInterval = int(updateInterval)
            self.plotType = daeChooseVariable.plot2DAutoUpdated

        self.setWindowTitle("2D plot")
        self.setWindowIcon(QtGui.QIcon(join(images_dir, 'line-chart.png')))

        exit = QtWidgets.QAction(QtGui.QIcon(join(images_dir, 'close.png')), 'Exit', self)
        exit.setShortcut('Ctrl+Q')
        exit.setStatusTip('Exit application')
        exit.triggered.connect(self.close)

        export = QtWidgets.QAction(QtGui.QIcon(join(images_dir, 'template.png')), 'Export template', self)
        export.setShortcut('Ctrl+X')
        export.setStatusTip('Export template')
        export.triggered.connect(self.slotExportTemplate)

        properties = QtWidgets.QAction(QtGui.QIcon(join(images_dir, 'preferences.png')), 'Options', self)
        properties.setShortcut('Ctrl+P')
        properties.setStatusTip('Options')
        properties.triggered.connect(self.slotProperties)

        grid = QtWidgets.QAction(QtGui.QIcon(join(images_dir, 'grid.png')), 'Grid on/off', self)
        grid.setShortcut('Ctrl+G')
        grid.setStatusTip('Grid on/off')
        grid.triggered.connect(self.slotToggleGrid)

        legend = QtWidgets.QAction(QtGui.QIcon(join(images_dir, 'legend.png')), 'Legend on/off', self)
        legend.setShortcut('Ctrl+L')
        legend.setStatusTip('Legend on/off')
        legend.triggered.connect(self.slotToggleLegend)

        viewdata = QtWidgets.QAction(QtGui.QIcon(join(images_dir, 'data.png')), 'View tabular data', self)
        viewdata.setShortcut('Ctrl+T')
        viewdata.setStatusTip('View tabular data')
        viewdata.triggered.connect(self.slotViewTabularData)

        export_csv = QtWidgets.QAction(QtGui.QIcon(join(images_dir, 'csv.png')), 'Export CSV', self)
        export_csv.setShortcut('Ctrl+S')
        export_csv.setStatusTip('Export CSV')
        export_csv.triggered.connect(self.slotExportCSV)

        fromUserData = QtWidgets.QAction(QtGui.QIcon(join(images_dir, 'add-user-data.png')), 'Add line from the user-provided data...', self)
        fromUserData.setShortcut('Ctrl+D')
        fromUserData.setStatusTip('Add line from the user-provided data')
        fromUserData.triggered.connect(self.slotFromUserData)

        remove_line = QtWidgets.QAction(QtGui.QIcon(join(images_dir, 'remove.png')), 'Remove line', self)
        remove_line.setShortcut('Ctrl+R')
        remove_line.setStatusTip('Remove line')
        remove_line.triggered.connect(self.slotRemoveLine)

        new_line = QtWidgets.QAction(QtGui.QIcon(join(images_dir, 'add.png')), 'Add line', self)
        new_line.setShortcut('Ctrl+A')
        new_line.setStatusTip('Add line')
        new_line.triggered.connect(self.newCurve)

        play_animation = QtWidgets.QAction(QtGui.QIcon(join(images_dir, 'media-playback-start.png')), 'Start animation', self)
        play_animation.setShortcut('Ctrl+S')
        play_animation.setStatusTip('Start animation')
        play_animation.triggered.connect(self.playAnimation)
        self.play_animation = play_animation # save it

        stop_animation = QtWidgets.QAction(QtGui.QIcon(join(images_dir, 'media-playback-stop.png')), 'Stop animation', self)
        stop_animation.setShortcut('Ctrl+E')
        stop_animation.setStatusTip('Stop animation')
        stop_animation.triggered.connect(self.stopAnimation)
        self.stop_animation = stop_animation # save it

        export_video = QtWidgets.QAction(QtGui.QIcon(join(images_dir, 'save-video.png')), 'Export video/sequence of images', self)
        export_video.setShortcut('Ctrl+V')
        export_video.setStatusTip('Export video/sequence of images')
        export_video.triggered.connect(self.exportVideo)

        self.actions_to_disable = [export, viewdata, export_csv, grid, legend, properties]
        self.actions_to_disable_permanently = [new_line, fromUserData, remove_line]

        self.toolbar_widget = QtWidgets.QWidget(self)
        layoutToolbar = QtWidgets.QVBoxLayout(self.toolbar_widget)
        layoutToolbar.setContentsMargins(0,0,0,0)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.toolbar_widget.setSizePolicy(sizePolicy)

        layoutPlot = QtWidgets.QVBoxLayout(self)
        layoutPlot.setContentsMargins(2,2,2,2)
        self.figure = Figure((8, 6.5), dpi=100, facecolor='white')#"#E5E5E5")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        self.canvas.axes = self.figure.add_subplot(111)

        self.mpl_toolbar = NavigationToolbar(self.canvas, self.toolbar_widget, False)

        #self.mpl_toolbar.addSeparator()
        self.mpl_toolbar.addAction(export)
        self.mpl_toolbar.addAction(export_csv)
        self.mpl_toolbar.addAction(viewdata)
        self.mpl_toolbar.addSeparator()
        self.mpl_toolbar.addAction(grid)
        self.mpl_toolbar.addAction(legend)
        self.mpl_toolbar.addSeparator()
        self.mpl_toolbar.addAction(new_line)
        self.mpl_toolbar.addAction(fromUserData)
        self.mpl_toolbar.addAction(remove_line)
        self.mpl_toolbar.addSeparator()
        #self.mpl_toolbar.addAction(properties)
        #self.mpl_toolbar.addSeparator()
        #self.mpl_toolbar.addAction(exit)
        if self.plotType == daeChooseVariable.plot2DAnimated:
            self.mpl_toolbar.addSeparator()
            self.mpl_toolbar.addAction(play_animation)
            self.mpl_toolbar.addAction(stop_animation)
            self.mpl_toolbar.addAction(export_video)

        self.fp9  = matplotlib.font_manager.FontProperties(family='sans-serif', style='normal', variant='normal', weight='normal', size=9)
        self.fp10 = matplotlib.font_manager.FontProperties(family='sans-serif', style='normal', variant='normal', weight='normal', size=10)
        self.fp12 = matplotlib.font_manager.FontProperties(family='sans-serif', style='normal', variant='normal', weight='normal', size=12)

        self.textTime = self.figure.text(0.01, 0.01, '', fontproperties = self.fp10)

        self.xtransform = 1.0
        self.ytransform = 1.0

        for xlabel in self.canvas.axes.get_xticklabels():
            xlabel.set_fontproperties(self.fp10)
        for ylabel in self.canvas.axes.get_yticklabels():
            ylabel.set_fontproperties(self.fp10)

        layoutToolbar.addWidget(self.mpl_toolbar)
        layoutPlot.addWidget(self.canvas)
        layoutPlot.addWidget(self.toolbar_widget)

        if animated == False and self.updateInterval > 0:
            self._timer = QtCore.QTimer()
            self._timer.timeout.connect(self.updateCurves)
            self._timer.start(self.updateInterval)
        
    def closeEvent(self, event):
        #print("dae2DPlot.closeEvent")
        if self.funcAnimation:
            self.funcAnimation.event_source.stop()

        if self._timer:
            self._timer.stop()

        return QtWidgets.QDialog.closeEvent(self, event)

    def updateCurves(self):
        try:
            #                                                      these three not used
            for (line, variable, domainIndexes, domainPoints, fun, times, xPoints, yPoints_2D) in self.curves:
                results = fun(variable, domainIndexes, domainPoints)
                if self.xtransform != 1:
                    xPoints = numpy.array(results[5])*self.xtransform
                else:
                    xPoints = results[5]

                if self.ytransform != 1:
                    yPoints = numpy.array(results[6])*self.ytransform
                else:
                    yPoints = results[6]

                currentTime = results[7]

                line.set_xdata(xPoints)
                line.set_ydata(yPoints)

                if self.textTime:
                    t = 'Time = {0} s'.format(currentTime)
                    self.textTime.set_text(t)

            #self.reformatPlot()

        except Exception as e:
            print((str(e)))

    #@QtCore.pyqtSlot()
    def slotExportTemplate(self):
        try:
            curves = []
            template = {'curves':          curves,
                        'plotType':        self.plotType,
                        'updateInterval' : self.updateInterval,
                        'xlabel' :         self.canvas.axes.get_xlabel(),
                        'xmin' :           self.canvas.axes.get_xlim()[0],
                        'xmax' :           self.canvas.axes.get_xlim()[1],
                        'xscale' :         self.canvas.axes.get_xscale(),
                        'xtransform':      1.0,
                        'ylabel' :         self.canvas.axes.get_ylabel(),
                        'ymin' :           self.canvas.axes.get_ylim()[0],
                        'ymax' :           self.canvas.axes.get_ylim()[1],
                        'yscale' :         self.canvas.axes.get_yscale(),
                        'ytransform':      1.0,
                        'legendOn' :       self.legendOn,
                        'gridOn' :         self.gridOn,
                        'plotTitle' :      self.canvas.axes.get_title(),
                        'windowTitle':     str(self.windowTitle()),
                        'xmin_policy':     int(self.xmin_policy),
                        'xmax_policy':     int(self.xmax_policy),
                        'ymin_policy':     int(self.ymin_policy),
                        'ymax_policy':     int(self.ymax_policy)
                       }

            for (line, variable, domainIndexes, domainPoints, fun, times, xPoints, yPoints_2D) in self.curves:
                # variableName, indexes, points, linelabel, style = {linecolor, linewidth, linestyle, marker, markersize, markerfacecolor, markeredgecolor}
                style = daePlot2dDefaults(line.get_color(), line.get_linewidth(), line.get_linestyle(),
                                          line.get_marker(), line.get_markersize(), line.get_markerfacecolor(), line.get_markeredgecolor())
                curves.append((variable.Name, domainIndexes, domainPoints, line.get_label(), style.to_dict()))

            s = json.dumps(template, indent=2, sort_keys=True)

            filename, ok = QtWidgets.QFileDialog.getSaveFileName(self, "Save 2D plot template", "template.pt", "Templates (*.pt)")
            if not ok:
                return

            f = open(filename, 'w')
            f.write(s)
            f.close()

        except Exception as e:
            print((str(e)))

    #@QtCore.pyqtSlot()
    def slotProperties(self):
        figure_edit(self.canvas, self)

    #@QtCore.pyqtSlot()
    def slotToggleLegend(self):
        self.legendOn = not self.legendOn
        self.updateLegend()
        
    #@QtCore.pyqtSlot()
    def slotToggleGrid(self):
        self.gridOn = not self.gridOn
        self.updateGrid()
        
    def updateLegend(self):
        if self.legendOn:
            self.canvas.axes.legend(loc = 0, prop=self.fp9, numpoints = 1, fancybox=True)
        else:
            self.canvas.axes.legend_ = None
        self.canvas.draw()
        #self.reformatPlot()

    def updateGrid(self):
        self.canvas.axes.grid(self.gridOn)
        self.canvas.draw()
        #self.reformatPlot()

    #@QtCore.pyqtSlot()
    def slotExportCSV(self):
        strInitialFilename = QtCore.QDir.current().path()
        strInitialFilename += "/untitled.csv";
        strExt = "Comma separated files (*.csv)"
        strCaption = "Save file"
        fileName, ok = QtWidgets.QFileDialog.getSaveFileName(self, strCaption, strInitialFilename, strExt)
        if not ok:
            return

        datafile = open(str(fileName), 'w')
        lines = self.canvas.axes.get_lines()

        for line in lines:
            xlabel = self.canvas.axes.get_xlabel()
            ylabel = line.get_label()
            x = line.get_xdata()
            y = line.get_ydata()
            datafile.write('\"' + xlabel + '\",\"' + ylabel + '\"\n' )
            for i in range(0, len(x)):
                datafile.write('%.14e,%.14e\n' % (x[i], y[i]))
                #datafile.write(str(x[i]) + ',' + str(y[i]) + '\n')
            datafile.write('\n')

    #@QtCore.pyqtSlot()
    def slotViewTabularData(self):
        lines = self.canvas.axes.get_lines()

        tableDialog = daeTableDialog(self)
        tableDialog.setWindowTitle('Raw data')
        table = tableDialog.ui.tableWidget
        nr = 0
        ncol = len(lines)
        for line in lines:
            n = len(line.get_xdata())
            if nr < n:
                nr = n

        xlabel = self.canvas.axes.get_xlabel()
        table.setRowCount(nr)
        table.setColumnCount(ncol)
        horHeader = []
        verHeader = []
        for i, line in enumerate(lines):
            xlabel = self.canvas.axes.get_xlabel()
            ylabel = line.get_label()
            x = line.get_xdata()
            y = line.get_ydata()
            horHeader.append(ylabel)
            for k in range(0, len(x)):
                newItem = QtWidgets.QTableWidgetItem(str(y[k]))
                table.setItem(k, i, newItem)
            for k in range(0, len(x)):
                verHeader.append(str(x[k]))

        table.setHorizontalHeaderLabels(horHeader)
        table.setVerticalHeaderLabels(verHeader)
        table.resizeRowsToContents()
        tableDialog.exec_()

    #@QtCore.pyqtSlot()
    def slotRemoveLine(self):
        lines = self.canvas.axes.get_lines()
        items = []
        for line in lines:
            label = line.get_label()
            items.append(label)

        nameToRemove, ok = QtWidgets.QInputDialog.getItem(self, "Choose line to remove", "Lines:", items, 0, False)
        if ok:
            for i, line in enumerate(lines):
                label = line.get_label()
                if label == str(nameToRemove):
                    self.canvas.axes.lines.pop(i)
                    #self.reformatPlot()
                    # updateLegend will also call canvas.draw()
                    self.updateLegend()
                    return

    def newFromTemplate(self, template):
        """
        template is a dictionary:

        .. code-block:: javascript

           {
               'curves' : [variableName, domainIndexes, domainPoints, lineTitle, style],
               'updateInterval' : float,
               'xlabel' : string,
               'xmin' : float,
               'xmax' : float,
               'xscale' : string [linear, log],
               'xtransform': float,
               'ylabel' : string,
               'ymin' : float,
               'ymax' : float,
               'yscale' : string [linear, log],
               'ytransform': float,
               'legendOn' : Bool,
               'gridOn' : Bool,
               'plotTitle' : string,
               'windowTitle' : string
               'xmin_policy': int,
               'xmax_policy': int,
               'ymin_policy': int,
               'ymax_policy': int
           }
        """
        processes = {}
        for process in self.plotter.getProcesses():
            processes[process.Name] = process

        if len(processes) == 0:
            return

        if len(template) == 0:
            return False

        curves = template['curves']

        if 'plotType' in template:
            self.plotType = int(template['plotType'])

        if 'xtransform' in template:
            self.xtransform = template['xtransform']
        if 'ytransform' in template:
            self.ytransform = template['ytransform']

        for i, curve in enumerate(curves):
            variableName  = curve[0]
            domainIndexes = curve[1]
            domainPoints  = curve[2]
            label         = None
            pd            = None
            if len(curve) > 3:
                label = curve[3]
            if len(curve) > 4:
                pd = daePlot2dDefaults.from_dict(curve[4])

            windowTitle     = "Select process for variable {0} (of {1})".format(i+1, len(curves))
            var_to_look_for = "Variable: {0}({1})".format(variableName, ','.join(domainPoints))
            items       = sorted(processes.keys())
            processName, ok = self.showSelectProcessDialog(windowTitle, var_to_look_for, items)
            if not ok:
                return False

            process = processes[str(processName)]
            for variable in process.Variables:
                if variableName == variable.Name:
                    if self.plotType == daeChooseVariable.plot2D or self.plotType == daeChooseVariable.plot2DAutoUpdated:
                        variable, domainIndexes, domainPoints, xAxisLabel, yAxisLabel, xPoints, yPoints, currentTime = daeChooseVariable.get2DData(variable, domainIndexes, domainPoints)
                        self._addNewCurve(variable, domainIndexes, domainPoints, xAxisLabel, yAxisLabel, xPoints, yPoints, currentTime, label, pd)
                        break

                    elif self.plotType == daeChooseVariable.plot2DAnimated:
                        variable, domainIndexes, domainPoints, xAxisLabel, yAxisLabel, xPoints, yPoints, times = daeChooseVariable.get2DAnimatedData(variable, domainIndexes, domainPoints)
                        self._addNewAnimatedCurve(variable, domainIndexes, domainPoints, xAxisLabel, yAxisLabel, xPoints, yPoints, times, None, None)
                        for action in self.actions_to_disable_permanently:
                            action.setEnabled(False)
                        break

                    else:
                        raise RuntimeError('Invalid plot type')

        if 'xlabel' in template:
            self.canvas.axes.set_xlabel(template['xlabel'], fontproperties=self.fp12)
        if 'xmin' in template and 'xmax' in template:
            self.canvas.axes.set_xlim(float(template['xmin']), float(template['xmax']))
        if 'xscale' in template:
            self.canvas.axes.set_xscale(template['xscale'])

        if 'ylabel' in template:
            self.canvas.axes.set_ylabel(template['ylabel'], fontproperties=self.fp12)
        if 'ymin' in template and 'ymax' in template:
            self.canvas.axes.set_ylim(float(template['ymin']), float(template['ymax']))
        if 'yscale' in template:
            self.canvas.axes.set_yscale(template['yscale'])

        if 'gridOn' in template:
            self.gridOn = template['gridOn']
            self.canvas.axes.grid(self.gridOn)

        if 'legendOn' in template:
            self.legendOn = template['legendOn']
            if self.legendOn:
                self.canvas.axes.legend(loc = 0, prop=self.fp9, numpoints = 1, fancybox=True)
            else:
                self.canvas.axes.legend_ = None

        if 'plotTitle' in template:
            self.canvas.axes.set_title(template['plotTitle'])

        if 'windowTitle' in template:
            self.setWindowTitle(template['windowTitle'])

        if 'xmin_policy' in template:
            self.xmin_policy = int(template['xmin_policy'])
        if 'xmax_policy' in template:
            self.xmax_policy = int(template['xmax_policy'])
        if 'ymin_policy' in template:
            self.ymin_policy = int(template['ymin_policy'])
        if 'ymax_policy' in template:
            self.ymax_policy = int(template['ymax_policy'])

        #fmt = matplotlib.ticker.ScalarFormatter(useOffset = False)
        #fmt.set_scientific(False)
        #fmt.set_powerlimits((-3, 4))
        #self.canvas.axes.xaxis.set_major_formatter(fmt)
        #self.canvas.axes.yaxis.set_major_formatter(fmt)

        self.figure.tight_layout()
        self.canvas.draw()

        return True

    def showSelectProcessDialog(self, windowTitle, label, items):
        dlg = QtWidgets.QInputDialog(self)
        dlg.resize(500, 300)
        dlg.setWindowTitle(windowTitle)
        dlg.setLabelText(label)
        dlg.setComboBoxItems(items)

        dlg.setComboBoxEditable(False)
        dlg.setOption(QtWidgets.QInputDialog.UseListViewForComboBoxItems)

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            return str(dlg.textValue()), True
        else:
            return '', False

    def _updateFrame(self, frame):
        curve = self.curves[0]
        line    = curve[0]
        times   = curve[5]
        xPoints = curve[6]
        yPoints = curve[7]
        yData = yPoints[frame]
        line.set_ydata(yData)
        time = times[frame]

        if self.xmin_policy == 0:   # From 1st frame
            xmin = numpy.min(xPoints)
        elif self.xmin_policy == 1: # Overall min value
            xmin = numpy.min(xPoints)
        elif self.xmin_policy == 2: # Adaptive
            xmin = numpy.min(xPoints)
        else:                       # Do not change it
            xmin = self.canvas.axes.get_xlim()[0]

        if self.xmax_policy == 0:   # From 1st frame
            xmax = numpy.max(xPoints)
            dx = 0.5 * (xmax-xmin)*0.05
        elif self.xmax_policy == 1: # Overall max value
            xmax = numpy.max(xPoints)
            dx = 0.5 * (xmax-xmin)*0.05
        elif self.xmax_policy == 2: # Adaptive
            xmax = numpy.max(xPoints)
            dx = 0.5 * (xmax-xmin)*0.05
        else:                       # Do not change it
            xmax = self.canvas.axes.get_xlim()[1]
            dx = 0.0

        if self.ymin_policy == 0:   # From 1st frame
            ymin = numpy.min(yPoints[0])
        elif self.ymin_policy == 1: # Overall min value
            ymin = numpy.min(yPoints)
        elif self.ymin_policy == 2: # Adaptive
            ymin = numpy.min(yPoints[frame])
        else:                       # Do not change it
            ymin = self.canvas.axes.get_ylim()[0]

        if self.ymax_policy == 0:   # From 1st frame
            ymax = numpy.max(yPoints[0])
            dy = 0.5 * (ymax-ymin)*0.05
        elif self.ymax_policy == 1: # Overall max value
            ymax = numpy.max(yPoints)
            dy = 0.5 * (ymax-ymin)*0.05
        elif self.ymax_policy == 2: # Adaptive
            ymax = numpy.max(yPoints[frame])
            dy = 0.5 * (ymax-ymin)*0.05
        else:                       # Do not change it
            ymax = self.canvas.axes.get_ylim()[1]
            dy = 0.0

        self.canvas.axes.set_xlim(xmin-dx, xmax+dx)
        self.canvas.axes.set_ylim(ymin-dy, ymax+dy)

        self.canvas.axes.set_title('time = %f s' % time, fontproperties=self.fp10)

        if frame == len(times)-1: # the last frame
            for action in self.actions_to_disable:
                action.setEnabled(True)

            del self.funcAnimation
            self.funcAnimation = None

            self.play_animation.setIcon(QtGui.QIcon(join(images_dir, 'media-playback-start.png')))
            self.play_animation.setStatusTip('Start animation')
            self.play_animation.setText('Start animation')
            self._isAnimating = False

        return line,

    def _startAnimation(self):
        if len(self.curves) != 1:
            return

        # Set properties for the frame 0
        curve = self.curves[0]
        times  = curve[5]
        frames = numpy.arange(0, len(times))

        self.canvas.axes.set_title('time = %f s' % times[0], fontproperties=self.fp10)
        self.funcAnimation = animation.FuncAnimation(self.figure,
                                                     self._updateFrame,
                                                     frames,
                                                     interval=self.updateInterval,
                                                     blit=False,
                                                     repeat=False)
        self.play_animation.setIcon(QtGui.QIcon(join(images_dir, 'media-playback-pause.png')))
        self.play_animation.setStatusTip('Pause animation')
        self.play_animation.setText('Pause animation')
        self._isAnimating = True
        #At the end do not call show() nor save(), they will be ran by a caller

    #@QtCore.pyqtSlot()
    def playAnimation(self):
        if self.funcAnimation: # animation started - pause/resume it
            if self._isAnimating: # pause it
                for action in self.actions_to_disable:
                    action.setEnabled(True)
                self.funcAnimation.event_source.stop()
                self.play_animation.setIcon(QtGui.QIcon(join(images_dir, 'media-playback-start.png')))
                self.play_animation.setStatusTip('Start animation')
                self.play_animation.setText('Start animation')
                self._isAnimating = False
                self.canvas.draw()
            else: # restart it
                for action in self.actions_to_disable:
                    action.setEnabled(False)
                self.funcAnimation.event_source.start()
                self.play_animation.setIcon(QtGui.QIcon(join(images_dir, 'media-playback-pause.png')))
                self.play_animation.setStatusTip('Pause animation')
                self.play_animation.setText('Pause animation')
                self._isAnimating = True
                self.canvas.draw()

        else: # start animation
            for action in self.actions_to_disable:
                action.setEnabled(False)
            self._startAnimation()
            self.canvas.draw()

    #@QtCore.pyqtSlot()
    def stopAnimation(self):
        if self.funcAnimation: # animated started - stop it
            for action in self.actions_to_disable:
                action.setEnabled(True)

            self.funcAnimation.event_source.stop()
            del self.funcAnimation
            self.funcAnimation = None

            self.play_animation.setIcon(QtGui.QIcon(join(images_dir, 'media-playback-start.png')))
            self.play_animation.setStatusTip('Start animation')
            self.play_animation.setText('Start animation')
            self._isAnimating = False

        # Go back to frame 0
        self._updateFrame(0)
        self.canvas.draw()

    #@QtCore.pyqtSlot()
    def exportVideo(self):
        dlg = daeSavePlot2DVideo()
        for enc in sorted(animation.writers.list()):
            dlg.ui.comboEncoders.addItem(str(enc))
        dlg.ui.lineeditCodec.setText('')
        dlg.ui.lineeditFilename.setText(os.path.join(os.path.expanduser('~'), 'video.avi'))
        dlg.ui.spinFPS.setValue(10)
        dlg.ui.lineeditExtraArgs.setText(json.dumps([])) # ['-pix_fmt', 'yuv420p']
        dlg.ui.spinBitrate.setValue(-1)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return False

        filename   = str(dlg.ui.lineeditFilename.text())
        fps        = int(dlg.ui.spinFPS.value())
        encoder    = str(dlg.ui.comboEncoders.currentText())
        codec      = str(dlg.ui.lineeditCodec.text())
        bitrate    = int(dlg.ui.spinBitrate.value())
        extra_args = []
        try:
            extra_args = list(json.loads(str(dlg.ui.lineeditExtraArgs.text())))
        except:
            pass
        if bitrate == -1:
            bitrate = None
        if codec == '':
            codec = None
        if not extra_args:
            extra_args = None

        print('%s(fps = %s, codec = %s, bitrate = %s, extra_args = %s) -> %s' % (encoder, fps, codec, bitrate, extra_args, filename))

        # First stop the existing animation, if already started
        self.stopAnimation()
        Writer = animation.writers[encoder]
        writer = Writer(fps = fps, codec = codec, bitrate = bitrate, extra_args = extra_args)
        self._startAnimation()
        self.funcAnimation.save(filename, writer=writer)

    def slotFromUserData(self):
        dlg = daeUserData()
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        self.newCurveFromUserData(dlg.xLabel, dlg.yLabel, dlg.lineLabel, dlg.xPoints, dlg.yPoints)

    def newCurveFromUserData(self, xAxisLabel, yAxisLabel, lineLabel, xPoints, yPoints):
        class dummyVariable(object):
            def __init__(self, name = '', units = ''):
                self.Name  = name
                self.Units = units

        self._addNewCurve(dummyVariable(lineLabel), [], [], xAxisLabel, yAxisLabel, xPoints, yPoints, None, lineLabel, None)
        return True

    #@QtCore.pyqtSlot()
    def newCurve(self):
        processes = self.plotter.getProcesses()
        
        if not self._cv_dlg:
            self._cv_dlg = daeChooseVariable(self.plotType)
        self._cv_dlg.updateProcessesList(processes)
        self._cv_dlg.setWindowTitle('Choose variable for 2D plot')
        if self._cv_dlg.exec_() != QtWidgets.QDialog.Accepted:
            return False

        variable, domainIndexes, domainPoints, xAxisLabel, yAxisLabel, xPoints, yPoints, currentTime = self._cv_dlg.getPlot2DData()
        self._addNewCurve(variable, domainIndexes, domainPoints, xAxisLabel, yAxisLabel, xPoints, yPoints, currentTime, None, None)

        return True

    #@QtCore.pyqtSlot()
    def newAnimatedCurve(self):
        processes = self.plotter.getProcesses()

        if not self._cv_dlg:
            self._cv_dlg = daeChooseVariable(self.plotType)
        self._cv_dlg.updateProcessesList(processes)
        self._cv_dlg.setWindowTitle('Choose variable for animated 2D plot')
        if self._cv_dlg.exec_() != QtWidgets.QDialog.Accepted:
            return False

        dlg = daeAnimationParameters()
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        self.updateInterval = int(dlg.ui.spinUpdateInterval.value())
        self.xmin_policy    = dlg.ui.comboXmin.currentIndex()
        self.xmax_policy    = dlg.ui.comboXmax.currentIndex()
        self.ymin_policy    = dlg.ui.comboYmin.currentIndex()
        self.ymax_policy    = dlg.ui.comboYmax.currentIndex()

        variable, domainIndexes, domainPoints, xAxisLabel, yAxisLabel, xPoints, yPoints, times = self._cv_dlg.getPlot2DAnimatedData()
        self._addNewAnimatedCurve(variable, domainIndexes, domainPoints, xAxisLabel, yAxisLabel, xPoints, yPoints, times, None, None)

        for action in self.actions_to_disable_permanently:
            action.setEnabled(False)

        self._updateFrame(0)

        return True

    def _addNewAnimatedCurve(self, variable, domainIndexes, domainPoints, xAxisLabel, yAxisLabel, xPoints, yPoints_2D, times, label = None, pd = None):
        domains = '(' + ', '.join(domainPoints) + ')'

        if not label:
            label = variable.Name.replace("&", "").replace(";", "") + domains

        line = self.addLine(xAxisLabel, yAxisLabel, xPoints, yPoints_2D[0], label, pd)
        self.setWindowTitle(label)

        #                                                                 update fun is None
        self.curves.append( (line, variable, domainIndexes, domainPoints, None, times, xPoints, yPoints_2D) )

    def _addNewCurve(self, variable, domainIndexes, domainPoints, xAxisLabel, yAxisLabel, xPoints, yPoints, currentTime, label = None, pd = None):
        domains = "("
        for i in range(0, len(domainPoints)):
            if i != 0:
                domains += ", "
            domains += domainPoints[i]
        domains += ")"

        if not label:
            label = variable.Name.replace("&", "").replace(";", "") + domains

        line = self.addLine(xAxisLabel, yAxisLabel, xPoints, yPoints, label, pd)
        self.setWindowTitle(label)
        #                                                                                              everything after update fun is none
        self.curves.append( (line, variable, domainIndexes, domainPoints, daeChooseVariable.get2DData, None, None, None) )

    def addLine(self, xAxisLabel, yAxisLabel, xPoints, yPoints, label, pd):
        no_lines = len(self.canvas.axes.get_lines())
        if not pd:
            n = no_lines % len(dae2DPlot.plotDefaults)
            pd = dae2DPlot.plotDefaults[n]

        xPoints_ = numpy.array(xPoints) * self.xtransform
        yPoints_ = numpy.array(yPoints) * self.ytransform

        line, = self.canvas.axes.plot(xPoints_, yPoints_, label=label, color=pd.color, linewidth=pd.linewidth, \
                                      linestyle=pd.linestyle, marker=pd.marker, markersize=pd.markersize, \
                                      markerfacecolor=pd.markerfacecolor, markeredgecolor=pd.markeredgecolor)

        if no_lines == 0: 
            # Set labels, fonts, gridlines and limits only when adding the first line
            self.canvas.axes.set_xlabel(xAxisLabel, fontproperties=self.fp12)
            self.canvas.axes.set_ylabel(yAxisLabel, fontproperties=self.fp12)
            t = self.canvas.axes.xaxis.get_offset_text()
            t.set_fontproperties(self.fp10)
            t = self.canvas.axes.yaxis.get_offset_text()
            t.set_fontproperties(self.fp10)
            self.updateGrid()
            
        # Update the legend and (x,y) limits after every addition
        self.updateLegend()
        self.reformatPlot()

        return line

    def reformatPlot(self):
        lines = self.canvas.axes.get_lines()
        xmin = 1e20
        xmax = -1e20
        ymin = 1e20
        ymax = -1e20
        for line in lines:
            if numpy.min(line.get_xdata()) < xmin:
                xmin = numpy.min(line.get_xdata())
            if numpy.max(line.get_xdata()) > xmax:
                xmax = numpy.max(line.get_xdata())

            if numpy.min(line.get_ydata()) < ymin:
                ymin = numpy.min(line.get_ydata())
            if numpy.max(line.get_ydata()) > ymax:
                ymax = numpy.max(line.get_ydata())

        dx = (xmax - xmin) * 0.05
        dy = (ymax - ymin) * 0.05
        xmin -= dx
        xmax += dx
        ymin -= dy
        ymax += dy

        self.canvas.axes.set_xlim(xmin, xmax)
        self.canvas.axes.set_ylim(ymin, ymax)

        #self.canvas.axes.grid(self.gridOn)

        #if self.legendOn:
        #    self.canvas.axes.legend(loc = 0, prop=self.fp9, numpoints = 1, fancybox=True)
        #else:
        #    self.canvas.axes.legend_ = None

        self.figure.tight_layout()
        self.canvas.draw()
