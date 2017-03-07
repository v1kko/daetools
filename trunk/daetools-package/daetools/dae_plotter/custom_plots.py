"""********************************************************************************
                            custom_plots.py
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
import os, sys, numpy, pickle
from os.path import join, realpath, dirname
from PyQt5 import QtCore, QtGui, QtWidgets
from daetools.pyDAE import *
from .custom_plots_ui import Ui_CustomPlots

images_dir = join(dirname(__file__), 'images')

class daeCustomPlots(QtWidgets.QDialog):
    def __init__(self):
        QtWidgets.QDialog.__init__(self)
        self.ui = Ui_CustomPlots()
        self.ui.setupUi(self)

        self.plot_source = ''

        self.setWindowIcon(QtGui.QIcon(join(images_dir, 'daetools-48x48.png')))

        self.ui.buttonAddPlot.clicked.connect(self.slotAddPlot)
        self.ui.buttonRemovePlot.clicked.connect(self.slotRemovePlot)
        self.ui.buttonMakePlot.clicked.connect(self.slotMakePlot)
        self.ui.buttonSave.clicked.connect(self.slotSaveSource)
        self.ui.listPlots.itemSelectionChanged.connect(self.itemSelectionChanged)
        self.ui.listPlots.itemDoubleClicked.connect(self.itemDoubleClicked)

        self.currentItem = None

        homepath = os.path.expanduser('~')
        daetools_cfg_path = os.path.join(homepath, '.daetools')
        self.custom_plots_path = os.path.join(daetools_cfg_path, 'custom_plots.pickle')
        if not os.path.exists(daetools_cfg_path):
            os.mkdir(daetools_cfg_path)
        else:
            if os.path.exists(self.custom_plots_path):
                cp = open(self.custom_plots_path, 'rb')
                plots = pickle.load(cp)
                for i, (name, source) in enumerate(plots):
                    item = QtWidgets.QListWidgetItem(name, self.ui.listPlots)
                    item.setData(QtCore.Qt.UserRole, source)
                    self.ui.listPlots.addItem(item)
                    if i == 0:
                        item.setSelected(True)
            else:
                name = 'Simple 2D plot'
                source = '''import matplotlib.pyplot
# User-defined plots must be specified in the make_custom_plot() function
def make_custom_plot(processes):
    # Common code to get the data for the plot
    # If other type of data is required use daeChooseVariable.plot3D or daeChooseVariable.plot2DAnimated
    cv_dlg = daeChooseVariable(daeChooseVariable.plot2D)
    cv_dlg.updateProcessesList(processes)
    cv_dlg.setWindowTitle('Choose variable for a user-defined 2D plot')
    if cv_dlg.exec_() != QtWidgets.QDialog.Accepted:
        return
    variable, domainIndexes, domainPoints, xAxisLabel, yAxisLabel, xPoints, yPoints, currentTime = cv_dlg.getPlot2DData()

    # User-defined part
    matplotlib.pyplot.plot(xPoints, yPoints)
    matplotlib.pyplot.show()'''
                item = QtWidgets.QListWidgetItem(name, self.ui.listPlots)
                item.setData(QtCore.Qt.UserRole, source)
                item.setSelected(True)
                self.ui.listPlots.addItem(item)

    def reject(self):
        self.slotSaveSource()
        QtWidgets.QDialog.reject(self)

    def slotAddPlot(self):
        name, ok = QtWidgets.QInputDialog.getText(self, 'Insert the name of the plot', 'Plot name:', QtWidgets.QLineEdit.Normal, '')
        if ok:
            item = QtWidgets.QListWidgetItem(name, self.ui.listPlots)
            item.setData(QtCore.Qt.UserRole, 'def plot(processes):')
            item.setSelected(True)
            self.ui.listPlots.addItem(item)

    def slotRemovePlot(self):
        items = self.ui.listPlots.selectedItems()
        del_item = self.ui.listPlots.takeItem(self.ui.listPlots.row(items[0]))
        del_item = None
        self.ui.editSourceCode.setPlainText('')

    def slotMakePlot(self):
        self.plot_source = str(self.ui.editSourceCode.toPlainText())
        self.slotSaveSource()
        return self.accept()

    def slotSaveSource(self):
        # First save the current source into the list item
        items = self.ui.listPlots.selectedItems()
        if not items:
            return
        item = items[0]
        source = str(self.ui.editSourceCode.toPlainText())
        item.setData(QtCore.Qt.UserRole, source)

        # Now pickle all items
        items = []
        for i in range(len(self.ui.listPlots)):
            item = self.ui.listPlots.item(i)
            name   = str(item.text())
            data = item.data(QtCore.Qt.UserRole)
            if isinstance(data, QtCore.QVariant):
                source = str(data.toString())
            else:
                source = str(data)
            items.append((name, source))
        print(items)
        cp = open(self.custom_plots_path, 'wb')
        pickle.dump(items, cp)

    def itemSelectionChanged(self):
        # Save the source string from the text edit into the listwidgetitem
        if self.currentItem:
            source = str(self.ui.editSourceCode.toPlainText())
            self.currentItem.setData(QtCore.Qt.UserRole, source)

        items = self.ui.listPlots.selectedItems()
        if not items:
            self.currentItem = None
            return

        item = items[0]
        data = item.data(QtCore.Qt.UserRole)
        if isinstance(data, QtCore.QVariant):
            source = str(data.toString())
        else:
            source = str(data)
        self.ui.editSourceCode.setPlainText(source)
        self.currentItem = item

    def itemDoubleClicked(self, item):
        name   = str(item.text())
        name, ok = QtWidgets.QInputDialog.getText(self, 'Set the name of the plot', 'Plot name:', QtWidgets.QLineEdit.Normal, name)
        if ok:
            item.setText(name)
