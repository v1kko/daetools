"""********************************************************************************
                            custom_plots.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2016
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
from PyQt4 import QtCore, QtGui
from daetools.pyDAE import *
from .custom_plots_ui import Ui_CustomPlots

images_dir = join(dirname(__file__), 'images')

class daeCustomPlots(QtGui.QDialog):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_CustomPlots()
        self.ui.setupUi(self)

        self.plot_source = ''

        self.setWindowIcon(QtGui.QIcon(join(images_dir, 'daetools-48x48.png')))

        self.connect(self.ui.buttonAddPlot,    QtCore.SIGNAL("clicked()"), self.slotAddPlot)
        self.connect(self.ui.buttonRemovePlot, QtCore.SIGNAL("clicked()"), self.slotRemovePlot)
        self.connect(self.ui.buttonMakePlot,   QtCore.SIGNAL("clicked()"), self.slotMakePlot)
        self.connect(self.ui.buttonSave,       QtCore.SIGNAL("clicked()"), self.slotSaveSource)
        self.connect(self.ui.listPlots,        QtCore.SIGNAL("itemSelectionChanged()"), self.itemSelectionChanged)
        self.connect(self.ui.listPlots,        QtCore.SIGNAL("itemDoubleClicked(QListWidgetItem*)"), self.itemDoubleClicked)

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
                    item = QtGui.QListWidgetItem(name, self.ui.listPlots)
                    item.setData(QtCore.Qt.UserRole, source)
                    self.ui.listPlots.addItem(item)
                    if i == 0:
                        self.ui.listPlots.setItemSelected(item, True)
            else:
                name = 'Simple 2D plot'
                source = '''import matplotlib.pyplot
# User-defined plots must be specified in the make_custom_plot() function
def make_custom_plot(processes):
    # Common code to get the data for the plot
    cv_dlg = daeChooseVariable(daeChooseVariable.plot2D)
    cv_dlg.updateProcessesList(processes)
    cv_dlg.setWindowTitle('Choose variable for a user-defined 2D plot')
    if cv_dlg.exec_() != QtGui.QDialog.Accepted:
        return
    variable, domainIndexes, domainPoints, xAxisLabel, yAxisLabel, xPoints, yPoints, currentTime = cv_dlg.getPlot2DData()

    # User-defined part
    matplotlib.pyplot.plot(xPoints, yPoints)
    matplotlib.pyplot.show()'''
                item = QtGui.QListWidgetItem(name, self.ui.listPlots)
                item.setData(QtCore.Qt.UserRole, source)
                self.ui.listPlots.addItem(item)
                self.ui.listPlots.setItemSelected(item, True)

    def reject(self):
        self.slotSaveSource()
        QtGui.QDialog.reject(self)

    def slotAddPlot(self):
        name, ok = QtGui.QInputDialog.getText(self, 'Insert the name of the plot', 'Plot name:', QtGui.QLineEdit.Normal, '')
        if ok:
            item = QtGui.QListWidgetItem(name, self.ui.listPlots)
            item.setData(QtCore.Qt.UserRole, 'def plot(processes):')
            self.ui.listPlots.addItem(item)
            self.ui.listPlots.setItemSelected(item, True)

    def slotRemovePlot(self):
        items = self.ui.listPlots.selectedItems()
        del_item = self.ui.listPlots.takeItem(self.ui.listPlots.row(items[0]))
        del_item = None
        self.ui.editSourceCode.setPlainText('')

    def slotMakePlot(self):
        self.plot_source = str(self.ui.editSourceCode.toPlainText())
        self.slotSaveSource()
        self.accept()

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
        name, ok = QtGui.QInputDialog.getText(self, 'Set the name of the plot', 'Plot name:', QtGui.QLineEdit.Normal, name)
        if ok:
            item.setText(name)
