"""********************************************************************************
                            user_data.py
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
import sys, numpy
from os.path import join, realpath, dirname
from PyQt4 import QtCore, QtGui
from daetools.pyDAE import *
from .user_data_ui import Ui_UserData

images_dir = join(dirname(__file__), 'images')

class daeUserData(QtGui.QDialog):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_UserData()
        self.ui.setupUi(self)

        self.xPoints   = numpy.array([])
        self.yPoints   = numpy.array([])
        self.xLabel    = ''
        self.yLabel    = ''
        self.lineLabel = ''

        self.setWindowIcon(QtGui.QIcon(join(images_dir, 'daetools-48x48.png')))

        self.ui.buttonBox.accepted.connect(self.accept)
        self.ui.buttonBox.rejected.connect(self.reject)

    def accept(self):
        self.xPoints   = numpy.fromstring(str(self.ui.editXdata.toPlainText()), dtype=float, sep=',')
        self.yPoints   = numpy.fromstring(str(self.ui.editYdata.toPlainText()), dtype=float, sep=',')
        self.xLabel    = str(self.ui.editXlabel.text())
        self.yLabel    = str(self.ui.editYlabel.text())
        self.lineLabel = str(self.ui.editLineLabel.text())

        if not self.xLabel:
            QtGui.QMessageBox.warning(self, "User-data", "X axes label is empty")
            return
        if not self.yLabel:
            QtGui.QMessageBox.warning(self, "User-data", "Y axes label is empty")
            return
        if not self.lineLabel:
            QtGui.QMessageBox.warning(self, "User-data", "Line label is empty")
            return
        if self.xPoints.ndim != 1 or self.xPoints.size == 0:
            QtGui.QMessageBox.warning(self, "User-data", "Invalid x data array")
            return
        if self.yPoints.ndim != 1 or self.yPoints.size == 0:
            QtGui.QMessageBox.warning(self, "User-data", "Invalid x data array")
            return
        if self.xPoints.size != self.yPoints.size:
            QtGui.QMessageBox.warning(self, "User-data", "The size of x and y data arrays does not match")
            return

        return QtGui.QDialog.accept(self)

