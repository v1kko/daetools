from PyQt4 import QtCore, QtGui
from about_ui import Ui_About
from daetools.pyDAE import *

class AboutDialog(QtGui.QDialog):
    def __init__(self, process):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_About()
        self.ui.setupUi(self)
        self.setWindowTitle("DAE Tools Project v" + daeVersion(True))
