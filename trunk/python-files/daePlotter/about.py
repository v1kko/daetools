from PyQt4 import QtCore, QtGui
from about_ui import Ui_About

class AboutDialog(QtGui.QDialog):
    def __init__(self, process):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_About()
        self.ui.setupUi(self)
        self.setWindowTitle(QtGui.QApplication.translate("About", "DAE Tools Project v1.1-0", None, QtGui.QApplication.UnicodeUTF8))
