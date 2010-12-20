from PyQt4 import QtCore, QtGui
from daetools.pyDAE.WebView_ui import Ui_WebViewDialog

class WebView(QtGui.QDialog):
    def __init__(self, url):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_WebViewDialog()
        self.ui.setupUi(self)
        self.ui.webView.load(url)
