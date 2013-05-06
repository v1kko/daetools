# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'WebView.ui'
#
# Created: Sun May  5 15:12:39 2013
#      by: PyQt4 UI code generator 4.9.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_WebViewDialog(object):
    def setupUi(self, WebViewDialog):
        WebViewDialog.setObjectName(_fromUtf8("WebViewDialog"))
        WebViewDialog.setWindowModality(QtCore.Qt.WindowModal)
        WebViewDialog.resize(800, 500)
        WebViewDialog.setWindowTitle(_fromUtf8("Dialog"))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8("images/py.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        WebViewDialog.setWindowIcon(icon)
        WebViewDialog.setSizeGripEnabled(True)
        self.verticalLayout = QtGui.QVBoxLayout(WebViewDialog)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.webView = QtWebKit.QWebView(WebViewDialog)
        self.webView.setProperty("url", QtCore.QUrl(_fromUtf8("about:blank")))
        self.webView.setObjectName(_fromUtf8("webView"))
        self.verticalLayout.addWidget(self.webView)

        self.retranslateUi(WebViewDialog)
        QtCore.QMetaObject.connectSlotsByName(WebViewDialog)

    def retranslateUi(self, WebViewDialog):
        pass

from PyQt4 import QtWebKit
