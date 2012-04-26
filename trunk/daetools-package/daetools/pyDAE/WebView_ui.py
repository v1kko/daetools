# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'WebView.ui'
#
# Created: Tue Dec 14 13:14:34 2010
#      by: PyQt4 UI code generator 4.7.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_WebViewDialog(object):
    def setupUi(self, WebViewDialog):
        WebViewDialog.setObjectName("WebViewDialog")
        WebViewDialog.resize(800, 500)
        WebViewDialog.setWindowTitle("Dialog")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("images/py.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        WebViewDialog.setWindowIcon(icon)
        WebViewDialog.setSizeGripEnabled(True)
        self.verticalLayout = QtGui.QVBoxLayout(WebViewDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.webView = QtWebKit.QWebView(WebViewDialog)
        self.webView.setUrl(QtCore.QUrl("about:blank"))
        self.webView.setObjectName("webView")
        self.verticalLayout.addWidget(self.webView)

        self.retranslateUi(WebViewDialog)
        QtCore.QMetaObject.connectSlotsByName(WebViewDialog)

    def retranslateUi(self, WebViewDialog):
        pass

from PyQt4 import QtWebKit
