# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'WebView.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_WebViewDialog(object):
    def setupUi(self, WebViewDialog):
        WebViewDialog.setObjectName("WebViewDialog")
        WebViewDialog.setWindowModality(QtCore.Qt.WindowModal)
        WebViewDialog.resize(800, 500)
        WebViewDialog.setWindowTitle("Dialog")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("images/py.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        WebViewDialog.setWindowIcon(icon)
        WebViewDialog.setSizeGripEnabled(True)
        self.verticalLayout = QtWidgets.QVBoxLayout(WebViewDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.webView = QtWebKitWidgets.QWebView(WebViewDialog)
        self.webView.setProperty("url", QtCore.QUrl("about:blank"))
        self.webView.setObjectName("webView")
        self.verticalLayout.addWidget(self.webView)

        self.retranslateUi(WebViewDialog)
        QtCore.QMetaObject.connectSlotsByName(WebViewDialog)

    def retranslateUi(self, WebViewDialog):
        pass

from PyQt5 import QtWebKitWidgets
