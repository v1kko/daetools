# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tutorial17.ui'
#
# Created: Mon May 20 14:06:33 2013
#      by: PyQt4 UI code generator 4.9.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_tcpipLogServerMainWindow(object):
    def setupUi(self, tcpipLogServerMainWindow):
        tcpipLogServerMainWindow.setObjectName(_fromUtf8("tcpipLogServerMainWindow"))
        tcpipLogServerMainWindow.resize(600, 150)
        self.centralwidget = QtGui.QWidget(tcpipLogServerMainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.messagesEdit = QtGui.QTextEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Monospace"))
        font.setPointSize(9)
        self.messagesEdit.setFont(font)
        self.messagesEdit.setFrameShape(QtGui.QFrame.NoFrame)
        self.messagesEdit.setFrameShadow(QtGui.QFrame.Plain)
        self.messagesEdit.setUndoRedoEnabled(False)
        self.messagesEdit.setReadOnly(True)
        self.messagesEdit.setAcceptRichText(False)
        self.messagesEdit.setObjectName(_fromUtf8("messagesEdit"))
        self.verticalLayout.addWidget(self.messagesEdit)
        self.horizontalLayout.addLayout(self.verticalLayout)
        tcpipLogServerMainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(tcpipLogServerMainWindow)
        QtCore.QMetaObject.connectSlotsByName(tcpipLogServerMainWindow)

    def retranslateUi(self, tcpipLogServerMainWindow):
        tcpipLogServerMainWindow.setWindowTitle(QtGui.QApplication.translate("tcpipLogServerMainWindow", "TCP/IP Log Server", None, QtGui.QApplication.UnicodeUTF8))
        self.messagesEdit.setHtml(QtGui.QApplication.translate("tcpipLogServerMainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Monospace\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Here the output from the simulation transmitted by TCP/IP will be shown:</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))

