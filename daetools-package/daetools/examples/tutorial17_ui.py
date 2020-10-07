# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tutorial17.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_tcpipLogServerMainWindow(object):
    def setupUi(self, tcpipLogServerMainWindow):
        tcpipLogServerMainWindow.setObjectName("tcpipLogServerMainWindow")
        tcpipLogServerMainWindow.resize(600, 150)
        self.centralwidget = QtWidgets.QWidget(tcpipLogServerMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.messagesEdit = QtWidgets.QTextEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Monospace")
        font.setPointSize(9)
        self.messagesEdit.setFont(font)
        self.messagesEdit.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.messagesEdit.setFrameShadow(QtWidgets.QFrame.Plain)
        self.messagesEdit.setUndoRedoEnabled(False)
        self.messagesEdit.setReadOnly(True)
        self.messagesEdit.setAcceptRichText(False)
        self.messagesEdit.setObjectName("messagesEdit")
        self.verticalLayout.addWidget(self.messagesEdit)
        self.horizontalLayout.addLayout(self.verticalLayout)
        tcpipLogServerMainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(tcpipLogServerMainWindow)
        QtCore.QMetaObject.connectSlotsByName(tcpipLogServerMainWindow)

    def retranslateUi(self, tcpipLogServerMainWindow):
        _translate = QtCore.QCoreApplication.translate
        tcpipLogServerMainWindow.setWindowTitle(_translate("tcpipLogServerMainWindow", "TCP/IP Log Server"))
        self.messagesEdit.setHtml(_translate("tcpipLogServerMainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Monospace\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Here the output from the simulation transmitted by TCP/IP will be shown:</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))

