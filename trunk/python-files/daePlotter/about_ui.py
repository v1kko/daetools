# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'about.ui'
#
# Created: Sun Dec  5 03:54:00 2010
#      by: PyQt4 UI code generator 4.7.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_About(object):
    def setupUi(self, About):
        About.setObjectName("About")
        About.resize(600, 350)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("images/app.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        About.setWindowIcon(icon)
        self.horizontalLayout = QtGui.QHBoxLayout(About)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.textEdit = QtGui.QTextEdit(About)
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        self.textEdit.setFont(font)
        self.textEdit.setFrameShape(QtGui.QFrame.NoFrame)
        self.textEdit.setFrameShadow(QtGui.QFrame.Plain)
        self.textEdit.setUndoRedoEnabled(False)
        self.textEdit.setLineWrapMode(QtGui.QTextEdit.WidgetWidth)
        self.textEdit.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByKeyboard|QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextBrowserInteraction|QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
        self.textEdit.setObjectName("textEdit")
        self.horizontalLayout.addWidget(self.textEdit)

        self.retranslateUi(About)
        QtCore.QMetaObject.connectSlotsByName(About)

    def retranslateUi(self, About):
        About.setWindowTitle(QtGui.QApplication.translate("About", "DAE Tools Project: daePlotter", None, QtGui.QApplication.UnicodeUTF8))
        self.textEdit.setHtml(QtGui.QApplication.translate("About", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans Serif\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'Ubuntu\';\"></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))

