# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'exception_dlg.ui'
#
# Created: Fri Oct 18 21:53:33 2013
#      by: PyQt4 UI code generator 4.9.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_ExceptionDialog(object):
    def setupUi(self, ExceptionDialog):
        ExceptionDialog.setObjectName(_fromUtf8("ExceptionDialog"))
        ExceptionDialog.resize(700, 250)
        self.verticalLayout = QtGui.QVBoxLayout(ExceptionDialog)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.exceptionLabel = QtGui.QLabel(ExceptionDialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.exceptionLabel.sizePolicy().hasHeightForWidth())
        self.exceptionLabel.setSizePolicy(sizePolicy)
        self.exceptionLabel.setMinimumSize(QtCore.QSize(0, 50))
        self.exceptionLabel.setTextFormat(QtCore.Qt.LogText)
        self.exceptionLabel.setScaledContents(False)
        self.exceptionLabel.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.exceptionLabel.setWordWrap(True)
        self.exceptionLabel.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
        self.exceptionLabel.setObjectName(_fromUtf8("exceptionLabel"))
        self.verticalLayout.addWidget(self.exceptionLabel)
        self.tracebackEdit = QtGui.QPlainTextEdit(ExceptionDialog)
        self.tracebackEdit.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.tracebackEdit.sizePolicy().hasHeightForWidth())
        self.tracebackEdit.setSizePolicy(sizePolicy)
        self.tracebackEdit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.tracebackEdit.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.tracebackEdit.setLineWrapMode(QtGui.QPlainTextEdit.NoWrap)
        self.tracebackEdit.setReadOnly(True)
        self.tracebackEdit.setObjectName(_fromUtf8("tracebackEdit"))
        self.verticalLayout.addWidget(self.tracebackEdit)
        self.buttonBox = QtGui.QDialogButtonBox(ExceptionDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Close)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(ExceptionDialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), ExceptionDialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), ExceptionDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(ExceptionDialog)

    def retranslateUi(self, ExceptionDialog):
        ExceptionDialog.setWindowTitle(QtGui.QApplication.translate("ExceptionDialog", "Dialog", None, QtGui.QApplication.UnicodeUTF8))
        self.exceptionLabel.setText(QtGui.QApplication.translate("ExceptionDialog", "Exception", None, QtGui.QApplication.UnicodeUTF8))

