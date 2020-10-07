# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'exception_dlg.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_ExceptionDialog(object):
    def setupUi(self, ExceptionDialog):
        ExceptionDialog.setObjectName("ExceptionDialog")
        ExceptionDialog.resize(700, 250)
        self.verticalLayout = QtWidgets.QVBoxLayout(ExceptionDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.exceptionLabel = QtWidgets.QLabel(ExceptionDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.exceptionLabel.sizePolicy().hasHeightForWidth())
        self.exceptionLabel.setSizePolicy(sizePolicy)
        self.exceptionLabel.setMinimumSize(QtCore.QSize(0, 50))
        self.exceptionLabel.setMouseTracking(False)
        self.exceptionLabel.setTextFormat(QtCore.Qt.PlainText)
        self.exceptionLabel.setScaledContents(False)
        self.exceptionLabel.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.exceptionLabel.setWordWrap(True)
        self.exceptionLabel.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
        self.exceptionLabel.setObjectName("exceptionLabel")
        self.verticalLayout.addWidget(self.exceptionLabel)
        self.tracebackEdit = QtWidgets.QPlainTextEdit(ExceptionDialog)
        self.tracebackEdit.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.tracebackEdit.sizePolicy().hasHeightForWidth())
        self.tracebackEdit.setSizePolicy(sizePolicy)
        self.tracebackEdit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.tracebackEdit.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.tracebackEdit.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.tracebackEdit.setReadOnly(True)
        self.tracebackEdit.setObjectName("tracebackEdit")
        self.verticalLayout.addWidget(self.tracebackEdit)
        self.buttonBox = QtWidgets.QDialogButtonBox(ExceptionDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Close)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(ExceptionDialog)
        self.buttonBox.accepted.connect(ExceptionDialog.accept)
        self.buttonBox.rejected.connect(ExceptionDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(ExceptionDialog)

    def retranslateUi(self, ExceptionDialog):
        _translate = QtCore.QCoreApplication.translate
        ExceptionDialog.setWindowTitle(_translate("ExceptionDialog", "Dialog"))
        self.exceptionLabel.setText(_translate("ExceptionDialog", "Exception"))

