# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'user_data.ui'
#
# Created: Wed Jun  8 17:36:50 2016
#      by: PyQt4 UI code generator 4.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_UserData(object):
    def setupUi(self, UserData):
        UserData.setObjectName(_fromUtf8("UserData"))
        UserData.resize(450, 400)
        UserData.setSizeGripEnabled(True)
        UserData.setModal(True)
        self.verticalLayout = QtGui.QVBoxLayout(UserData)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.formLayout = QtGui.QFormLayout()
        self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.label2 = QtGui.QLabel(UserData)
        self.label2.setObjectName(_fromUtf8("label2"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.label2)
        self.editXdata = QtGui.QPlainTextEdit(UserData)
        self.editXdata.setObjectName(_fromUtf8("editXdata"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.editXdata)
        self.label6 = QtGui.QLabel(UserData)
        self.label6.setObjectName(_fromUtf8("label6"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.label6)
        self.editYdata = QtGui.QPlainTextEdit(UserData)
        self.editYdata.setObjectName(_fromUtf8("editYdata"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.editYdata)
        self.label5 = QtGui.QLabel(UserData)
        self.label5.setObjectName(_fromUtf8("label5"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.label5)
        self.editXlabel = QtGui.QLineEdit(UserData)
        self.editXlabel.setObjectName(_fromUtf8("editXlabel"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.editXlabel)
        self.label4 = QtGui.QLabel(UserData)
        self.label4.setObjectName(_fromUtf8("label4"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.LabelRole, self.label4)
        self.editYlabel = QtGui.QLineEdit(UserData)
        self.editYlabel.setObjectName(_fromUtf8("editYlabel"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.FieldRole, self.editYlabel)
        self.curveTitleLabel = QtGui.QLabel(UserData)
        self.curveTitleLabel.setObjectName(_fromUtf8("curveTitleLabel"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.LabelRole, self.curveTitleLabel)
        self.editLineLabel = QtGui.QLineEdit(UserData)
        self.editLineLabel.setObjectName(_fromUtf8("editLineLabel"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.FieldRole, self.editLineLabel)
        self.verticalLayout.addLayout(self.formLayout)
        self.buttonBox = QtGui.QDialogButtonBox(UserData)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(UserData)
        QtCore.QMetaObject.connectSlotsByName(UserData)

    def retranslateUi(self, UserData):
        UserData.setWindowTitle(_translate("UserData", "Add user data", None))
        self.label2.setText(_translate("UserData", "x data", None))
        self.editXdata.setToolTip(_translate("UserData", "<html><head/><body><p>Array of values (numpy style, separator is \',\'):</p><p>1.0, 2.0, 3.0</p></body></html>", None))
        self.label6.setText(_translate("UserData", "y data", None))
        self.editYdata.setToolTip(_translate("UserData", "<html><head/><body><p>Array of values (numpy style, separator is \',\'):</p><p>1.0, 2.0, 3.0</p></body></html>", None))
        self.label5.setText(_translate("UserData", "x label", None))
        self.label4.setText(_translate("UserData", "y label", None))
        self.curveTitleLabel.setText(_translate("UserData", "Line label", None))

