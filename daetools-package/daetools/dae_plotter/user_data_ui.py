# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'user_data.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_UserData(object):
    def setupUi(self, UserData):
        UserData.setObjectName("UserData")
        UserData.resize(450, 400)
        UserData.setSizeGripEnabled(True)
        UserData.setModal(True)
        self.verticalLayout = QtWidgets.QVBoxLayout(UserData)
        self.verticalLayout.setObjectName("verticalLayout")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setObjectName("formLayout")
        self.label2 = QtWidgets.QLabel(UserData)
        self.label2.setObjectName("label2")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label2)
        self.editXdata = QtWidgets.QPlainTextEdit(UserData)
        self.editXdata.setObjectName("editXdata")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.editXdata)
        self.label6 = QtWidgets.QLabel(UserData)
        self.label6.setObjectName("label6")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label6)
        self.editYdata = QtWidgets.QPlainTextEdit(UserData)
        self.editYdata.setObjectName("editYdata")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.editYdata)
        self.label5 = QtWidgets.QLabel(UserData)
        self.label5.setObjectName("label5")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label5)
        self.editXlabel = QtWidgets.QLineEdit(UserData)
        self.editXlabel.setObjectName("editXlabel")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.editXlabel)
        self.label4 = QtWidgets.QLabel(UserData)
        self.label4.setObjectName("label4")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label4)
        self.editYlabel = QtWidgets.QLineEdit(UserData)
        self.editYlabel.setObjectName("editYlabel")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.editYlabel)
        self.curveTitleLabel = QtWidgets.QLabel(UserData)
        self.curveTitleLabel.setObjectName("curveTitleLabel")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.curveTitleLabel)
        self.editLineLabel = QtWidgets.QLineEdit(UserData)
        self.editLineLabel.setObjectName("editLineLabel")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.editLineLabel)
        self.verticalLayout.addLayout(self.formLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(UserData)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(UserData)
        QtCore.QMetaObject.connectSlotsByName(UserData)

    def retranslateUi(self, UserData):
        _translate = QtCore.QCoreApplication.translate
        UserData.setWindowTitle(_translate("UserData", "Add user data"))
        self.label2.setText(_translate("UserData", "x data"))
        self.editXdata.setToolTip(_translate("UserData", "<html><head/><body><p>Array of values (numpy style, separator is \',\'):</p><p>1.0, 2.0, 3.0</p></body></html>"))
        self.label6.setText(_translate("UserData", "y data"))
        self.editYdata.setToolTip(_translate("UserData", "<html><head/><body><p>Array of values (numpy style, separator is \',\'):</p><p>1.0, 2.0, 3.0</p></body></html>"))
        self.label5.setText(_translate("UserData", "x label"))
        self.label4.setText(_translate("UserData", "y label"))
        self.curveTitleLabel.setText(_translate("UserData", "Line label"))

