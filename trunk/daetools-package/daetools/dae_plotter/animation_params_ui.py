# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'animation_params.ui'
#
# Created: Fri Jun  3 00:02:55 2016
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

class Ui_AnimationParameters(object):
    def setupUi(self, AnimationParameters):
        AnimationParameters.setObjectName(_fromUtf8("AnimationParameters"))
        AnimationParameters.resize(370, 238)
        AnimationParameters.setSizeGripEnabled(True)
        AnimationParameters.setModal(True)
        self.verticalLayout = QtGui.QVBoxLayout(AnimationParameters)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.formLayout = QtGui.QFormLayout()
        self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.label2 = QtGui.QLabel(AnimationParameters)
        self.label2.setObjectName(_fromUtf8("label2"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.label2)
        self.spinUpdateInterval = QtGui.QSpinBox(AnimationParameters)
        self.spinUpdateInterval.setMinimum(1)
        self.spinUpdateInterval.setMaximum(10000)
        self.spinUpdateInterval.setProperty("value", 100)
        self.spinUpdateInterval.setObjectName(_fromUtf8("spinUpdateInterval"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.spinUpdateInterval)
        self.label6 = QtGui.QLabel(AnimationParameters)
        self.label6.setObjectName(_fromUtf8("label6"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.label6)
        self.label5 = QtGui.QLabel(AnimationParameters)
        self.label5.setObjectName(_fromUtf8("label5"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.label5)
        self.label4 = QtGui.QLabel(AnimationParameters)
        self.label4.setObjectName(_fromUtf8("label4"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.LabelRole, self.label4)
        self.comboXmin = QtGui.QComboBox(AnimationParameters)
        self.comboXmin.setObjectName(_fromUtf8("comboXmin"))
        self.comboXmin.addItem(_fromUtf8(""))
        self.comboXmin.addItem(_fromUtf8(""))
        self.comboXmin.addItem(_fromUtf8(""))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.comboXmin)
        self.comboXmax = QtGui.QComboBox(AnimationParameters)
        self.comboXmax.setObjectName(_fromUtf8("comboXmax"))
        self.comboXmax.addItem(_fromUtf8(""))
        self.comboXmax.addItem(_fromUtf8(""))
        self.comboXmax.addItem(_fromUtf8(""))
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.comboXmax)
        self.comboYmin = QtGui.QComboBox(AnimationParameters)
        self.comboYmin.setObjectName(_fromUtf8("comboYmin"))
        self.comboYmin.addItem(_fromUtf8(""))
        self.comboYmin.addItem(_fromUtf8(""))
        self.comboYmin.addItem(_fromUtf8(""))
        self.formLayout.setWidget(3, QtGui.QFormLayout.FieldRole, self.comboYmin)
        self.comboYmax = QtGui.QComboBox(AnimationParameters)
        self.comboYmax.setObjectName(_fromUtf8("comboYmax"))
        self.comboYmax.addItem(_fromUtf8(""))
        self.comboYmax.addItem(_fromUtf8(""))
        self.comboYmax.addItem(_fromUtf8(""))
        self.formLayout.setWidget(4, QtGui.QFormLayout.FieldRole, self.comboYmax)
        self.label = QtGui.QLabel(AnimationParameters)
        self.label.setObjectName(_fromUtf8("label"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.LabelRole, self.label)
        self.verticalLayout.addLayout(self.formLayout)
        self.buttonBox = QtGui.QDialogButtonBox(AnimationParameters)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(AnimationParameters)
        self.comboYmin.setCurrentIndex(1)
        self.comboYmax.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(AnimationParameters)

    def retranslateUi(self, AnimationParameters):
        AnimationParameters.setWindowTitle(_translate("AnimationParameters", "Animation settings", None))
        self.label2.setText(_translate("AnimationParameters", "Update interval (ms)", None))
        self.label6.setText(_translate("AnimationParameters", "x min", None))
        self.label5.setText(_translate("AnimationParameters", "x max", None))
        self.label4.setText(_translate("AnimationParameters", "y min", None))
        self.comboXmin.setItemText(0, _translate("AnimationParameters", "From 1st frame", None))
        self.comboXmin.setItemText(1, _translate("AnimationParameters", "Overall x minimum value", None))
        self.comboXmin.setItemText(2, _translate("AnimationParameters", "Adaptive", None))
        self.comboXmax.setItemText(0, _translate("AnimationParameters", "From 1st frame", None))
        self.comboXmax.setItemText(1, _translate("AnimationParameters", "Overall x maximum value", None))
        self.comboXmax.setItemText(2, _translate("AnimationParameters", "Adaptive", None))
        self.comboYmin.setItemText(0, _translate("AnimationParameters", "From 1st frame", None))
        self.comboYmin.setItemText(1, _translate("AnimationParameters", "Overall y minimum value", None))
        self.comboYmin.setItemText(2, _translate("AnimationParameters", "Adaptive", None))
        self.comboYmax.setItemText(0, _translate("AnimationParameters", "From 1st frame", None))
        self.comboYmax.setItemText(1, _translate("AnimationParameters", "Overall y maximum value", None))
        self.comboYmax.setItemText(2, _translate("AnimationParameters", "Adaptive", None))
        self.label.setText(_translate("AnimationParameters", "y max", None))

