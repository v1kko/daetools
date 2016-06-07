# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'custom_plots.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
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

class Ui_CustomPlots(object):
    def setupUi(self, CustomPlots):
        CustomPlots.setObjectName(_fromUtf8("CustomPlots"))
        CustomPlots.resize(750, 350)
        self.horizontalLayout_4 = QtGui.QHBoxLayout(CustomPlots)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtGui.QLayout.SetFixedSize)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.label = QtGui.QLabel(CustomPlots)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout.addWidget(self.label)
        self.listPlots = QtGui.QListWidget(CustomPlots)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.listPlots.sizePolicy().hasHeightForWidth())
        self.listPlots.setSizePolicy(sizePolicy)
        self.listPlots.setMinimumSize(QtCore.QSize(100, 0))
        self.listPlots.setBaseSize(QtCore.QSize(0, 0))
        self.listPlots.setAlternatingRowColors(True)
        self.listPlots.setObjectName(_fromUtf8("listPlots"))
        self.verticalLayout.addWidget(self.listPlots)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.buttonAddPlot = QtGui.QPushButton(CustomPlots)
        self.buttonAddPlot.setObjectName(_fromUtf8("buttonAddPlot"))
        self.horizontalLayout_2.addWidget(self.buttonAddPlot)
        self.buttonRemovePlot = QtGui.QPushButton(CustomPlots)
        self.buttonRemovePlot.setObjectName(_fromUtf8("buttonRemovePlot"))
        self.horizontalLayout_2.addWidget(self.buttonRemovePlot)
        spacerItem = QtGui.QSpacerItem(5, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.label_2 = QtGui.QLabel(CustomPlots)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.verticalLayout_2.addWidget(self.label_2)
        self.editSourceCode = QtGui.QPlainTextEdit(CustomPlots)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.editSourceCode.sizePolicy().hasHeightForWidth())
        self.editSourceCode.setSizePolicy(sizePolicy)
        self.editSourceCode.setMinimumSize(QtCore.QSize(400, 0))
        self.editSourceCode.setLineWrapMode(QtGui.QPlainTextEdit.NoWrap)
        self.editSourceCode.setObjectName(_fromUtf8("editSourceCode"))
        self.verticalLayout_2.addWidget(self.editSourceCode)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.buttonSave = QtGui.QPushButton(CustomPlots)
        self.buttonSave.setObjectName(_fromUtf8("buttonSave"))
        self.horizontalLayout_3.addWidget(self.buttonSave)
        self.buttonMakePlot = QtGui.QPushButton(CustomPlots)
        self.buttonMakePlot.setObjectName(_fromUtf8("buttonMakePlot"))
        self.horizontalLayout_3.addWidget(self.buttonMakePlot)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.horizontalLayout.setStretch(1, 3)
        self.horizontalLayout_4.addLayout(self.horizontalLayout)

        self.retranslateUi(CustomPlots)
        QtCore.QMetaObject.connectSlotsByName(CustomPlots)

    def retranslateUi(self, CustomPlots):
        CustomPlots.setWindowTitle(_translate("CustomPlots", "User-defined plots", None))
        self.label.setText(_translate("CustomPlots", "Plots:", None))
        self.buttonAddPlot.setText(_translate("CustomPlots", "Add...", None))
        self.buttonRemovePlot.setText(_translate("CustomPlots", "Remove", None))
        self.label_2.setText(_translate("CustomPlots", "Plot function:", None))
        self.buttonSave.setText(_translate("CustomPlots", "Save plots", None))
        self.buttonMakePlot.setText(_translate("CustomPlots", "Make plot", None))

