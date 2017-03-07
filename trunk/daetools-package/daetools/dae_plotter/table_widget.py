# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'table_widget.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_tableWidgetDialog(object):
    def setupUi(self, tableWidgetDialog):
        tableWidgetDialog.setObjectName("tableWidgetDialog")
        tableWidgetDialog.resize(640, 480)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(tableWidgetDialog)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.tableWidget = QtWidgets.QTableWidget(tableWidgetDialog)
        self.tableWidget.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.tableWidget.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tableWidget.setAlternatingRowColors(True)
        self.tableWidget.setSelectionMode(QtWidgets.QAbstractItemView.ContiguousSelection)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.verticalLayout.addWidget(self.tableWidget)
        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.retranslateUi(tableWidgetDialog)
        QtCore.QMetaObject.connectSlotsByName(tableWidgetDialog)

    def retranslateUi(self, tableWidgetDialog):
        _translate = QtCore.QCoreApplication.translate
        tableWidgetDialog.setWindowTitle(_translate("tableWidgetDialog", "Data dialog"))

