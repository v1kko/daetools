# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'table_widget.ui'
#
# Created: Wed May  5 03:55:28 2010
#      by: PyQt4 UI code generator 4.7.2
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_tableWidgetDialog(object):
    def setupUi(self, tableWidgetDialog):
        tableWidgetDialog.setObjectName("tableWidgetDialog")
        tableWidgetDialog.resize(640, 480)
        self.verticalLayout_2 = QtGui.QVBoxLayout(tableWidgetDialog)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.tableWidget = QtGui.QTableWidget(tableWidgetDialog)
        self.tableWidget.setFrameShape(QtGui.QFrame.StyledPanel)
        self.tableWidget.setFrameShadow(QtGui.QFrame.Sunken)
        self.tableWidget.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.tableWidget.setAlternatingRowColors(True)
        self.tableWidget.setSelectionMode(QtGui.QAbstractItemView.ContiguousSelection)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.verticalLayout.addWidget(self.tableWidget)
        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.retranslateUi(tableWidgetDialog)
        QtCore.QMetaObject.connectSlotsByName(tableWidgetDialog)

    def retranslateUi(self, tableWidgetDialog):
        tableWidgetDialog.setWindowTitle(QtGui.QApplication.translate("tableWidgetDialog", "Data dialog", None, QtGui.QApplication.UnicodeUTF8))

