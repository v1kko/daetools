# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'nineml_tester.ui'
#
# Created: Tue Sep 13 15:54:21 2011
#      by: PyQt4 UI code generator 4.7.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_ninemlTester(object):
    def setupUi(self, ninemlTester):
        ninemlTester.setObjectName("ninemlTester")
        ninemlTester.resize(720, 420)
        self.verticalLayout_5 = QtGui.QVBoxLayout(ninemlTester)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.tabWidget = QtGui.QTabWidget(ninemlTester)
        self.tabWidget.setTabPosition(QtGui.QTabWidget.North)
        self.tabWidget.setTabShape(QtGui.QTabWidget.Rounded)
        self.tabWidget.setDocumentMode(True)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_1 = QtGui.QWidget()
        self.tab_1.setObjectName("tab_1")
        self.verticalLayout_6 = QtGui.QVBoxLayout(self.tab_1)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.formLayout = QtGui.QFormLayout()
        self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setObjectName("formLayout")
        self.timeHorizonLabel = QtGui.QLabel(self.tab_1)
        self.timeHorizonLabel.setObjectName("timeHorizonLabel")
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.timeHorizonLabel)
        self.timeHorizonSLineEdit = QtGui.QLineEdit(self.tab_1)
        self.timeHorizonSLineEdit.setInputMask("")
        self.timeHorizonSLineEdit.setObjectName("timeHorizonSLineEdit")
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.timeHorizonSLineEdit)
        self.reportingIntervalLabel = QtGui.QLabel(self.tab_1)
        self.reportingIntervalLabel.setObjectName("reportingIntervalLabel")
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.reportingIntervalLabel)
        self.reportingIntervalSLineEdit = QtGui.QLineEdit(self.tab_1)
        self.reportingIntervalSLineEdit.setObjectName("reportingIntervalSLineEdit")
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.reportingIntervalSLineEdit)
        self.verticalLayout_6.addLayout(self.formLayout)
        self.tabWidget.addTab(self.tab_1, "")
        self.tab_2 = QtGui.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.tab_2)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.treeParameters = QtGui.QTreeWidget(self.tab_2)
        self.treeParameters.setAlternatingRowColors(True)
        self.treeParameters.setObjectName("treeParameters")
        self.treeParameters.header().setDefaultSectionSize(300)
        self.treeParameters.header().setMinimumSectionSize(100)
        self.verticalLayout_4.addWidget(self.treeParameters)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtGui.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.tab_3)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.treeInitialConditions = QtGui.QTreeWidget(self.tab_3)
        self.treeInitialConditions.setAlternatingRowColors(True)
        self.treeInitialConditions.setObjectName("treeInitialConditions")
        self.treeInitialConditions.header().setDefaultSectionSize(300)
        self.treeInitialConditions.header().setMinimumSectionSize(100)
        self.verticalLayout_3.addWidget(self.treeInitialConditions)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtGui.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.tab_4)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.treeAnalogPorts = QtGui.QTreeWidget(self.tab_4)
        self.treeAnalogPorts.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.treeAnalogPorts.setAlternatingRowColors(True)
        self.treeAnalogPorts.setObjectName("treeAnalogPorts")
        self.treeAnalogPorts.header().setDefaultSectionSize(300)
        self.treeAnalogPorts.header().setMinimumSectionSize(100)
        self.verticalLayout_2.addWidget(self.treeAnalogPorts)
        self.tabWidget.addTab(self.tab_4, "")
        self.tab_6 = QtGui.QWidget()
        self.tab_6.setEnabled(True)
        self.tab_6.setObjectName("tab_6")
        self.verticalLayout = QtGui.QVBoxLayout(self.tab_6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.treeEventPorts = QtGui.QTreeWidget(self.tab_6)
        self.treeEventPorts.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.treeEventPorts.setAlternatingRowColors(True)
        self.treeEventPorts.setObjectName("treeEventPorts")
        self.treeEventPorts.header().setDefaultSectionSize(300)
        self.treeEventPorts.header().setMinimumSectionSize(100)
        self.verticalLayout.addWidget(self.treeEventPorts)
        self.tabWidget.addTab(self.tab_6, "")
        self.tab_5 = QtGui.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.verticalLayout_7 = QtGui.QVBoxLayout(self.tab_5)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.treeRegimes = QtGui.QTreeWidget(self.tab_5)
        self.treeRegimes.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.treeRegimes.setAlternatingRowColors(True)
        self.treeRegimes.setObjectName("treeRegimes")
        self.treeRegimes.header().setDefaultSectionSize(300)
        self.treeRegimes.header().setMinimumSectionSize(100)
        self.verticalLayout_7.addWidget(self.treeRegimes)
        self.tabWidget.addTab(self.tab_5, "")
        self.tab_7 = QtGui.QWidget()
        self.tab_7.setObjectName("tab_7")
        self.verticalLayout_8 = QtGui.QVBoxLayout(self.tab_7)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.treeResultsVariables = QtGui.QTreeWidget(self.tab_7)
        self.treeResultsVariables.setAlternatingRowColors(True)
        self.treeResultsVariables.setColumnCount(1)
        self.treeResultsVariables.setObjectName("treeResultsVariables")
        self.treeResultsVariables.header().setVisible(False)
        self.treeResultsVariables.header().setDefaultSectionSize(300)
        self.treeResultsVariables.header().setMinimumSectionSize(100)
        self.verticalLayout_8.addWidget(self.treeResultsVariables)
        self.tabWidget.addTab(self.tab_7, "")
        self.verticalLayout_5.addWidget(self.tabWidget)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.buttonCancel = QtGui.QPushButton(ninemlTester)
        self.buttonCancel.setObjectName("buttonCancel")
        self.horizontalLayout_2.addWidget(self.buttonCancel)
        self.buttonOk = QtGui.QPushButton(ninemlTester)
        self.buttonOk.setDefault(True)
        self.buttonOk.setObjectName("buttonOk")
        self.horizontalLayout_2.addWidget(self.buttonOk)
        self.verticalLayout_5.addLayout(self.horizontalLayout_2)

        self.retranslateUi(ninemlTester)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(ninemlTester)

    def retranslateUi(self, ninemlTester):
        ninemlTester.setWindowTitle(QtGui.QApplication.translate("ninemlTester", "NineML Component Tester", None, QtGui.QApplication.UnicodeUTF8))
        self.timeHorizonLabel.setText(QtGui.QApplication.translate("ninemlTester", "Time horizon, s", None, QtGui.QApplication.UnicodeUTF8))
        self.timeHorizonSLineEdit.setText(QtGui.QApplication.translate("ninemlTester", "0.0", None, QtGui.QApplication.UnicodeUTF8))
        self.reportingIntervalLabel.setText(QtGui.QApplication.translate("ninemlTester", "Reporting interval, s", None, QtGui.QApplication.UnicodeUTF8))
        self.reportingIntervalSLineEdit.setText(QtGui.QApplication.translate("ninemlTester", "0.0", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_1), QtGui.QApplication.translate("ninemlTester", "General", None, QtGui.QApplication.UnicodeUTF8))
        self.treeParameters.headerItem().setText(0, QtGui.QApplication.translate("ninemlTester", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.treeParameters.headerItem().setText(1, QtGui.QApplication.translate("ninemlTester", "Value", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QtGui.QApplication.translate("ninemlTester", "Parameters", None, QtGui.QApplication.UnicodeUTF8))
        self.treeInitialConditions.headerItem().setText(0, QtGui.QApplication.translate("ninemlTester", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.treeInitialConditions.headerItem().setText(1, QtGui.QApplication.translate("ninemlTester", "Value", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), QtGui.QApplication.translate("ninemlTester", "Initial Conditions", None, QtGui.QApplication.UnicodeUTF8))
        self.treeAnalogPorts.headerItem().setText(0, QtGui.QApplication.translate("ninemlTester", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.treeAnalogPorts.headerItem().setText(1, QtGui.QApplication.translate("ninemlTester", "Value", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), QtGui.QApplication.translate("ninemlTester", "Analog Ports", None, QtGui.QApplication.UnicodeUTF8))
        self.treeEventPorts.headerItem().setText(0, QtGui.QApplication.translate("ninemlTester", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.treeEventPorts.headerItem().setText(1, QtGui.QApplication.translate("ninemlTester", "Value", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_6), QtGui.QApplication.translate("ninemlTester", "Event Ports", None, QtGui.QApplication.UnicodeUTF8))
        self.treeRegimes.headerItem().setText(0, QtGui.QApplication.translate("ninemlTester", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.treeRegimes.headerItem().setText(1, QtGui.QApplication.translate("ninemlTester", "Value", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), QtGui.QApplication.translate("ninemlTester", "Regimes", None, QtGui.QApplication.UnicodeUTF8))
        self.treeResultsVariables.headerItem().setText(0, QtGui.QApplication.translate("ninemlTester", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_7), QtGui.QApplication.translate("ninemlTester", "Report Variables", None, QtGui.QApplication.UnicodeUTF8))
        self.buttonCancel.setText(QtGui.QApplication.translate("ninemlTester", "Cancel", None, QtGui.QApplication.UnicodeUTF8))
        self.buttonOk.setText(QtGui.QApplication.translate("ninemlTester", "Ok", None, QtGui.QApplication.UnicodeUTF8))

