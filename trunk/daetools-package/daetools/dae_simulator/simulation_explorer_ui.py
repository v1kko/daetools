# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'simulation_explorer.ui'
#
# Created: Tue Oct  8 00:52:38 2013
#      by: PyQt4 UI code generator 4.9.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_SimulationExplorer(object):
    def setupUi(self, SimulationExplorer):
        SimulationExplorer.setObjectName(_fromUtf8("SimulationExplorer"))
        SimulationExplorer.setWindowModality(QtCore.Qt.WindowModal)
        SimulationExplorer.resize(760, 550)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(SimulationExplorer.sizePolicy().hasHeightForWidth())
        SimulationExplorer.setSizePolicy(sizePolicy)
        self.verticalLayout_2 = QtGui.QVBoxLayout(SimulationExplorer)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.tabWidget = QtGui.QTabWidget(SimulationExplorer)
        self.tabWidget.setMinimumSize(QtCore.QSize(250, 0))
        self.tabWidget.setTabPosition(QtGui.QTabWidget.North)
        self.tabWidget.setTabShape(QtGui.QTabWidget.Rounded)
        self.tabWidget.setDocumentMode(True)
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.tab_Runtume = QtGui.QWidget()
        self.tab_Runtume.setObjectName(_fromUtf8("tab_Runtume"))
        self.verticalLayout_6 = QtGui.QVBoxLayout(self.tab_Runtume)
        self.verticalLayout_6.setObjectName(_fromUtf8("verticalLayout_6"))
        self.formLayout = QtGui.QFormLayout()
        self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.testNameLabel = QtGui.QLabel(self.tab_Runtume)
        self.testNameLabel.setMinimumSize(QtCore.QSize(200, 0))
        self.testNameLabel.setObjectName(_fromUtf8("testNameLabel"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.testNameLabel)
        self.testNameLineEdit = QtGui.QLineEdit(self.tab_Runtume)
        self.testNameLineEdit.setObjectName(_fromUtf8("testNameLineEdit"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.testNameLineEdit)
        self.testDescriptionLabel = QtGui.QLabel(self.tab_Runtume)
        self.testDescriptionLabel.setMinimumSize(QtCore.QSize(200, 0))
        self.testDescriptionLabel.setObjectName(_fromUtf8("testDescriptionLabel"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.testDescriptionLabel)
        self.testDescriptionLineEdit = QtGui.QLineEdit(self.tab_Runtume)
        self.testDescriptionLineEdit.setText(_fromUtf8(""))
        self.testDescriptionLineEdit.setObjectName(_fromUtf8("testDescriptionLineEdit"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.testDescriptionLineEdit)
        self.timeHorizonLabel = QtGui.QLabel(self.tab_Runtume)
        self.timeHorizonLabel.setObjectName(_fromUtf8("timeHorizonLabel"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.timeHorizonLabel)
        self.timeHorizonSLineEdit = QtGui.QLineEdit(self.tab_Runtume)
        self.timeHorizonSLineEdit.setInputMask(_fromUtf8(""))
        self.timeHorizonSLineEdit.setObjectName(_fromUtf8("timeHorizonSLineEdit"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.timeHorizonSLineEdit)
        self.reportingIntervalLabel = QtGui.QLabel(self.tab_Runtume)
        self.reportingIntervalLabel.setObjectName(_fromUtf8("reportingIntervalLabel"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.LabelRole, self.reportingIntervalLabel)
        self.reportingIntervalSLineEdit = QtGui.QLineEdit(self.tab_Runtume)
        self.reportingIntervalSLineEdit.setObjectName(_fromUtf8("reportingIntervalSLineEdit"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.FieldRole, self.reportingIntervalSLineEdit)
        self.daesolverLabel = QtGui.QLabel(self.tab_Runtume)
        self.daesolverLabel.setObjectName(_fromUtf8("daesolverLabel"))
        self.formLayout.setWidget(6, QtGui.QFormLayout.LabelRole, self.daesolverLabel)
        self.daesolverComboBox = QtGui.QComboBox(self.tab_Runtume)
        self.daesolverComboBox.setObjectName(_fromUtf8("daesolverComboBox"))
        self.formLayout.setWidget(6, QtGui.QFormLayout.FieldRole, self.daesolverComboBox)
        self.lasolverLabel = QtGui.QLabel(self.tab_Runtume)
        self.lasolverLabel.setObjectName(_fromUtf8("lasolverLabel"))
        self.formLayout.setWidget(7, QtGui.QFormLayout.LabelRole, self.lasolverLabel)
        self.lasolverComboBox = QtGui.QComboBox(self.tab_Runtume)
        self.lasolverComboBox.setObjectName(_fromUtf8("lasolverComboBox"))
        self.formLayout.setWidget(7, QtGui.QFormLayout.FieldRole, self.lasolverComboBox)
        self.dataReporterLabel = QtGui.QLabel(self.tab_Runtume)
        self.dataReporterLabel.setObjectName(_fromUtf8("dataReporterLabel"))
        self.formLayout.setWidget(8, QtGui.QFormLayout.LabelRole, self.dataReporterLabel)
        self.dataReporterComboBox = QtGui.QComboBox(self.tab_Runtume)
        self.dataReporterComboBox.setObjectName(_fromUtf8("dataReporterComboBox"))
        self.formLayout.setWidget(8, QtGui.QFormLayout.FieldRole, self.dataReporterComboBox)
        self.logLabel = QtGui.QLabel(self.tab_Runtume)
        self.logLabel.setObjectName(_fromUtf8("logLabel"))
        self.formLayout.setWidget(9, QtGui.QFormLayout.LabelRole, self.logLabel)
        self.logComboBox = QtGui.QComboBox(self.tab_Runtume)
        self.logComboBox.setObjectName(_fromUtf8("logComboBox"))
        self.formLayout.setWidget(9, QtGui.QFormLayout.FieldRole, self.logComboBox)
        self.relativeToleranceLabel = QtGui.QLabel(self.tab_Runtume)
        self.relativeToleranceLabel.setObjectName(_fromUtf8("relativeToleranceLabel"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.LabelRole, self.relativeToleranceLabel)
        self.relativeToleranceLineEdit = QtGui.QLineEdit(self.tab_Runtume)
        self.relativeToleranceLineEdit.setObjectName(_fromUtf8("relativeToleranceLineEdit"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.FieldRole, self.relativeToleranceLineEdit)
        self.initialConditionsLabel = QtGui.QLabel(self.tab_Runtume)
        self.initialConditionsLabel.setMinimumSize(QtCore.QSize(300, 0))
        self.initialConditionsLabel.setObjectName(_fromUtf8("initialConditionsLabel"))
        self.formLayout.setWidget(5, QtGui.QFormLayout.LabelRole, self.initialConditionsLabel)
        self.quazySteadyStateInitialConditionsCheckBox = QtGui.QCheckBox(self.tab_Runtume)
        self.quazySteadyStateInitialConditionsCheckBox.setObjectName(_fromUtf8("quazySteadyStateInitialConditionsCheckBox"))
        self.formLayout.setWidget(5, QtGui.QFormLayout.FieldRole, self.quazySteadyStateInitialConditionsCheckBox)
        self.verticalLayout_6.addLayout(self.formLayout)
        self.tabWidget.addTab(self.tab_Runtume, _fromUtf8(""))
        self.tab_Domains = QtGui.QWidget()
        self.tab_Domains.setObjectName(_fromUtf8("tab_Domains"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.tab_Domains)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.treeDomains = QtGui.QTreeWidget(self.tab_Domains)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeDomains.sizePolicy().hasHeightForWidth())
        self.treeDomains.setSizePolicy(sizePolicy)
        self.treeDomains.setMinimumSize(QtCore.QSize(400, 200))
        self.treeDomains.setAlternatingRowColors(True)
        self.treeDomains.setObjectName(_fromUtf8("treeDomains"))
        self.treeDomains.header().setDefaultSectionSize(300)
        self.treeDomains.header().setMinimumSectionSize(100)
        self.horizontalLayout.addWidget(self.treeDomains)
        self.frameDomains = QtGui.QFrame(self.tab_Domains)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.frameDomains.sizePolicy().hasHeightForWidth())
        self.frameDomains.setSizePolicy(sizePolicy)
        self.frameDomains.setMinimumSize(QtCore.QSize(260, 200))
        self.frameDomains.setFrameShape(QtGui.QFrame.NoFrame)
        self.frameDomains.setFrameShadow(QtGui.QFrame.Raised)
        self.frameDomains.setObjectName(_fromUtf8("frameDomains"))
        self.horizontalLayout.addWidget(self.frameDomains)
        self.tabWidget.addTab(self.tab_Domains, _fromUtf8(""))
        self.tab_Parameters = QtGui.QWidget()
        self.tab_Parameters.setObjectName(_fromUtf8("tab_Parameters"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout(self.tab_Parameters)
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.treeParameters = QtGui.QTreeWidget(self.tab_Parameters)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeParameters.sizePolicy().hasHeightForWidth())
        self.treeParameters.setSizePolicy(sizePolicy)
        self.treeParameters.setMinimumSize(QtCore.QSize(100, 200))
        self.treeParameters.setAlternatingRowColors(True)
        self.treeParameters.setObjectName(_fromUtf8("treeParameters"))
        self.treeParameters.header().setDefaultSectionSize(300)
        self.treeParameters.header().setMinimumSectionSize(100)
        self.horizontalLayout_3.addWidget(self.treeParameters)
        self.frameParameters = QtGui.QFrame(self.tab_Parameters)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.frameParameters.sizePolicy().hasHeightForWidth())
        self.frameParameters.setSizePolicy(sizePolicy)
        self.frameParameters.setMinimumSize(QtCore.QSize(260, 200))
        self.frameParameters.setFrameShape(QtGui.QFrame.NoFrame)
        self.frameParameters.setFrameShadow(QtGui.QFrame.Raised)
        self.frameParameters.setObjectName(_fromUtf8("frameParameters"))
        self.horizontalLayout_3.addWidget(self.frameParameters)
        self.tabWidget.addTab(self.tab_Parameters, _fromUtf8(""))
        self.tab_InitialConditions = QtGui.QWidget()
        self.tab_InitialConditions.setObjectName(_fromUtf8("tab_InitialConditions"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout(self.tab_InitialConditions)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.treeInitialConditions = QtGui.QTreeWidget(self.tab_InitialConditions)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeInitialConditions.sizePolicy().hasHeightForWidth())
        self.treeInitialConditions.setSizePolicy(sizePolicy)
        self.treeInitialConditions.setMinimumSize(QtCore.QSize(400, 200))
        self.treeInitialConditions.setAlternatingRowColors(True)
        self.treeInitialConditions.setObjectName(_fromUtf8("treeInitialConditions"))
        self.treeInitialConditions.header().setDefaultSectionSize(300)
        self.treeInitialConditions.header().setMinimumSectionSize(100)
        self.horizontalLayout_4.addWidget(self.treeInitialConditions)
        self.frameInitialConditions = QtGui.QFrame(self.tab_InitialConditions)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.frameInitialConditions.sizePolicy().hasHeightForWidth())
        self.frameInitialConditions.setSizePolicy(sizePolicy)
        self.frameInitialConditions.setMinimumSize(QtCore.QSize(260, 200))
        self.frameInitialConditions.setFrameShape(QtGui.QFrame.NoFrame)
        self.frameInitialConditions.setFrameShadow(QtGui.QFrame.Raised)
        self.frameInitialConditions.setObjectName(_fromUtf8("frameInitialConditions"))
        self.horizontalLayout_4.addWidget(self.frameInitialConditions)
        self.tabWidget.addTab(self.tab_InitialConditions, _fromUtf8(""))
        self.tab_DOFs = QtGui.QWidget()
        self.tab_DOFs.setObjectName(_fromUtf8("tab_DOFs"))
        self.horizontalLayout_6 = QtGui.QHBoxLayout(self.tab_DOFs)
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        self.treeDOFs = QtGui.QTreeWidget(self.tab_DOFs)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeDOFs.sizePolicy().hasHeightForWidth())
        self.treeDOFs.setSizePolicy(sizePolicy)
        self.treeDOFs.setMinimumSize(QtCore.QSize(400, 200))
        self.treeDOFs.setAlternatingRowColors(True)
        self.treeDOFs.setObjectName(_fromUtf8("treeDOFs"))
        self.treeDOFs.header().setDefaultSectionSize(300)
        self.treeDOFs.header().setMinimumSectionSize(100)
        self.horizontalLayout_6.addWidget(self.treeDOFs)
        self.frameDOFs = QtGui.QFrame(self.tab_DOFs)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.frameDOFs.sizePolicy().hasHeightForWidth())
        self.frameDOFs.setSizePolicy(sizePolicy)
        self.frameDOFs.setMinimumSize(QtCore.QSize(260, 200))
        self.frameDOFs.setFrameShape(QtGui.QFrame.NoFrame)
        self.frameDOFs.setFrameShadow(QtGui.QFrame.Raised)
        self.frameDOFs.setObjectName(_fromUtf8("frameDOFs"))
        self.horizontalLayout_6.addWidget(self.frameDOFs)
        self.tabWidget.addTab(self.tab_DOFs, _fromUtf8(""))
        self.tab_StateTransitions = QtGui.QWidget()
        self.tab_StateTransitions.setObjectName(_fromUtf8("tab_StateTransitions"))
        self.horizontalLayout_5 = QtGui.QHBoxLayout(self.tab_StateTransitions)
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.treeStateTransitions = QtGui.QTreeWidget(self.tab_StateTransitions)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeStateTransitions.sizePolicy().hasHeightForWidth())
        self.treeStateTransitions.setSizePolicy(sizePolicy)
        self.treeStateTransitions.setMinimumSize(QtCore.QSize(400, 200))
        self.treeStateTransitions.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.treeStateTransitions.setAlternatingRowColors(True)
        self.treeStateTransitions.setObjectName(_fromUtf8("treeStateTransitions"))
        self.treeStateTransitions.header().setDefaultSectionSize(300)
        self.treeStateTransitions.header().setMinimumSectionSize(100)
        self.horizontalLayout_5.addWidget(self.treeStateTransitions)
        self.frameStateTransitions = QtGui.QFrame(self.tab_StateTransitions)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.frameStateTransitions.sizePolicy().hasHeightForWidth())
        self.frameStateTransitions.setSizePolicy(sizePolicy)
        self.frameStateTransitions.setMinimumSize(QtCore.QSize(260, 200))
        self.frameStateTransitions.setFrameShape(QtGui.QFrame.NoFrame)
        self.frameStateTransitions.setFrameShadow(QtGui.QFrame.Raised)
        self.frameStateTransitions.setObjectName(_fromUtf8("frameStateTransitions"))
        self.horizontalLayout_5.addWidget(self.frameStateTransitions)
        self.tabWidget.addTab(self.tab_StateTransitions, _fromUtf8(""))
        self.tab_Outputs = QtGui.QWidget()
        self.tab_Outputs.setObjectName(_fromUtf8("tab_Outputs"))
        self.verticalLayout_8 = QtGui.QVBoxLayout(self.tab_Outputs)
        self.verticalLayout_8.setObjectName(_fromUtf8("verticalLayout_8"))
        self.treeOutputVariables = QtGui.QTreeWidget(self.tab_Outputs)
        self.treeOutputVariables.setAlternatingRowColors(True)
        self.treeOutputVariables.setColumnCount(1)
        self.treeOutputVariables.setObjectName(_fromUtf8("treeOutputVariables"))
        self.treeOutputVariables.header().setVisible(False)
        self.treeOutputVariables.header().setDefaultSectionSize(300)
        self.treeOutputVariables.header().setMinimumSectionSize(100)
        self.verticalLayout_8.addWidget(self.treeOutputVariables)
        self.tabWidget.addTab(self.tab_Outputs, _fromUtf8(""))
        self.verticalLayout_2.addWidget(self.tabWidget)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.buttonOk = QtGui.QPushButton(SimulationExplorer)
        self.buttonOk.setDefault(True)
        self.buttonOk.setObjectName(_fromUtf8("buttonOk"))
        self.horizontalLayout_2.addWidget(self.buttonOk)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.retranslateUi(SimulationExplorer)
        self.tabWidget.setCurrentIndex(6)
        QtCore.QMetaObject.connectSlotsByName(SimulationExplorer)

    def retranslateUi(self, SimulationExplorer):
        SimulationExplorer.setWindowTitle(QtGui.QApplication.translate("SimulationExplorer", "Simulation Explorer", None, QtGui.QApplication.UnicodeUTF8))
        self.testNameLabel.setText(QtGui.QApplication.translate("SimulationExplorer", "Simulation name", None, QtGui.QApplication.UnicodeUTF8))
        self.testNameLineEdit.setText(QtGui.QApplication.translate("SimulationExplorer", "Simulation", None, QtGui.QApplication.UnicodeUTF8))
        self.testDescriptionLabel.setText(QtGui.QApplication.translate("SimulationExplorer", "Simulation description", None, QtGui.QApplication.UnicodeUTF8))
        self.timeHorizonLabel.setText(QtGui.QApplication.translate("SimulationExplorer", "Time horizon, s", None, QtGui.QApplication.UnicodeUTF8))
        self.timeHorizonSLineEdit.setText(QtGui.QApplication.translate("SimulationExplorer", "0.0", None, QtGui.QApplication.UnicodeUTF8))
        self.reportingIntervalLabel.setText(QtGui.QApplication.translate("SimulationExplorer", "Reporting interval, s", None, QtGui.QApplication.UnicodeUTF8))
        self.reportingIntervalSLineEdit.setText(QtGui.QApplication.translate("SimulationExplorer", "0.0", None, QtGui.QApplication.UnicodeUTF8))
        self.daesolverLabel.setText(QtGui.QApplication.translate("SimulationExplorer", "DAE Solver", None, QtGui.QApplication.UnicodeUTF8))
        self.lasolverLabel.setText(QtGui.QApplication.translate("SimulationExplorer", "LA Solver", None, QtGui.QApplication.UnicodeUTF8))
        self.dataReporterLabel.setText(QtGui.QApplication.translate("SimulationExplorer", "Data Reporter", None, QtGui.QApplication.UnicodeUTF8))
        self.logLabel.setText(QtGui.QApplication.translate("SimulationExplorer", "Log", None, QtGui.QApplication.UnicodeUTF8))
        self.relativeToleranceLabel.setText(QtGui.QApplication.translate("SimulationExplorer", "Relative Tolerance", None, QtGui.QApplication.UnicodeUTF8))
        self.initialConditionsLabel.setText(QtGui.QApplication.translate("SimulationExplorer", "Quazy Steady State InitialConditions", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_Runtume), QtGui.QApplication.translate("SimulationExplorer", "Runtime", None, QtGui.QApplication.UnicodeUTF8))
        self.treeDomains.headerItem().setText(0, QtGui.QApplication.translate("SimulationExplorer", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.treeDomains.headerItem().setText(1, QtGui.QApplication.translate("SimulationExplorer", "Value", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_Domains), QtGui.QApplication.translate("SimulationExplorer", "Domains", None, QtGui.QApplication.UnicodeUTF8))
        self.treeParameters.headerItem().setText(0, QtGui.QApplication.translate("SimulationExplorer", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.treeParameters.headerItem().setText(1, QtGui.QApplication.translate("SimulationExplorer", "Value", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_Parameters), QtGui.QApplication.translate("SimulationExplorer", "Parameters", None, QtGui.QApplication.UnicodeUTF8))
        self.treeInitialConditions.headerItem().setText(0, QtGui.QApplication.translate("SimulationExplorer", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.treeInitialConditions.headerItem().setText(1, QtGui.QApplication.translate("SimulationExplorer", "Value", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_InitialConditions), QtGui.QApplication.translate("SimulationExplorer", "Initial Conditions", None, QtGui.QApplication.UnicodeUTF8))
        self.treeDOFs.headerItem().setText(0, QtGui.QApplication.translate("SimulationExplorer", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.treeDOFs.headerItem().setText(1, QtGui.QApplication.translate("SimulationExplorer", "Value", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_DOFs), QtGui.QApplication.translate("SimulationExplorer", "DOFs", None, QtGui.QApplication.UnicodeUTF8))
        self.treeStateTransitions.headerItem().setText(0, QtGui.QApplication.translate("SimulationExplorer", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.treeStateTransitions.headerItem().setText(1, QtGui.QApplication.translate("SimulationExplorer", "Value", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_StateTransitions), QtGui.QApplication.translate("SimulationExplorer", "State Transitions", None, QtGui.QApplication.UnicodeUTF8))
        self.treeOutputVariables.headerItem().setText(0, QtGui.QApplication.translate("SimulationExplorer", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_Outputs), QtGui.QApplication.translate("SimulationExplorer", "Outputs", None, QtGui.QApplication.UnicodeUTF8))
        self.buttonOk.setText(QtGui.QApplication.translate("SimulationExplorer", "Run", None, QtGui.QApplication.UnicodeUTF8))

