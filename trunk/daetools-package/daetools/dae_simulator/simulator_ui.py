# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Simulator.ui'
#
# Created: Sun Dec 14 15:20:56 2014
#      by: PyQt4 UI code generator 4.9.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_SimulatorDialog(object):
    def setupUi(self, SimulatorDialog):
        SimulatorDialog.setObjectName(_fromUtf8("SimulatorDialog"))
        SimulatorDialog.resize(700, 580)
        self.verticalLayout_2 = QtGui.QVBoxLayout(SimulatorDialog)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.formLayout = QtGui.QFormLayout()
        self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.simulationLabel = QtGui.QLabel(SimulatorDialog)
        self.simulationLabel.setObjectName(_fromUtf8("simulationLabel"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.simulationLabel)
        self.SimulationLineEdit = QtGui.QLineEdit(SimulatorDialog)
        self.SimulationLineEdit.setEnabled(False)
        self.SimulationLineEdit.setObjectName(_fromUtf8("SimulationLineEdit"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.SimulationLineEdit)
        self.DataReporterTCPIPAddressLabel = QtGui.QLabel(SimulatorDialog)
        self.DataReporterTCPIPAddressLabel.setObjectName(_fromUtf8("DataReporterTCPIPAddressLabel"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.DataReporterTCPIPAddressLabel)
        self.DataReporterTCPIPAddressLineEdit = QtGui.QLineEdit(SimulatorDialog)
        self.DataReporterTCPIPAddressLineEdit.setText(_fromUtf8(""))
        self.DataReporterTCPIPAddressLineEdit.setObjectName(_fromUtf8("DataReporterTCPIPAddressLineEdit"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.DataReporterTCPIPAddressLineEdit)
        self.MINLPSolverLabel = QtGui.QLabel(SimulatorDialog)
        self.MINLPSolverLabel.setObjectName(_fromUtf8("MINLPSolverLabel"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.MINLPSolverLabel)
        self.MINLPSolverComboBox = QtGui.QComboBox(SimulatorDialog)
        self.MINLPSolverComboBox.setObjectName(_fromUtf8("MINLPSolverComboBox"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.MINLPSolverComboBox)
        self.DAESolverLabel = QtGui.QLabel(SimulatorDialog)
        self.DAESolverLabel.setObjectName(_fromUtf8("DAESolverLabel"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.LabelRole, self.DAESolverLabel)
        self.DAESolverComboBox = QtGui.QComboBox(SimulatorDialog)
        self.DAESolverComboBox.setObjectName(_fromUtf8("DAESolverComboBox"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.FieldRole, self.DAESolverComboBox)
        self.LASolverLabel = QtGui.QLabel(SimulatorDialog)
        self.LASolverLabel.setObjectName(_fromUtf8("LASolverLabel"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.LabelRole, self.LASolverLabel)
        self.LASolverComboBox = QtGui.QComboBox(SimulatorDialog)
        self.LASolverComboBox.setObjectName(_fromUtf8("LASolverComboBox"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.FieldRole, self.LASolverComboBox)
        self.TimeHorizonLabel = QtGui.QLabel(SimulatorDialog)
        self.TimeHorizonLabel.setObjectName(_fromUtf8("TimeHorizonLabel"))
        self.formLayout.setWidget(5, QtGui.QFormLayout.LabelRole, self.TimeHorizonLabel)
        self.TimeHorizonDoubleSpinBox = QtGui.QDoubleSpinBox(SimulatorDialog)
        self.TimeHorizonDoubleSpinBox.setMaximum(1000000000.0)
        self.TimeHorizonDoubleSpinBox.setObjectName(_fromUtf8("TimeHorizonDoubleSpinBox"))
        self.formLayout.setWidget(5, QtGui.QFormLayout.FieldRole, self.TimeHorizonDoubleSpinBox)
        self.ReportingIntervalLabel = QtGui.QLabel(SimulatorDialog)
        self.ReportingIntervalLabel.setObjectName(_fromUtf8("ReportingIntervalLabel"))
        self.formLayout.setWidget(6, QtGui.QFormLayout.LabelRole, self.ReportingIntervalLabel)
        self.ReportingIntervalDoubleSpinBox = QtGui.QDoubleSpinBox(SimulatorDialog)
        self.ReportingIntervalDoubleSpinBox.setMaximum(1000000000.0)
        self.ReportingIntervalDoubleSpinBox.setObjectName(_fromUtf8("ReportingIntervalDoubleSpinBox"))
        self.formLayout.setWidget(6, QtGui.QFormLayout.FieldRole, self.ReportingIntervalDoubleSpinBox)
        self.verticalLayout_2.addLayout(self.formLayout)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.ExportButton = QtGui.QPushButton(SimulatorDialog)
        self.ExportButton.setEnabled(True)
        self.ExportButton.setObjectName(_fromUtf8("ExportButton"))
        self.horizontalLayout.addWidget(self.ExportButton)
        self.MatrixButton = QtGui.QPushButton(SimulatorDialog)
        self.MatrixButton.setEnabled(True)
        self.MatrixButton.setObjectName(_fromUtf8("MatrixButton"))
        self.horizontalLayout.addWidget(self.MatrixButton)
        self.RunButton = QtGui.QPushButton(SimulatorDialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.RunButton.sizePolicy().hasHeightForWidth())
        self.RunButton.setSizePolicy(sizePolicy)
        self.RunButton.setDefault(True)
        self.RunButton.setObjectName(_fromUtf8("RunButton"))
        self.horizontalLayout.addWidget(self.RunButton)
        self.PauseButton = QtGui.QPushButton(SimulatorDialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PauseButton.sizePolicy().hasHeightForWidth())
        self.PauseButton.setSizePolicy(sizePolicy)
        self.PauseButton.setObjectName(_fromUtf8("PauseButton"))
        self.horizontalLayout.addWidget(self.PauseButton)
        self.ResumeButton = QtGui.QPushButton(SimulatorDialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ResumeButton.sizePolicy().hasHeightForWidth())
        self.ResumeButton.setSizePolicy(sizePolicy)
        self.ResumeButton.setCheckable(False)
        self.ResumeButton.setChecked(False)
        self.ResumeButton.setObjectName(_fromUtf8("ResumeButton"))
        self.horizontalLayout.addWidget(self.ResumeButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.textEdit = QtGui.QTextEdit(SimulatorDialog)
        self.textEdit.setEnabled(True)
        self.textEdit.setMinimumSize(QtCore.QSize(510, 180))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Monospace"))
        font.setPointSize(9)
        self.textEdit.setFont(font)
        self.textEdit.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.textEdit.setUndoRedoEnabled(False)
        self.textEdit.setLineWrapMode(QtGui.QTextEdit.NoWrap)
        self.textEdit.setReadOnly(True)
        self.textEdit.setHtml(_fromUtf8("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Monospace\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'Courier New\';\"><br /></p></body></html>"))
        self.textEdit.setObjectName(_fromUtf8("textEdit"))
        self.verticalLayout.addWidget(self.textEdit)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.progressBar = QtGui.QProgressBar(SimulatorDialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy)
        self.progressBar.setMinimumSize(QtCore.QSize(150, 0))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setInvertedAppearance(False)
        self.progressBar.setObjectName(_fromUtf8("progressBar"))
        self.horizontalLayout_3.addWidget(self.progressBar)
        self.progressLabel = QtGui.QLabel(SimulatorDialog)
        self.progressLabel.setText(_fromUtf8(""))
        self.progressLabel.setObjectName(_fromUtf8("progressLabel"))
        self.horizontalLayout_3.addWidget(self.progressLabel)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)

        self.retranslateUi(SimulatorDialog)
        QtCore.QMetaObject.connectSlotsByName(SimulatorDialog)

    def retranslateUi(self, SimulatorDialog):
        SimulatorDialog.setWindowTitle(QtGui.QApplication.translate("SimulatorDialog", "DAE Tools Simulator", None, QtGui.QApplication.UnicodeUTF8))
        self.simulationLabel.setText(QtGui.QApplication.translate("SimulatorDialog", "Simulation", None, QtGui.QApplication.UnicodeUTF8))
        self.DataReporterTCPIPAddressLabel.setText(QtGui.QApplication.translate("SimulatorDialog", "Data Reporter (TCP/IP : port)", None, QtGui.QApplication.UnicodeUTF8))
        self.MINLPSolverLabel.setText(QtGui.QApplication.translate("SimulatorDialog", "(MI)NLP Solver", None, QtGui.QApplication.UnicodeUTF8))
        self.DAESolverLabel.setText(QtGui.QApplication.translate("SimulatorDialog", "DAE Solver", None, QtGui.QApplication.UnicodeUTF8))
        self.LASolverLabel.setText(QtGui.QApplication.translate("SimulatorDialog", "LA Solver", None, QtGui.QApplication.UnicodeUTF8))
        self.TimeHorizonLabel.setText(QtGui.QApplication.translate("SimulatorDialog", "Time Horizon, s", None, QtGui.QApplication.UnicodeUTF8))
        self.ReportingIntervalLabel.setText(QtGui.QApplication.translate("SimulatorDialog", "Reporting Interval, s", None, QtGui.QApplication.UnicodeUTF8))
        self.ExportButton.setText(QtGui.QApplication.translate("SimulatorDialog", "Export Matrix...", None, QtGui.QApplication.UnicodeUTF8))
        self.MatrixButton.setText(QtGui.QApplication.translate("SimulatorDialog", "Matrix Preview...", None, QtGui.QApplication.UnicodeUTF8))
        self.RunButton.setText(QtGui.QApplication.translate("SimulatorDialog", "Start...", None, QtGui.QApplication.UnicodeUTF8))
        self.PauseButton.setText(QtGui.QApplication.translate("SimulatorDialog", "Pause", None, QtGui.QApplication.UnicodeUTF8))
        self.ResumeButton.setText(QtGui.QApplication.translate("SimulatorDialog", "Resume", None, QtGui.QApplication.UnicodeUTF8))

