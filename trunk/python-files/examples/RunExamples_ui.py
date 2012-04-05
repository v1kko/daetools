# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'RunExamples.ui'
#
# Created: Thu Apr  5 23:49:57 2012
#      by: PyQt4 UI code generator 4.8.5
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_RunExamplesDialog(object):
    def setupUi(self, RunExamplesDialog):
        RunExamplesDialog.setObjectName(_fromUtf8("RunExamplesDialog"))
        RunExamplesDialog.resize(510, 160)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(RunExamplesDialog.sizePolicy().hasHeightForWidth())
        RunExamplesDialog.setSizePolicy(sizePolicy)
        RunExamplesDialog.setMinimumSize(QtCore.QSize(510, 160))
        RunExamplesDialog.setMaximumSize(QtCore.QSize(510, 160))
        RunExamplesDialog.setWindowTitle(QtGui.QApplication.translate("RunExamplesDialog", "DAE Tools Tutorials", None, QtGui.QApplication.UnicodeUTF8))
        RunExamplesDialog.setSizeGripEnabled(False)
        self.verticalLayout_3 = QtGui.QVBoxLayout(RunExamplesDialog)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label = QtGui.QLabel(RunExamplesDialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setText(QtGui.QApplication.translate("RunExamplesDialog", "Choose tutorial:", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        self.comboBoxExample = QtGui.QComboBox(RunExamplesDialog)
        self.comboBoxExample.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.comboBoxExample.sizePolicy().hasHeightForWidth())
        self.comboBoxExample.setSizePolicy(sizePolicy)
        self.comboBoxExample.setObjectName(_fromUtf8("comboBoxExample"))
        self.horizontalLayout.addWidget(self.comboBoxExample)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.toolButtonCode = QtGui.QToolButton(RunExamplesDialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toolButtonCode.sizePolicy().hasHeightForWidth())
        self.toolButtonCode.setSizePolicy(sizePolicy)
        self.toolButtonCode.setText(QtGui.QApplication.translate("RunExamplesDialog", "Show code...", None, QtGui.QApplication.UnicodeUTF8))
        self.toolButtonCode.setObjectName(_fromUtf8("toolButtonCode"))
        self.verticalLayout.addWidget(self.toolButtonCode)
        self.toolButtonRun = QtGui.QToolButton(RunExamplesDialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toolButtonRun.sizePolicy().hasHeightForWidth())
        self.toolButtonRun.setSizePolicy(sizePolicy)
        self.toolButtonRun.setText(QtGui.QApplication.translate("RunExamplesDialog", "Run...", None, QtGui.QApplication.UnicodeUTF8))
        self.toolButtonRun.setObjectName(_fromUtf8("toolButtonRun"))
        self.verticalLayout.addWidget(self.toolButtonRun)
        self.toolButtonModelReport = QtGui.QToolButton(RunExamplesDialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toolButtonModelReport.sizePolicy().hasHeightForWidth())
        self.toolButtonModelReport.setSizePolicy(sizePolicy)
        self.toolButtonModelReport.setText(QtGui.QApplication.translate("RunExamplesDialog", "Show model report ... *", None, QtGui.QApplication.UnicodeUTF8))
        self.toolButtonModelReport.setObjectName(_fromUtf8("toolButtonModelReport"))
        self.verticalLayout.addWidget(self.toolButtonModelReport)
        self.toolButtonRuntimeModelReport = QtGui.QToolButton(RunExamplesDialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toolButtonRuntimeModelReport.sizePolicy().hasHeightForWidth())
        self.toolButtonRuntimeModelReport.setSizePolicy(sizePolicy)
        self.toolButtonRuntimeModelReport.setMinimumSize(QtCore.QSize(0, 0))
        self.toolButtonRuntimeModelReport.setText(QtGui.QApplication.translate("RunExamplesDialog", "Show runtime model report ... *", None, QtGui.QApplication.UnicodeUTF8))
        self.toolButtonRuntimeModelReport.setObjectName(_fromUtf8("toolButtonRuntimeModelReport"))
        self.verticalLayout.addWidget(self.toolButtonRuntimeModelReport)
        self.label_2 = QtGui.QLabel(RunExamplesDialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setText(QtGui.QApplication.translate("RunExamplesDialog", "* Currently Mozilla Firefox is supported", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.verticalLayout.addWidget(self.label_2)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)

        self.retranslateUi(RunExamplesDialog)
        QtCore.QMetaObject.connectSlotsByName(RunExamplesDialog)

    def retranslateUi(self, RunExamplesDialog):
        pass

