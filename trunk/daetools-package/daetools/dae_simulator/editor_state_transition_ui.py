# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'editor_state_transition.ui'
#
# Created: Fri Oct 11 01:56:04 2013
#      by: PyQt4 UI code generator 4.9.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_EditorStateTransition(object):
    def setupUi(self, EditorStateTransition):
        EditorStateTransition.setObjectName(_fromUtf8("EditorStateTransition"))
        EditorStateTransition.resize(250, 300)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(EditorStateTransition.sizePolicy().hasHeightForWidth())
        EditorStateTransition.setSizePolicy(sizePolicy)
        EditorStateTransition.setMinimumSize(QtCore.QSize(250, 200))
        EditorStateTransition.setWindowTitle(_fromUtf8(""))
        EditorStateTransition.setFrameShape(QtGui.QFrame.NoFrame)
        EditorStateTransition.setFrameShadow(QtGui.QFrame.Raised)
        self.verticalLayout = QtGui.QVBoxLayout(EditorStateTransition)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label = QtGui.QLabel(EditorStateTransition)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(50, 0))
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        self.activeStateComboBox = QtGui.QComboBox(EditorStateTransition)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.activeStateComboBox.sizePolicy().hasHeightForWidth())
        self.activeStateComboBox.setSizePolicy(sizePolicy)
        self.activeStateComboBox.setObjectName(_fromUtf8("activeStateComboBox"))
        self.horizontalLayout.addWidget(self.activeStateComboBox)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.updateButton = QtGui.QPushButton(EditorStateTransition)
        self.updateButton.setAutoDefault(True)
        self.updateButton.setDefault(True)
        self.updateButton.setObjectName(_fromUtf8("updateButton"))
        self.horizontalLayout_2.addWidget(self.updateButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        spacerItem1 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)

        self.retranslateUi(EditorStateTransition)
        QtCore.QMetaObject.connectSlotsByName(EditorStateTransition)

    def retranslateUi(self, EditorStateTransition):
        self.label.setText(QtGui.QApplication.translate("EditorStateTransition", "Active State", None, QtGui.QApplication.UnicodeUTF8))
        self.activeStateComboBox.setToolTip(QtGui.QApplication.translate("EditorStateTransition", "<html><head/><body><p>Select the active state from the list of available states (string).</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.updateButton.setToolTip(QtGui.QApplication.translate("EditorStateTransition", "<html><head/><body><p>Update the active state.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.updateButton.setText(QtGui.QApplication.translate("EditorStateTransition", "Update", None, QtGui.QApplication.UnicodeUTF8))

