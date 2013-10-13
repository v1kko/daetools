# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'editor_domain_array.ui'
#
# Created: Sat Oct 12 17:37:46 2013
#      by: PyQt4 UI code generator 4.9.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_EditorArrayDomain(object):
    def setupUi(self, EditorArrayDomain):
        EditorArrayDomain.setObjectName(_fromUtf8("EditorArrayDomain"))
        EditorArrayDomain.resize(250, 243)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(EditorArrayDomain.sizePolicy().hasHeightForWidth())
        EditorArrayDomain.setSizePolicy(sizePolicy)
        EditorArrayDomain.setMinimumSize(QtCore.QSize(250, 200))
        EditorArrayDomain.setWindowTitle(_fromUtf8(""))
        EditorArrayDomain.setFrameShape(QtGui.QFrame.NoFrame)
        EditorArrayDomain.setFrameShadow(QtGui.QFrame.Raised)
        self.verticalLayout = QtGui.QVBoxLayout(EditorArrayDomain)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.label_3 = QtGui.QLabel(EditorArrayDomain)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setMinimumSize(QtCore.QSize(100, 0))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.horizontalLayout_4.addWidget(self.label_3)
        self.numberOfPointsEdit = QtGui.QLineEdit(EditorArrayDomain)
        self.numberOfPointsEdit.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.numberOfPointsEdit.sizePolicy().hasHeightForWidth())
        self.numberOfPointsEdit.setSizePolicy(sizePolicy)
        self.numberOfPointsEdit.setMinimumSize(QtCore.QSize(50, 0))
        self.numberOfPointsEdit.setObjectName(_fromUtf8("numberOfPointsEdit"))
        self.horizontalLayout_4.addWidget(self.numberOfPointsEdit)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setSpacing(1)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.label_4 = QtGui.QLabel(EditorArrayDomain)
        self.label_4.setMinimumSize(QtCore.QSize(80, 0))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.verticalLayout_2.addWidget(self.label_4)
        self.descriptionEdit = QtGui.QTextEdit(EditorArrayDomain)
        self.descriptionEdit.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.descriptionEdit.sizePolicy().hasHeightForWidth())
        self.descriptionEdit.setSizePolicy(sizePolicy)
        self.descriptionEdit.setMinimumSize(QtCore.QSize(100, 50))
        self.descriptionEdit.setMaximumSize(QtCore.QSize(16777215, 100))
        self.descriptionEdit.setFrameShape(QtGui.QFrame.StyledPanel)
        self.descriptionEdit.setFrameShadow(QtGui.QFrame.Sunken)
        self.descriptionEdit.setReadOnly(True)
        self.descriptionEdit.setObjectName(_fromUtf8("descriptionEdit"))
        self.verticalLayout_2.addWidget(self.descriptionEdit)
        self.horizontalLayout_5.addLayout(self.verticalLayout_2)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.updateButton = QtGui.QPushButton(EditorArrayDomain)
        self.updateButton.setEnabled(False)
        self.updateButton.setAutoDefault(True)
        self.updateButton.setDefault(True)
        self.updateButton.setObjectName(_fromUtf8("updateButton"))
        self.horizontalLayout_2.addWidget(self.updateButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        spacerItem1 = QtGui.QSpacerItem(20, 10, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)

        self.retranslateUi(EditorArrayDomain)
        QtCore.QMetaObject.connectSlotsByName(EditorArrayDomain)

    def retranslateUi(self, EditorArrayDomain):
        self.label_3.setText(QtGui.QApplication.translate("EditorArrayDomain", "Number of Points", None, QtGui.QApplication.UnicodeUTF8))
        self.numberOfPointsEdit.setToolTip(QtGui.QApplication.translate("EditorArrayDomain", "<html><head/><body>Insert the number of points in the domain (integer).</body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("EditorArrayDomain", "Description", None, QtGui.QApplication.UnicodeUTF8))
        self.descriptionEdit.setToolTip(QtGui.QApplication.translate("EditorArrayDomain", "<html><head/><body><p>Description of the object.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.updateButton.setToolTip(QtGui.QApplication.translate("EditorArrayDomain", "<html><head/><body><p>Update the number of points.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.updateButton.setText(QtGui.QApplication.translate("EditorArrayDomain", "Update", None, QtGui.QApplication.UnicodeUTF8))

