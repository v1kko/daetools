# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'editor_domain_array.ui'
#
# Created: Tue Oct  8 16:44:11 2013
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
        EditorArrayDomain.resize(250, 300)
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
        self.numberOfPointsEdit.setMinimumSize(QtCore.QSize(100, 0))
        self.numberOfPointsEdit.setObjectName(_fromUtf8("numberOfPointsEdit"))
        self.horizontalLayout_4.addWidget(self.numberOfPointsEdit)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
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
        spacerItem1 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)

        self.retranslateUi(EditorArrayDomain)
        QtCore.QMetaObject.connectSlotsByName(EditorArrayDomain)

    def retranslateUi(self, EditorArrayDomain):
        self.label_3.setText(QtGui.QApplication.translate("EditorArrayDomain", "No. Points", None, QtGui.QApplication.UnicodeUTF8))
        self.numberOfPointsEdit.setToolTip(QtGui.QApplication.translate("EditorArrayDomain", "<html><head/><body><p>Insert the unit expressions using daetools.pyUnits.unit objects using mathematical operators *, / and **. All base and derived SI units are allowed including the common prefixes (more information at <a href=\"https://en.wikipedia.org/wiki/International_System_of_Units#Units_and_prefixes\"><span style=\" text-decoration: underline; color:#0000ff;\">https://en.wikipedia.org/wiki/International_System_of_Units#Units_and_prefixes</span></a>). The expression entered is a valid Python code. Examples:</p>\n"
"<pre>m/kg</pre>\n"
"<pre>m / s**2</pre>\n"
"<pre>J / (kg*K)</pre>\n"
"</body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.updateButton.setToolTip(QtGui.QApplication.translate("EditorArrayDomain", "<html><head/><body><p>Update the value.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.updateButton.setText(QtGui.QApplication.translate("EditorArrayDomain", "Update", None, QtGui.QApplication.UnicodeUTF8))

