# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'editor_parameter.ui'
#
# Created: Tue Oct  8 00:21:52 2013
#      by: PyQt4 UI code generator 4.9.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_EditorParameter(object):
    def setupUi(self, EditorParameter):
        EditorParameter.setObjectName(_fromUtf8("EditorParameter"))
        EditorParameter.resize(250, 300)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(EditorParameter.sizePolicy().hasHeightForWidth())
        EditorParameter.setSizePolicy(sizePolicy)
        EditorParameter.setMinimumSize(QtCore.QSize(250, 200))
        EditorParameter.setWindowTitle(_fromUtf8(""))
        EditorParameter.setFrameShape(QtGui.QFrame.NoFrame)
        EditorParameter.setFrameShadow(QtGui.QFrame.Raised)
        self.verticalLayout = QtGui.QVBoxLayout(EditorParameter)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label = QtGui.QLabel(EditorParameter)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(50, 0))
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        self.valueEdit = QtGui.QLineEdit(EditorParameter)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.valueEdit.sizePolicy().hasHeightForWidth())
        self.valueEdit.setSizePolicy(sizePolicy)
        self.valueEdit.setMinimumSize(QtCore.QSize(100, 0))
        self.valueEdit.setObjectName(_fromUtf8("valueEdit"))
        self.horizontalLayout.addWidget(self.valueEdit)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.label_2 = QtGui.QLabel(EditorParameter)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setMinimumSize(QtCore.QSize(50, 0))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_3.addWidget(self.label_2)
        self.unitsEdit = QtGui.QLineEdit(EditorParameter)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.unitsEdit.sizePolicy().hasHeightForWidth())
        self.unitsEdit.setSizePolicy(sizePolicy)
        self.unitsEdit.setMinimumSize(QtCore.QSize(100, 0))
        self.unitsEdit.setObjectName(_fromUtf8("unitsEdit"))
        self.horizontalLayout_3.addWidget(self.unitsEdit)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.updateButton = QtGui.QPushButton(EditorParameter)
        self.updateButton.setAutoDefault(True)
        self.updateButton.setDefault(True)
        self.updateButton.setObjectName(_fromUtf8("updateButton"))
        self.horizontalLayout_2.addWidget(self.updateButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        spacerItem1 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)

        self.retranslateUi(EditorParameter)
        QtCore.QMetaObject.connectSlotsByName(EditorParameter)

    def retranslateUi(self, EditorParameter):
        self.label.setText(QtGui.QApplication.translate("EditorParameter", "Value", None, QtGui.QApplication.UnicodeUTF8))
        self.valueEdit.setToolTip(QtGui.QApplication.translate("EditorParameter", "<html><head/><body><p>Insert the floating point value.The acceptable value is a valid Python float expression. Examples:</p>\n"
"<pre>1.02</pre>\n"
"<pre>5.032E-3</pre>\n"
"</body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("EditorParameter", "Units", None, QtGui.QApplication.UnicodeUTF8))
        self.unitsEdit.setToolTip(QtGui.QApplication.translate("EditorParameter", "<html><head/><body><p>Insert the unit expressions using daetools.pyUnits.unit objects using mathematical operators *, / and **. All base and derived SI units are allowed including the common prefixes (more information at <a href=\"https://en.wikipedia.org/wiki/International_System_of_Units#Units_and_prefixes\"><span style=\" text-decoration: underline; color:#0000ff;\">https://en.wikipedia.org/wiki/International_System_of_Units#Units_and_prefixes</span></a>). The expression entered is a valid Python code. Examples:</p>\n"
"<pre>m/kg</pre>\n"
"<pre>m / s**2</pre>\n"
"<pre>J / (kg*K)</pre>\n"
"</body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.updateButton.setToolTip(QtGui.QApplication.translate("EditorParameter", "<html><head/><body><p>Update the value.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.updateButton.setText(QtGui.QApplication.translate("EditorParameter", "Update", None, QtGui.QApplication.UnicodeUTF8))

