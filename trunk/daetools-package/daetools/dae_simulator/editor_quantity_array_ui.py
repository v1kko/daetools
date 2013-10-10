# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'editor_quantity_array.ui'
#
# Created: Thu Oct 10 01:19:39 2013
#      by: PyQt4 UI code generator 4.9.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_EditorQuantityArray(object):
    def setupUi(self, EditorQuantityArray):
        EditorQuantityArray.setObjectName(_fromUtf8("EditorQuantityArray"))
        EditorQuantityArray.resize(250, 303)
        EditorQuantityArray.setMinimumSize(QtCore.QSize(250, 200))
        EditorQuantityArray.setWindowTitle(_fromUtf8(""))
        EditorQuantityArray.setFrameShape(QtGui.QFrame.NoFrame)
        EditorQuantityArray.setFrameShadow(QtGui.QFrame.Raised)
        self.verticalLayout = QtGui.QVBoxLayout(EditorQuantityArray)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.label = QtGui.QLabel(EditorQuantityArray)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(50, 0))
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout_3.addWidget(self.label)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.valueEdit = QtGui.QPlainTextEdit(EditorQuantityArray)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.valueEdit.sizePolicy().hasHeightForWidth())
        self.valueEdit.setSizePolicy(sizePolicy)
        self.valueEdit.setMinimumSize(QtCore.QSize(100, 100))
        self.valueEdit.setLineWrapMode(QtGui.QPlainTextEdit.WidgetWidth)
        self.valueEdit.setObjectName(_fromUtf8("valueEdit"))
        self.horizontalLayout.addWidget(self.valueEdit)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.label_2 = QtGui.QLabel(EditorQuantityArray)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setMinimumSize(QtCore.QSize(50, 0))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_3.addWidget(self.label_2)
        self.unitsEdit = QtGui.QLineEdit(EditorQuantityArray)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
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
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.updateButton = QtGui.QPushButton(EditorQuantityArray)
        self.updateButton.setAutoDefault(True)
        self.updateButton.setDefault(True)
        self.updateButton.setObjectName(_fromUtf8("updateButton"))
        self.horizontalLayout_2.addWidget(self.updateButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        spacerItem2 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)

        self.retranslateUi(EditorQuantityArray)
        QtCore.QMetaObject.connectSlotsByName(EditorQuantityArray)

    def retranslateUi(self, EditorQuantityArray):
        self.label.setText(QtGui.QApplication.translate("EditorQuantityArray", "Values", None, QtGui.QApplication.UnicodeUTF8))
        self.valueEdit.setToolTip(QtGui.QApplication.translate("EditorQuantityArray", "<html><head/><body><p>Insert an array of values. The acceptable values are valid Python list expressions. Examples:</p>\n"
"<pre>[0.0, 0.1, 0.2, 0.3]</pre>\n"
"<pre>[ [0.0, 0.1, 0.2, 0.3],\n"
"  [0.0, 0.1, 0.2, 0.3], \n"
"  [0.0, 0.1, 0.2, 0.3] ]\n"
"</pre>\n"
"<p>etc.</p>\n"
"</body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("EditorQuantityArray", "Units", None, QtGui.QApplication.UnicodeUTF8))
        self.unitsEdit.setToolTip(QtGui.QApplication.translate("EditorQuantityArray", "<html><head/><body><p>Insert the unit expressions using daetools.pyUnits.unit objects using mathematical operators *, / and **. All base and derived SI units are allowed including the common prefixes (more information at <a href=\"https://en.wikipedia.org/wiki/International_System_of_Units#Units_and_prefixes\"><span style=\" text-decoration: underline; color:#0000ff;\">https://en.wikipedia.org/wiki/International_System_of_Units#Units_and_prefixes</span></a>). The expression entered is a valid Python code. Examples:</p>\n"
"<pre>m/kg</pre>\n"
"<pre>m / s**2</pre>\n"
"<pre>J / (kg*K)</pre>\n"
"</body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.updateButton.setToolTip(QtGui.QApplication.translate("EditorQuantityArray", "<html><head/><body><p>Update the value.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.updateButton.setText(QtGui.QApplication.translate("EditorQuantityArray", "Update", None, QtGui.QApplication.UnicodeUTF8))

