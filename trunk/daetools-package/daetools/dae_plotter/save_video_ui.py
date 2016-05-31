# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'save_video.ui'
#
# Created: Tue May 31 16:52:10 2016
#      by: PyQt4 UI code generator 4.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_SaveVideo(object):
    def setupUi(self, SaveVideo):
        SaveVideo.setObjectName(_fromUtf8("SaveVideo"))
        SaveVideo.resize(421, 261)
        SaveVideo.setSizeGripEnabled(True)
        SaveVideo.setModal(True)
        self.verticalLayout = QtGui.QVBoxLayout(SaveVideo)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.formLayout = QtGui.QFormLayout()
        self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.label1 = QtGui.QLabel(SaveVideo)
        self.label1.setObjectName(_fromUtf8("label1"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.label1)
        self.label2 = QtGui.QLabel(SaveVideo)
        self.label2.setObjectName(_fromUtf8("label2"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.label2)
        self.comboEncoders = QtGui.QComboBox(SaveVideo)
        self.comboEncoders.setObjectName(_fromUtf8("comboEncoders"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.comboEncoders)
        self.label3 = QtGui.QLabel(SaveVideo)
        self.label3.setObjectName(_fromUtf8("label3"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.LabelRole, self.label3)
        self.spinFPS = QtGui.QSpinBox(SaveVideo)
        self.spinFPS.setMinimum(1)
        self.spinFPS.setMaximum(10000)
        self.spinFPS.setProperty("value", 10)
        self.spinFPS.setObjectName(_fromUtf8("spinFPS"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.FieldRole, self.spinFPS)
        self.label4 = QtGui.QLabel(SaveVideo)
        self.label4.setObjectName(_fromUtf8("label4"))
        self.formLayout.setWidget(5, QtGui.QFormLayout.LabelRole, self.label4)
        self.lineeditExtraArgs = QtGui.QLineEdit(SaveVideo)
        self.lineeditExtraArgs.setText(_fromUtf8(""))
        self.lineeditExtraArgs.setObjectName(_fromUtf8("lineeditExtraArgs"))
        self.formLayout.setWidget(5, QtGui.QFormLayout.FieldRole, self.lineeditExtraArgs)
        self.label5 = QtGui.QLabel(SaveVideo)
        self.label5.setObjectName(_fromUtf8("label5"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.LabelRole, self.label5)
        self.spinBitrate = QtGui.QSpinBox(SaveVideo)
        self.spinBitrate.setMinimum(-1)
        self.spinBitrate.setMaximum(10000000)
        self.spinBitrate.setProperty("value", -1)
        self.spinBitrate.setObjectName(_fromUtf8("spinBitrate"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.FieldRole, self.spinBitrate)
        self.label6 = QtGui.QLabel(SaveVideo)
        self.label6.setObjectName(_fromUtf8("label6"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.label6)
        self.lineeditCodec = QtGui.QLineEdit(SaveVideo)
        self.lineeditCodec.setObjectName(_fromUtf8("lineeditCodec"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.lineeditCodec)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.lineeditFilename = QtGui.QLineEdit(SaveVideo)
        self.lineeditFilename.setEnabled(True)
        self.lineeditFilename.setText(_fromUtf8(""))
        self.lineeditFilename.setFrame(True)
        self.lineeditFilename.setObjectName(_fromUtf8("lineeditFilename"))
        self.horizontalLayout_2.addWidget(self.lineeditFilename)
        self.buttonFilename = QtGui.QPushButton(SaveVideo)
        self.buttonFilename.setObjectName(_fromUtf8("buttonFilename"))
        self.horizontalLayout_2.addWidget(self.buttonFilename)
        self.formLayout.setLayout(0, QtGui.QFormLayout.FieldRole, self.horizontalLayout_2)
        self.verticalLayout.addLayout(self.formLayout)
        self.buttonBox = QtGui.QDialogButtonBox(SaveVideo)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(SaveVideo)
        QtCore.QMetaObject.connectSlotsByName(SaveVideo)

    def retranslateUi(self, SaveVideo):
        SaveVideo.setWindowTitle(_translate("SaveVideo", "DAE Tools Simulator", None))
        self.label1.setText(_translate("SaveVideo", "Filename", None))
        self.label2.setText(_translate("SaveVideo", "Encoder", None))
        self.label3.setText(_translate("SaveVideo", "Framerate", None))
        self.label4.setText(_translate("SaveVideo", "Extra args", None))
        self.label5.setText(_translate("SaveVideo", "Bitrate", None))
        self.label6.setText(_translate("SaveVideo", "Codec", None))
        self.buttonFilename.setText(_translate("SaveVideo", "...", None))

