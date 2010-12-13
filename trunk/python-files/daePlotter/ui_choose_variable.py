import sys
from PyQt4 import QtCore, QtGui
from choose_variable import Ui_ChooseVariable

class ImageDialog(QtGui.QDialog):
	def __init__(self, process):
		QtGui.QDialog.__init__(self)
		self.ui = Ui_ChooseVariable()
		self.ui.setupUi(self)
		self.process = process
		#hideAndClearAll()
		#enableButtons()
		self.connect(self.ui.treeWidget, QtCore.SIGNAL("itemSelectionChanged()"), self, QtCore.SLOT("slotSelectionChanged()"))
		self.connect(self.ui.timeComboBox,    QtCore.SIGNAL("currentIndexChanged(int)"), self, QtCore.SLOT("slotCurrentIndexChanged(int)"))
		self.connect(self.ui.domain0ComboBox, QtCore.SIGNAL("currentIndexChanged(int)"), self, QtCore.SLOT("slotCurrentIndexChanged(int)"))
		self.connect(self.ui.domain1ComboBox, QtCore.SIGNAL("currentIndexChanged(int)"), self, QtCore.SLOT("slotCurrentIndexChanged(int)"))
		self.connect(self.ui.domain2ComboBox, QtCore.SIGNAL("currentIndexChanged(int)"), self, QtCore.SLOT("slotCurrentIndexChanged(int)"))
		self.connect(self.ui.domain3ComboBox, QtCore.SIGNAL("currentIndexChanged(int)"), self, QtCore.SLOT("slotCurrentIndexChanged(int)"))
		self.connect(self.ui.domain4ComboBox, QtCore.SIGNAL("currentIndexChanged(int)"), self, QtCore.SLOT("slotCurrentIndexChanged(int)"))

		rootItem = QtGui.QTreeWidgetItem(self.ui.treeWidget)
		rootItem.setText(0, self.process.Name)

		variables = self.process.GetVariables()
		for var in variables:
			currentItem = rootItem
			names = var.Name.split(".")
			for name in names:
				found = False
				for c in range(0, currentItem.childCount()):
					cname = currentItem.child[c].text(0)
					if name == cname:
						found = True
						#print "found"
						currentItem = currentItem.child[c]
						break

				if found == False:
					#print "not found"
					currentItem = QtGui.QTreeWidgetItem(currentItem)
					currentItem.setText(0, name)

			varData = QtCore.QVariant(var)
			currentItem.setData(0, QtCore.Qt.UserRole, varData)

	#@QtCore.pyqtSlot(int)
	def slotCurrentIndexChanged(self, index):
		pass

	#@QtCore.pyqtSlot()
	def slotSelectionChanged(self):
		items = self.ui.treeWidget.selectedItems()
		if len(items) != 1:
			return
		selItem = items[0]

		varData = selItem.data(0, QtCore.Qt.UserRole)
		if varData.isNull():
			return
		var = varData.toPyObject()
		if var == None:
			return
		#print var.Name
		domains = var.GetDomains()
		times = var.GetTimeValues()
		values = var.GetValues()
		dims = values.shape
                QtGui.QMessageBox.information(self, var.Name, str(values[:,0,0]))
		if len(domains) == 0: # Only time
			self.ui.timeLabel.setText("Time")

		elif len(domains) == 2:
			self.ui.timeLabel.setText("Time")
			self.ui.domain0Label.setText(domains[0].Name)
			self.ui.domain1Label.setText(domains[1].Name)
			self.ui.timeComboBox.clear()
			self.ui.domain0ComboBox.clear()
			self.ui.domain1ComboBox.clear()
			for i in range(0, len(times)):
				self.ui.timeComboBox.addItem(str(times[i]))
			for i in range(0, domains[0].NumberOfPoints):
				self.ui.domain0ComboBox.addItem(str(domains[0][i]))
			for i in range(0, domains[1].NumberOfPoints):
				self.ui.domain1ComboBox.addItem(str(domains[1][i]))
		else:
			pass


