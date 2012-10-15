"""********************************************************************************
                             daeChooseVariable.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the 
terms of the GNU General Public License version 3 as published by the Free Software 
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
********************************************************************************"""
"""
October 2012 by Caleb Hattingh:
  - 3D plot bug fix when detecting free domains
  - code refactoring
"""

import sys
from os.path import join, realpath, dirname

try:
    import numpy
except ImportError:
    print '[daeChooseVariable]: Cannot load numpy module'

try:
    from PyQt4 import QtCore, QtGui
except ImportError:
    print '[daeChooseVariable]: Cannot load pyQt4 modules'

try:
    from daetools.pyDAE import *
    from choose_variable import Ui_ChooseVariable
    from table_widget import Ui_tableWidgetDialog
except ImportError:
    print '[daeChooseVariable]: Cannot load daetools modules'

class daeTableDialog(QtGui.QDialog):
    def __init__(self, parent):
        QtGui.QDialog.__init__(self, parent)
        self.ui = Ui_tableWidgetDialog()
        self.ui.setupUi(self)

images_dir = join(dirname(__file__), 'images')

def nameFormat(name):
    return name.replace('&', '').replace(';', '')

class daeChooseVariable(QtGui.QDialog):

    (plot2D, plot2DAnimated, plot3D) = range(0, 3)
    FREE_DOMAIN = -1
    LAST_TIME   = -2
    
    def __init__(self, processes, plotType):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_ChooseVariable()
        self.ui.setupUi(self)
        
        self.setWindowIcon(QtGui.QIcon(join(images_dir, 'daetools-48x48.png')))
        
        self.variable = None
        self.domainIndexes   = []
        self.domainPoints    = []
        self.plotType        = plotType
        self.processes       = processes
        
        # For convenience, store list of combo widgets
        self.domainCombos = [] 
        self.domainLabels = [] 
        for i in range(8):
            self.domainCombos.append(eval('self.ui.domain%dComboBox'%i))
            self.domainLabels.append(eval('self.ui.domain%dLabel'%i))

        self.hideAndClearAll()
        
        self.connect(self.ui.treeWidget,      QtCore.SIGNAL("itemSelectionChanged()"),   self.slotSelectionChanged)
        self.connect(self.ui.timeComboBox,    QtCore.SIGNAL("currentIndexChanged(int)"), self.slotCurrentIndexChanged)
        # Connect the combo widgets
        for cb in self.domainCombos:
            self.connect(cb, QtCore.SIGNAL("currentIndexChanged(int)"), self.slotCurrentIndexChanged)

        self.initTree(processes)
        
    def initTree(self, processes):
        for process in processes:
            self.addProcess(process)

    def addProcess(self, process):
        rootItem = QtGui.QTreeWidgetItem(self.ui.treeWidget)
        rootItem.setText(0, process.Name)

        variables = process.Variables
        for var in variables:
            currentItem = rootItem
            names = var.Name.split(".")

            var_path = names[:-1]

            var_name = nameFormat(names[-1])

            # First find the parent QTreeWidgetItem
            for item in var_path:
                name = nameFormat(item)
                # All names in the path has to be enclosed with brackets [] to distinguish between variables and ports/models
                name = '[' + name + ']'

                found = False
                for c in range(0, currentItem.childCount()):
                    child = currentItem.child(c)
                    cname = child.text(0)
                    if name == cname:
                        found = True
                        currentItem = currentItem.child(c)
                        break

                if found == False:
                    currentItem = QtGui.QTreeWidgetItem(currentItem)
                    currentItem.setText(0, name)

            # Now we have the parrent in the currentItem, so add the new item to it with the variable data
            varItem = QtGui.QTreeWidgetItem(currentItem)
            varData = QtCore.QVariant(var)
            varItem.setText(0, var_name)
            varItem.setData(0, QtCore.Qt.UserRole, varData)
  
    #@QtCore.pyqtSlot(int)
    def slotCurrentIndexChanged(self, index):
        self.domainIndexes = []
        if(self.ui.timeComboBox.isVisible()):
            i                = self.ui.timeComboBox.currentIndex()
            domain_index, ok = self.ui.timeComboBox.itemData(i).toInt()
            self.domainIndexes.append(domain_index)
        
        for cb in self.domainCombos:
            if cb.isVisible():
                i = cb.currentIndex()
                domain_index, ok = cb.itemData(i).toInt()
                self.domainIndexes.append(domain_index)
        
        self.domainPoints = []
        if(self.ui.timeComboBox.isVisible()):
            self.domainPoints.append(str(self.ui.timeComboBox.currentText()))

        for cb in self.domainCombos:
            if cb.isVisible():
                self.domainPoints.append(str(cb.currentText()))

        freeDomains = len([ind for ind in self.domainIndexes if ind == daeChooseVariable.FREE_DOMAIN])
        
        valid2D = (freeDomains==1) and (self.plotType in (daeChooseVariable.plot2D, daeChooseVariable.plot2DAnimated)) 
        valid3D = (freeDomains==2) and (self.plotType == daeChooseVariable.plot3D)
        self.ui.buttonOk.setEnabled(valid2D or valid3D)

    def getPlot2DData(self):
        return daeChooseVariable.get2DData(self.variable, self.domainIndexes, self.domainPoints)
    
    @staticmethod
    def get2DData(variable, domainIndexes, domainPoints):
        # Achtung, achtung!!
        # It is important to get TimeValues first since the reporter
        # might add more values to the data receiver (in the meantime)
        # and the size of the xPoints and yPoints arrays will not match
        times   = variable.TimeValues
        values  = variable.Values
        domains = variable.Domains

        xAxisLabel = ""
        
        yname = variable.Name
        yAxisLabel = nameFormat(yname)
        xPoints = []

        noTimePoints = len(times)

        # x axis points
        # Only one domain is free
        for i, domainIndex in enumerate(domainIndexes):
            if domainIndex == daeChooseVariable.FREE_DOMAIN:
                if i == 0:
                    xAxisLabel = "Time"
                    data = times[0:noTimePoints]
                else:
                    d = domains[i-1] # because Time is not in a domain list
                    xAxisLabel = nameFormat(d.Name.split(".")[-1])
                    data = d.Points[0:d.NumberOfPoints]
                xPoints.extend(data)
                break
                
        # y axis points
        t = []

        for i, domainIndex in enumerate(domainIndexes):
            if domainIndex == daeChooseVariable.FREE_DOMAIN:
                if i == 0: # Time points
                    t.append(slice(0, noTimePoints))
                else: # Other domain's points
                    d = domains[i-1]
                    t.append(slice(0, d.NumberOfPoints))
            elif domainIndex == daeChooseVariable.LAST_TIME:
                # Special case when time = "Last value" has been selected (animated plots)
                t.append(noTimePoints-1)
                # Update domain points with the last time
                domainPoints[i] = 'ct'
            else:
                t.append(domainIndex)

        yPoints = values[t]

        print noTimePoints
        print 'Number of x points = {0}'.format(len(xPoints))
        print 'Number of y points = {0}'.format(len(yPoints))

        return variable, domainIndexes, domainPoints, xAxisLabel, yAxisLabel, xPoints, yPoints, times[-1]

    def getPlot3DData(self):
        return daeChooseVariable.get3DData(self.variable, self.domainIndexes, self.domainPoints)
    
    @staticmethod
    def get3DData(variable, domainIndexes, domainPoints):
        # Achtung, achtung!!
        # It is important to get TimeValues first since the reporter
        # might add more values to the data receiver (in the meantime)
        # and the size of xPoints, yPoints and zPoints arrays will not match
        times   = variable.TimeValues
        values  = variable.Values
        domains = variable.Domains

        xPoints = []
        yPoints = []
        xAxisLabel = ""
        yAxisLabel = ""

        zname = variable.Name
        zAxisLabel = nameFormat(zname)

        # Find 2 domains that are FREE
        freeDomainIndexes = [i for i, d in enumerate(domainIndexes) if d == daeChooseVariable.FREE_DOMAIN]
        if len(freeDomainIndexes) != 2:
            return
        
        noTimePoints = len(times)

        # x axis
        nd = freeDomainIndexes[0]
        if nd == 0: # Time domain
            xAxisLabel = "Time"
            xPoints.extend(times[0 : noTimePoints])
        else: # Some other domain
            d = domains[nd-1] # because Time is not in a domain list
            names = d.Name.split(".")
            xname = names[len(names)-1]
            xAxisLabel = nameFormat(xname)
            xPoints.extend(d.Points[0 : d.NumberOfPoints])
            
        # y axis
        nd = freeDomainIndexes[1]
        if nd == 0: # Time domain
            yAxisLabel = "Time"
            yPoints.extend(times[0 : noTimePoints])
        else: # Some other domain
            d = domains[nd-1] # because Time is not in a domain list
            names = d.Name.split(".")
            yname = names[len(names)-1]
            yAxisLabel = nameFormat(yname)
            yPoints.extend(d.Points[0 : d.NumberOfPoints])
            #for k in range(0, d.NumberOfPoints):
            #    yPoints.append(d[k]) 

        # z axis
        t = []
        for i in range(0, len(domainIndexes)):
            domainIndex = domainIndexes[i]
            if domainIndex == daeChooseVariable.FREE_DOMAIN:
                if i == 0: # Time domain
                    t.append(slice(0, noTimePoints))
                else:
                    d = domains[i-1] # because Time is not in a domain list
                    t.append(slice(0, d.NumberOfPoints))
            elif domainIndex == daeChooseVariable.LAST_TIME:
                t.append(noTimePoints-1)
                # Update domain points with the last time
                domainPoints[i] = 'ct'
            else:
                t.append(domainIndex)

        zPoints = values[t] 
        return variable, domainIndexes, domainPoints, xAxisLabel, yAxisLabel, zAxisLabel, xPoints, yPoints, zPoints, times[-1]

    #@QtCore.pyqtSlot()
    def slotSelectionChanged(self):
        self.variable = None
        self.hideAndClearAll()

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
        
        domains = var.Domains
        times   = var.TimeValues
        values  = var.Values
        
        self.variable = var
        
        self.enableControls(len(domains))
        self.insertTimeValues(times)
        for i,d in enumerate(domains):
            self.insertDomainValues(i+1, d)
            
    def insertTimeValues(self, times):
        label, comboBox    = self.getComboBoxAndLabel(0)
        label.setText("Time")
        comboBox.addItem("*", QtCore.QVariant(daeChooseVariable.FREE_DOMAIN))
        if self.plotType == daeChooseVariable.plot2DAnimated:
            comboBox.addItem("Current time", QtCore.QVariant(daeChooseVariable.LAST_TIME))
        for i, time in enumerate(times):
            comboBox.addItem(str(time), QtCore.QVariant(i))

    def insertDomainValues(self, n, domain):
        label, comboBox    = self.getComboBoxAndLabel(n)
        names = domain.Name.split(".")
        label.setText(names[len(names)-1])
        comboBox.addItem("*", QtCore.QVariant(daeChooseVariable.FREE_DOMAIN))
        for i in range(0, domain.NumberOfPoints):
            comboBox.addItem(str(domain[i]), QtCore.QVariant(i))
            
    def getComboBoxAndLabel(self, index):
        if index == 0:
            return self.ui.timeLabel, self.ui.timeComboBox
        else:
            try:
                return self.domainLabels[index-1], self.domainCombos[index-1]
            except IndexError:
                return None, None

    def enableControls(self, n):
        self.ui.timeLabel.setVisible(True)
        self.ui.timeComboBox.setVisible(True)
        for i, cb, lab in zip(range(n), self.domainCombos, self.domainLabels):
            lab.setVisible(True)
            cb.setVisible(True)
            
    def hideAndClearAll(self):
        del self.domainIndexes[:]
        self.ui.buttonOk.setEnabled(False)
        
        self.ui.timeLabel.setVisible(False)
        self.ui.timeComboBox.setVisible(False)
        for combo, label in zip(self.domainCombos, self.domainLabels):
            label.setVisible(False)
            combo.setVisible(False)
            combo.clear()
