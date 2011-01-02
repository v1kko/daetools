#!/usr/bin/env python

"""********************************************************************************
                             opt_tutorial3.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the 
terms of the GNU General Public License as published by the Free Software 
Foundation; either version 3 of the License, or (at your option) any later version.
The DAE Tools is distributed in the hope that it will be useful, but WITHOUT ANY 
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, write to the Free Software Foundation, Inc., 
59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
********************************************************************************"""

"""
"""

import sys
from daetools.pyDAE import *
from time import localtime, strftime

typeNone = daeVariableType("None", "-",  -1E20, 1E20,   1, 1e-6)

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)
        
        self.m = daeDomain("m", self, "Number of chemical elements in the mixture")
        self.n = daeDomain("n", self, "Number of compounds in the mixture")
        
        self.x = daeVariable("x", typeNone, self, "Number of moles for compound j")
        self.x.DistributeOnDomain(self.n)
        
        self.s = daeVariable("s", typeNone, self, "Total number of moles in the mixture")
        
        self.c = daeParameter("c", eReal, self, "")
        self.c.DistributeOnDomain(self.n)
        
        self.a = daeParameter("a", eReal, self, "Number of atoms of element i in a molecule of compound j")
        self.a.DistributeOnDomain(self.m)
        self.a.DistributeOnDomain(self.n)
        
        self.b  = daeParameter("b", eReal, self, "Atomic weight of element i in the mixture")
        self.b.DistributeOnDomain(self.m)
        
    def DeclareEquations(self):
        jr = daeIndexRange(self.n)
        
        eq = self.CreateEquation("s_equation")
        eq.Residual = self.s() - self.sum( self.x.array(jr) )
        
class simTutorial(daeSimulation):
    def __init__(self, **kwargs):
        daeSimulation.__init__(self)
        # Models/Simulations can have some input arguments if necessary, as it is shown in this tutorial.
        # This way the model depends only on parameters that it defines.
        # Here, for clarity, we use named arguments
        self.Nelements  = kwargs.get('Nelements')
        self.Ncompounds = kwargs.get('Ncompounds')
        self.aij        = kwargs.get('aij')
        self.bi         = kwargs.get('bi')
        self.cj         = kwargs.get('cj')
        
        self.m = modTutorial("opt_tutorial3")
        self.m.Description = "Bracken and McCormick (1968) problem.\n" \
                             "The problem is to determine the composition of a mixture of various chemicals that " \
                             "satisfy the mixture's chemical equilibrium state. The second law of thermodynamics " \
                             "implies that at a constant temperature and pressure, a mixture of chemicals satisfies " \
                             "its chemical equilibrium state when the free energy of the mixture is reduced to a " \
                             "minimum. Therefore, the composition of the chemicals satisfying its chemical equilibrium " \
                             "state can be found by minimizing the free energy of the mixture."
          
    def SetUpParametersAndDomains(self):
        self.m.m.CreateArray(self.Nelements)
        self.m.n.CreateArray(self.Ncompounds)

        for i in range(self.m.m.NumberOfPoints):
            self.m.b.SetValue(i, self.bi[i])

        for j in range(self.m.n.NumberOfPoints):
            self.m.c.SetValue(j, self.cj[j])
        
        for i in range(self.m.m.NumberOfPoints):
            for j in range(self.m.n.NumberOfPoints):
                self.m.a.SetValue(i, j, self.aij[i][j])
    
    def SetUpVariables(self):
        for j in range(self.m.n.NumberOfPoints):
            self.m.x.AssignValue(j, 0.1)
        
        #self.m.x.AssignValue(0, 0.040668) 
        #self.m.x.AssignValue(1, 0.147730) 
        #self.m.x.AssignValue(2, 0.783154) 
        #self.m.x.AssignValue(3, 0.001414) 
        #self.m.x.AssignValue(4, 0.485247) 
        #self.m.x.AssignValue(5, 0.000693) 
        #self.m.x.AssignValue(6, 0.027399) 
        #self.m.x.AssignValue(7, 0.017947)
        #self.m.x.AssignValue(8, 0.037314) 
        #self.m.x.AssignValue(9, 0.096871)
        
    def SetUpOptimization(self):
        jr = daeIndexRange(self.m.n)
        
        # Set the objective function (min)
        self.ObjectiveFunction.Residual = self.m.sum( self.m.x.array(jr) * (self.m.c.array(jr) + Log(self.m.x.array(jr) / self.m.s()) ) )
        
        # Set the constraints
        for i in range(self.m.m.NumberOfPoints):
            c = self.CreateEqualityConstraint(0, "Eq. Constraint" + str(i+1))
            c.Residual = self.m.sum( self.m.a.array(i, jr) * self.m.x.array(jr) ) - self.m.b(i)
            
        #c1 = self.CreateEqualityConstraint(0, "Eq. Constraint1")
        #c1.Residual = self.m.x(0) + 2*self.m.x(1) + 2*self.m.x(2) + self.m.x(5) + self.m.x(9) - 2
        
        #c2 = self.CreateEqualityConstraint(0, "Eq. Constraint2")
        #c2.Residual = self.m.x(3) + 2*self.m.x(4) + self.m.x(5) + self.m.x(6) - 1

        #c3 = self.CreateEqualityConstraint(0, "Eq. Constraint3")
        #c3.Residual = self.m.x(2) + self.m.x(6) + self.m.x(7) + 2*self.m.x(8) + self.m.x(9) - 1

        # Set the optimization variables and their lower and upper bounds
        for j in range(self.m.n.NumberOfPoints):
            self.SetContinuousOptimizationVariable(self.m.x(j), 1e-6, 1, 0.1)



Nelem = 3
Ncomp = 10
b  = [2, 1, 1]
c  = [-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.100, -10.708, -26.662, -22.179]
# A pythonic way of creating a two-dimensional list
a = [ [0 for col in range(Ncomp)] for row in range(Nelem) ]
a[0][:] = [1, 2, 2, 0, 0, 1, 0, 0, 0, 1]
a[1][:] = [0, 0, 0, 1, 2, 1, 1, 0, 0, 0]
a[2][:] = [0, 0, 1, 0, 0, 0, 1, 1, 2, 1]

# Use daeSimulator class
def guiRun(app):
    sim = simTutorial(Nelements=Nelem, Ncompounds=Ncomp, bi=b, cj=c, aij=a)
    opt = daeOptimization()
    nlp = daeBONMIN()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 1
    sim.TimeHorizon       = 5
    simulator = daeSimulator(app, simulation=sim, optimization=opt, nlpsolver=nlp)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    nlpsolver    = daeBONMIN()
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial(Nelements=Nelem, 
                               Ncompounds=Ncomp, 
                               aij=a,
                               bi=b, 
                               cj=c)
    optimization = daeOptimization()

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 1
    simulation.TimeHorizon = 5

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # Initialize the simulation
    optimization.Initialize(simulation, nlpsolver, daesolver, datareporter, log)

    nlpsolver.LoadOptionsFile("")
    #nlpsolver.SetOption('tol', 1e-7)
    #nlpsolver.SetOption('mu_strategy', 'adaptive')
    #nlpsolver.SetOption('mu_oracle', 'loqo')
    #nlpsolver.SetOption('mehrotra_algorithm', 'yes')
    nlpsolver.SetOption('expect_infeasible_problem', 'no')
    #nlpsolver.PrintOptions()
    #return
    
    # Save the model report and the runtime model report 
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Run
    optimization.Run()
    optimization.Finalize()
    
if __name__ == "__main__":
    runInGUI = True
    if len(sys.argv) > 1:
        if(sys.argv[1] == 'console'):
            runInGUI = False
    if runInGUI:
        from PyQt4 import QtCore, QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
    else:
        consoleRun()
