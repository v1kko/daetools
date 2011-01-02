#!/usr/bin/env python

"""********************************************************************************
                             opt_tutorial4.py
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
        
        self.t = daeDomain("time", self, "Number of chemical elements in the mixture")
        
        self.y       = daeVariable("y", typeNone, self, "Concentration")
        
        self.gamma_m = daeParameter("&gamma;_m", eReal, self, "")
        self.gamma_m.DistributeOnDomain(self.t)
        
        self.time    = daeParameter("time", eReal, self, "")
        self.time.DistributeOnDomain(self.t)

        
        self.theta1 = daeVariable("&theta;_1", typeNone, self, "Concentration")
        self.theta2 = daeVariable("&theta;_2", typeNone, self, "Concentration")
        self.theta3 = daeVariable("&theta;_3", typeNone, self, "Concentration")
        
        self.gamma1 = daeVariable("&gamma;_1", typeNone, self, "Concentration")
        self.gamma2 = daeVariable("&gamma;_2", typeNone, self, "Concentration")
        
    def DeclareEquations(self):
        eq = self.CreateEquation("&gamma;_1")
        eq.Residual = self.gamma1.dt() - self.theta1() * self.gamma1()
        
        eq = self.CreateEquation("&gamma;_2")
        eq.Residual = self.gamma2.dt() - self.theta1() * self.gamma1() + self.theta2() * self.gamma2()
        
class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("opt_tutorial4")
        self.m.Description = "Consider the concentration of tetracycline hydrochloride in blood serum. The tetracycline " \
                             "is administered to a subject orally, and the concentration of the tetracycline " \
                             "in the serum is measured. The biological system to be modeled will consist of two " \
                             "compartments: a gut compartment in which tetracycline is injected and a blood compartment " \
                             "that absorbs the tetracycline from the gut compartment for delivery to the " \
                             "body. Let gamma1(t) and gamma2(t) be the concentrations at time t in the gut and the serum, " \
                             "respectively. Let theta1 and theta2 be the transfer parameters."
          
    def SetUpParametersAndDomains(self):
        self.m.t.CreateArray(9)
        
        self.m.time.SetValue(0, 1)
        self.m.time.SetValue(1, 2)
        self.m.time.SetValue(2, 3)
        self.m.time.SetValue(3, 4)
        self.m.time.SetValue(4, 6)
        self.m.time.SetValue(5, 8)
        self.m.time.SetValue(6, 10)
        self.m.time.SetValue(7, 12)
        self.m.time.SetValue(8, 16)
        
        self.m.gamma_m.SetValue(0, 0.7)
        self.m.gamma_m.SetValue(1, 1.2)
        self.m.gamma_m.SetValue(2, 1.4)
        self.m.gamma_m.SetValue(3, 1.4)
        self.m.gamma_m.SetValue(4, 1.1)
        self.m.gamma_m.SetValue(5, 0.8)
        self.m.gamma_m.SetValue(6, 0.6)
        self.m.gamma_m.SetValue(7, 0.5)
        self.m.gamma_m.SetValue(8, 0.3)
    
    def SetUpVariables(self):
        self.m.y.AssignValue(0) 
        
        self.m.theta1.AssignValue(1) 
        self.m.theta2.AssignValue(1) 
        self.m.theta3.AssignValue(1)
        
        self.m.gamma1.SetInitialCondition(self.m.theta3.GetValue())
        self.m.gamma2.SetInitialCondition(0)
        
    def Run(self):
        print 'th1 =', self.m.theta1.GetValue(), 'th2 =', self.m.theta2.GetValue(), 'th3 =', self.m.theta3.GetValue()
        for i in range(self.m.t.NumberOfPoints):
            if i > 0:
                self.Reinitialize()
            time = self.IntegrateUntilTime(self.m.time.GetValue(i), eDoNotStopAtDiscontinuity)
            self.ReportData()
            self.m.y.ReAssignValue(self.m.gamma_m.GetValue(i))
            print 'time    =', self.m.time.GetValue(i)
            print 'y       =', self.m.y.GetValue()
            #print 'gamma_m =', self.m.gamma_m.GetValue(i)
            #print 'gamma2  =', self.m.gamma2.GetValue()

    def SetUpOptimization(self):
        # Set the objective function (min)
        self.ObjectiveFunction.Residual = Pow( self.m.y() - self.m.gamma2(), 2)
        
        # Set the optimization variables and their lower and upper bounds
        self.SetContinuousOptimizationVariable(self.m.theta1, 1e-6, 1,  0.1)
        self.SetContinuousOptimizationVariable(self.m.theta2, 1e-6, 1,  0.3)
        self.SetContinuousOptimizationVariable(self.m.theta3, 1e-6, 20, 10.0)


# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
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
    simulation   = simTutorial()
    optimization = daeOptimization()

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 16
    simulation.TimeHorizon = 16

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()


    ## Initialize the simulation
    #simulation.Initialize(daesolver, datareporter, log)

    ## Save the model report and the runtime model report 
    #simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    #simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    ## Solve at time=0 (initialization)
    #simulation.SolveInitial()

    ## Run
    #simulation.Run()
    #simulation.Finalize()



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
