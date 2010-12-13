"""********************************************************************************
                            Membrane.py
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

import sys
from daetools.pyDAE import *
from time import localtime, strftime

typeNone         = daeVariableType("None",         "-", -1E20, +1E20,     0, 1e-5)
typeFraction     = daeVariableType("Fraction",     "-",     0,     1,   0.1, 1e-5)
typePermeability = daeVariableType("Permeability", "-",     0,   100, 1e-10, 1e-5)
typeFlowrate     = daeVariableType("Flowrate",     "-",     0,   1E6,   1E4, 1e-5)
typeArea         = daeVariableType("Area",         "-",     0,   1E6,   1E4, 1e-5)
typeLength       = daeVariableType("Length",       "-",     0,  1E10,     1, 1e-5)
typePressure     = daeVariableType("Pressure",     "-",     0,  1E10,     1, 1e-5)

class modMembrane(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.Nc = daeDomain("N_c", self)

        self.xf = daeVariable("x_f", typeFraction, self)
        self.xf.DistributeOnDomain(self.Nc)
        
        self.x0 = daeVariable("x_0", typeFraction, self)
        self.x0.DistributeOnDomain(self.Nc)
        
        self.yp = daeVariable("y_p", typeFraction, self)
        self.yp.DistributeOnDomain(self.Nc)
        
        self.Perm = daeVariable("P_erm", typePermeability, self)
        self.Perm.DistributeOnDomain(self.Nc)

        self.L0 = daeVariable("L_0", typeFlowrate, self)
        self.Vp = daeVariable("V_p", typeFlowrate, self)
        self.Lf = daeVariable("L_f", typeFlowrate, self)
        
        self.Am    = daeVariable("A_m", typeArea, self)
        self.t     = daeVariable("&delta;",   typeLength, self)
        self.theta = daeVariable("&theta;",  typeFraction, self)

        self.Ph = daeVariable("P_h", typePressure, self)
        self.Pl = daeVariable("P_l", typePressure, self)
        
    def DeclareEquations(self):
        eq = self.CreateEquation("x_f")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed)
        eq.Residual = self.Vp() * self.yp(i) - (self.Perm(i) * self.Am() / self.t()) * (self.Ph() * self.x0(i) - self.Pl() * self.yp(i))

        eq = self.CreateEquation("x_0")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed)
        eq.Residual = self.x0(i) - self.xf(i) / (1 - self.theta()) + self.yp(i) * self.theta() / (1 - self.theta())

        eq = self.CreateEquation("sum_yp")
        r = daeIndexRange(self.Nc)
        eq.Residual = self.sum(self.yp.array(r)) - 1
        
        eq = self.CreateEquation("&Theta;")
        eq.Residual = self.theta() - self.Vp() / self.Lf()
        
        eq = self.CreateEquation("L_f")
        eq.Residual = self.Lf() - self.L0() - self.Vp()
        
class simMembrane(daeDynamicSimulation):
    def __init__(self):
        daeDynamicSimulation.__init__(self)
        self.m = modMembrane("Membrane")
          
    def SetUpParametersAndDomains(self):
        self.m.Nc.CreateArray(3)

    def SetUpVariables(self):
        self.m.Perm.AssignValue(0, 200E-10)
        self.m.Perm.AssignValue(1, 50E-10)
        self.m.Perm.AssignValue(2, 25E-10)

        self.m.xf.AssignValue(0, 0.25)
        self.m.xf.AssignValue(1, 0.55)
        self.m.xf.AssignValue(2, 0.20)

        self.m.Lf.AssignValue(1E4)
        self.m.theta.AssignValue(0.25)
        self.m.t.AssignValue(2.54E-3)
        self.m.Ph.AssignValue(300)
        self.m.Pl.AssignValue(30)

        #self.m.Am.SetInitialGuess(1E6)
        
        #self.m.yp.SetInitialGuess(0, 0.4555)
        #self.m.yp.SetInitialGuess(1, 0.4410)
        #self.m.yp.SetInitialGuess(2, 0.0922)
        
        #self.m.x0.SetInitialGuess(0, 0.18)
        #self.m.x0.SetInitialGuess(1, 0.58)
        #self.m.x0.SetInitialGuess(2, 0.23)

# Create Log, Solver, DataReporter and Simulation object
log          = daePythonStdOutLog()
solver       = daeIDASolver()
datareporter = daeTCPIPDataReporter()
simulation   = simMembrane()

# Enable reporting of all variables
simulation.m.SetReportingOn(True)

# Set the time horizon and the reporting interval
simulation.ReportingInterval = 1
simulation.TimeHorizon = 1

# Connect data reporter
simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
if(datareporter.Connect("", simName) == False):
    sys.exit()

simulation.Initialize(solver, datareporter, log)
daeSaveModel(simulation.m, simulation.m.Name + ".xml")

# Solve at time=0 (initialization)
simulation.SolveInitial()
# Run
simulation.Run()

