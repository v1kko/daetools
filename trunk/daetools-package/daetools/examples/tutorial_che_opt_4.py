#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_che_opt_4.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic, 2016
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
************************************************************************************
"""
__doc__ = """
COPS test 12: Catalytic Cracking of Gas Oil.

Determine the reaction coefficients for the catalytic cracking of gas oil into gas and other
byproducts.

Reference: Benchmarking Optimization Software with COPS 3.0, Mathematics and Computer
Science Division, Argonne National Laboratory, Technical Report ANL/MCS-273, 2004.
`PDF <http://www.mcs.anl.gov/~more/cops/cops3.pdf>`_

Experimental data generated following the procedure described in the COPS test.

Run options:

- Simulation with optimal parameters: python tutorial_che_opt_4.py simulation
- Parameter estimation console run:   python tutorial_che_opt_4.py console
- Parameter estimation GUI run:       python tutorial_che_opt_4.py gui

Currently, the parameter estimation results are (solver options/scaling should be tuned):

.. code-block:: none

   Fobj = 4.841995e-3
   p1   = 10.95289
   p2   =  7.70601
   p3   =  2.89625

The concentration plots (for optimal 'p' from the literature):

.. image:: _static/tutorial_che_opt_4-results.png
   :width: 500px
"""

import sys
from time import localtime, strftime
from daetools.pyDAE import *
from daetools.solvers.trilinos import pyTrilinos
from daetools.solvers.ipopt import pyIPOPT
from pyUnits import m, kg, s, K, Pa, mol, J, W, kJ, hour, l

y_t  = daeVariableType("y_t",  unit(), -1.0e+20, 1.0e+20, 0.0, 1e-07)
L2_t = daeVariableType("L2_t", unit(), -1.0e+20, 1.0e+20, 0.0, 1e-07)

#########################################################
#             Catalytic Cracking of Gas Oil
#########################################################
# Mathematical model
class modOilCracking(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        # Reaction coefficients
        self.p1 = daeVariable("p1",  no_t, self, "Reaction coefficient 1")
        self.p2 = daeVariable("p2",  no_t, self, "Reaction coefficient 2")
        self.p3 = daeVariable("p3",  no_t, self, "Reaction coefficient 3")

        # State variables
        self.y1 = daeVariable("y1", y_t, self, "1 concentration")
        self.y2 = daeVariable("y2", y_t, self, "2 concentration")

    def DeclareEquations(self):
        # Create adouble objects to make equations more readable
        y1 = self.y1()
        y2 = self.y2()
        p1 = self.p1()
        p2 = self.p2()
        p3 = self.p3()

        # Derivatives
        dy1_dt = self.y1.dt()
        dy2_dt = self.y2.dt()

        # y1
        eq = self.CreateEquation("y1", "")
        eq.Residual = dy1_dt + (p1+p3)*(y1**2)
        eq.CheckUnitsConsistency = False

        # y2
        eq = self.CreateEquation("y2", "")
        eq.Residual = dy2_dt - (p1*(y1**2) - p2*y2)
        eq.CheckUnitsConsistency = False

# Simulation (can be run independently from optimisation)
class simOilCracking(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modOilCracking("tutorial_che_opt_4")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        # The reaction coefficients below are optimal results found in the literature.
        # They should produce L2 norm of 4.12164e-03.
        self.m.p1.AssignValue(12)
        self.m.p2.AssignValue(8)
        self.m.p3.AssignValue(2)

        self.m.y1.SetInitialCondition(y1_t0)
        self.m.y2.SetInitialCondition(y2_t0)

#########################################################
#               Parameter Estimation Part
#########################################################
# We need some additional variables to determine reaction coefficients.
# Derive a new class from modOilCracking and add extra data.
# Nota Bene:
#   modOilCracking_Opt inherits all parameters/variables from the base class.
class modOilCracking_Opt(modOilCracking):
    def __init__(self, Name, Parent = None, Description = ""):
        modOilCracking.__init__(self, Name, Parent, Description)

        # Observed values at the specific time interval
        self.y1_obs = daeVariable("y1_obs", no_t, self, "Observed value 1 at the specified time interval")
        self.y2_obs = daeVariable("y2_obs", no_t, self, "Observed value 2 at the specified time interval")

        # This L2 norm sums all L2 norms in the previous time intervals
        self.L2      = daeVariable("L2",      L2_t, self, "Current L2 norm: ||yi(t) - yi_obs(t)||^2")
        self.L2_prev = daeVariable("L2_prev", L2_t, self, "L2 norm in previous time intrvals")

    def DeclareEquations(self):
        modOilCracking.DeclareEquations(self)

        # L2-norm ||yi(t) - yi_obs(t)||^2
        # L2 norm is a sum of the L2 norm in the previous time steps (L2_prev)
        # and the current norm: s1 + s2.
        # L2_prev will be reset after every time interval where we have observed values.
        s1 = (self.y1() - self.y1_obs())**2
        s2 = (self.y2() - self.y2_obs())**2
        eq = self.CreateEquation("L2", "")
        eq.Residual = self.L2() - (self.L2_prev() + s1 + s2)
        eq.CheckUnitsConsistency = False

# Simulation class that will be used by the optimisation.
class simOilCracking_opt(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modOilCracking_Opt("tutorial_che_opt_4")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        # modOilCracking part
        self.m.p1.AssignValue(0)
        self.m.p2.AssignValue(0)
        self.m.p3.AssignValue(0)

        self.m.y1.SetInitialCondition(y1_t0)
        self.m.y2.SetInitialCondition(y2_t0)

        # Initialise variables required for parameter estimation.
        # Notate bene:
        #   Observed values should match initial conditions at t = 0
        #   L2_prev should be 0.0 initially
        self.m.y1_obs.AssignValue(y1_t0)
        self.m.y2_obs.AssignValue(y2_t0)
        self.m.L2_prev.AssignValue(0.0)

    def Run(self):
        for t, tn in enumerate(times):
            # Reset L2_prev value to the current L2
            if t == 0:
                self.m.L2_prev.ReAssignValue(0.0)
            else:
                L2 = self.m.L2.GetValue()
                self.m.L2_prev.ReAssignValue(L2)

            # Reset observed values to match the current interval end time
            self.m.y1_obs.ReAssignValue(y1_obs[t])
            self.m.y2_obs.ReAssignValue(y2_obs[t])

            # Reinitialise the DAE system after all changes made above
            self.Reinitialize()

            # Integrate, report data and set progress
            self.Log.Message('Integrating from %f to %f ...' % (self.CurrentTime, tn), 0)
            self.IntegrateUntilTime(tn, eDoNotStopAtDiscontinuity)
            self.ReportData(self.CurrentTime)
            self.Log.SetProgress(int(100.0 * self.CurrentTime/self.TimeHorizon))

    def SetUpOptimization(self):
        # Minimise L2-norm ||yi(t) - yi_obs(t)||^2
        self.ObjectiveFunction.Residual = self.m.L2()

        p_lb   =  0.0
        p_ub   = 50.0
        p_init =  0.0

        p1 = self.SetContinuousOptimizationVariable(self.m.p1, p_lb, p_ub, p_init)
        p2 = self.SetContinuousOptimizationVariable(self.m.p2, p_lb, p_ub, p_init)
        p3 = self.SetContinuousOptimizationVariable(self.m.p3, p_lb, p_ub, p_init)

# Experimental data (20 measurements, skip t=0) generated by the simulation below
times  = numpy.array([0.050000, 0.100000, 0.150000, 0.200000, 0.250000, 0.300000,
                      0.350000, 0.400000, 0.450000, 0.500000, 0.550000, 0.600000, 0.650000,
                      0.700000, 0.750000, 0.800000, 0.850000, 0.900000, 0.950000, 1.000000])
y1_obs = numpy.array([0.539650, 0.436582, 0.335315, 0.260760, 0.214197, 0.175340,
                      0.157290, 0.156552, 0.131268, 0.113094, 0.114047, 0.102947, 0.095513,
                      0.094655, 0.081459, 0.083441, 0.077560, 0.066267, 0.072420, 0.067343])
y2_obs = numpy.array([0.277036, 0.298480, 0.269163, 0.209315, 0.176883, 0.135813,
                      0.115819, 0.085196, 0.073238, 0.051577, 0.040534, 0.036138, 0.028266,
                      0.022489, 0.019750, 0.016626, 0.013837, 0.011396, 0.010749, 0.009493])
# Initial conditions
y1_t0 = 1.0
y2_t0 = 0.0

# Use daeSimulator class
def guiRun(app):
    sim = simOilCracking_opt()
    opt = daeOptimization()
    nlp = pyIPOPT.daeIPOPT()
    sim.m.SetReportingOn(True)
    sim.ReportingTimes = times.tolist()
    simulator = daeSimulator(app, simulation = sim,
                                  optimization = opt,
                                  nlpsolver = nlp,
                                  nlpsolver_setoptions_fn = setOptions)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    nlpsolver    = pyIPOPT.daeIPOPT()
    datareporter = daeTCPIPDataReporter()
    simulation   = simOilCracking_opt()
    optimization = daeOptimization()
    lasolver     = pyTrilinos.daeCreateTrilinosSolver("Amesos_Klu", "")
    daesolver.SetLASolver(lasolver)

    daesolver.RelativeTolerance = 1e-6

    # Do no print progress
    log.PrintProgress = True
    
    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingTimes = times.tolist()

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # Initialize the optimization
    optimization.Initialize(simulation, nlpsolver, daesolver, datareporter, log)

    # Achtung! Achtung! NLP solver options can only be set after optimization.Initialize()
    # Otherwise seg. fault occurs for some reasons.
    nlpsolver.SetOption('print_level', 5)
    nlpsolver.SetOption('tol', 1e-6)
    #nlpsolver.SetOption('mu_strategy', 'adaptive')
    nlpsolver.SetOption('obj_scaling_factor', 10.0)
    nlpsolver.SetOption('nlp_scaling_method', 'none') #'user-scaling')

    # Run
    optimization.Run()
    optimization.Finalize()

def consoleSimulation():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeDelegateDataReporter()
    simulation   = simOilCracking()

    dr_tcpip     = daeTCPIPDataReporter()
    dr_data      = daeNoOpDataReporter()
    datareporter.AddDataReporter(dr_tcpip)
    datareporter.AddDataReporter(dr_data)

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 1.0/20
    simulation.TimeHorizon       = 1.0

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(dr_tcpip.Connect("", simName) == False):
        sys.exit()

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)

    # Save the model report and the runtime model report
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    # Run
    simulation.Run()
    simulation.Finalize()

    dvars = dr_data.Process.dictVariables
    # Skip the
    ts = dvars['tutorial_che_opt_4.y1'].TimeValues[1:]
    y1 = dvars['tutorial_che_opt_4.y1'].Values[1:]
    y2 = dvars['tutorial_che_opt_4.y2'].Values[1:]
    nt = len(ts)
    y1_exp = numpy.array([y + numpy.random.uniform(-y*0.1, y*0.1) for y in y1])
    y2_exp = numpy.array([y + numpy.random.uniform(-y*0.1, y*0.1) for y in y2])

    float_formatter = lambda x: "%.6f" % x
    numpy.set_printoptions(formatter={'float_kind':float_formatter})
    print('times  = numpy.%s' % repr(ts))
    print('y1     = numpy.%s' % repr(y1))
    print('y1_obs = numpy.%s' % repr(y1_exp))
    print('y2     = numpy.%s' % repr(y2))
    print('y2_obs = numpy.%s' % repr(y2_exp))

if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        consoleRun()
    elif len(sys.argv) > 1 and (sys.argv[1] == 'simulation'):
        consoleSimulation()
    else:
        from PyQt4 import QtCore, QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
