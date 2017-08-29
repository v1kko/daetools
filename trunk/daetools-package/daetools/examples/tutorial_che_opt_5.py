#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_che_opt_5.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic
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
COPS test 13: Methanol to Hydrocarbons.

Determine the reaction coefficients for the conversion of methanol into various hydrocarbons.

Reference: Benchmarking Optimization Software with COPS 3.0, Mathematics and Computer
Science Division, Argonne National Laboratory, Technical Report ANL/MCS-273, 2004.
`PDF <http://www.mcs.anl.gov/~more/cops/cops3.pdf>`_

Experimental data generated following the procedure described in the COPS test.

Run options:

- Simulation with optimal parameters: python tutorial_che_opt_5.py simulation
- Parameter estimation console run:   python tutorial_che_opt_5.py console
- Parameter estimation GUI run:       python tutorial_che_opt_5.py gui

Currently, the parameter estimation results are (solver options/scaling should be tuned):

.. code-block:: none

   Fobj = 1.274997e-2
   p1 = 2.641769
   p2 = 1.466245
   p3 = 1.884254
   p4 = 1.023885
   p5 = 0.471067

The concentration plots (for optimal 'p' from the literature):

.. image:: _static/tutorial_che_opt_5-results.png
   :width: 500px

The concentration plots (for optimal 'p' from this optimisation):

.. image:: _static/tutorial_che_opt_5-results2.png
   :width: 500px
"""

import sys
from time import localtime, strftime
from daetools.pyDAE import *
from daetools.solvers.trilinos import pyTrilinos
from daetools.solvers.nlopt import pyNLOPT
from pyUnits import m, kg, s, K, Pa, mol, J, W, kJ, hour, l

y_t  = daeVariableType("y_t",  unit(), -1.0e+20, 1.0e+20, 0.0, 1e-07)
L2_t = daeVariableType("L2_t", unit(), -1.0e+20, 1.0e+20, 0.0, 1e-07)

#########################################################
#             Methanol to Hydrocarbons
#########################################################
# Mathematical model
class modMethanol2Hydrocarbons(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        # Reaction coefficients
        self.p1 = daeVariable("p1",  no_t, self, "Reaction coefficient 1")
        self.p2 = daeVariable("p2",  no_t, self, "Reaction coefficient 2")
        self.p3 = daeVariable("p3",  no_t, self, "Reaction coefficient 3")
        self.p4 = daeVariable("p4",  no_t, self, "Reaction coefficient 4")
        self.p5 = daeVariable("p5",  no_t, self, "Reaction coefficient 5")

        # State variables
        self.y1 = daeVariable("y1", y_t, self, "1 concentration")
        self.y2 = daeVariable("y2", y_t, self, "2 concentration")
        self.y3 = daeVariable("y3", y_t, self, "3 concentration")

    def DeclareEquations(self):
        # Create adouble objects to make equations more readable
        y1 = self.y1()
        y2 = self.y2()
        y3 = self.y3()
        p1 = self.p1()
        p2 = self.p2()
        p3 = self.p3()
        p4 = self.p4()
        p5 = self.p5()

        # Derivatives
        dy1_dt = self.y1.dt()
        dy2_dt = self.y2.dt()
        dy3_dt = self.y3.dt()

        # y1
        eq = self.CreateEquation("y1", "")
        eq.Residual = dy1_dt + (2*p1 - p1*y2/((p2+p5)*y1+y2+1e-10) + p3 + p4)*y1
        eq.CheckUnitsConsistency = False

        # y2
        eq = self.CreateEquation("y2", "")
        eq.Residual = dy2_dt - (p1*y1*(p2*y1-y2)/((p2+p5)*y1+y2+1e-10) + p3*y1)
        eq.CheckUnitsConsistency = False

        # y3
        eq = self.CreateEquation("y3", "")
        eq.Residual = dy3_dt - (p1*y1*(y2+p5*y1)/((p2+p5)*y1+y2+1e-10) + p4*y1)
        eq.CheckUnitsConsistency = False

# Simulation (can be run independently from optimisation)
class simMethanol2Hydrocarbons(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modMethanol2Hydrocarbons("tutorial_che_opt_5")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        # The reaction coefficients below are used to generate the experimental data.
        # The noise
        self.m.p1.AssignValue(2.69)
        self.m.p2.AssignValue(0.5)
        self.m.p3.AssignValue(3.02)
        self.m.p4.AssignValue(0.5)
        self.m.p5.AssignValue(0.5)

        self.m.y1.SetInitialCondition(y1_t0)
        self.m.y2.SetInitialCondition(y2_t0)
        self.m.y3.SetInitialCondition(y3_t0)

#########################################################
#               Parameter Estimation Part
#########################################################
# We need some additional variables to determine reaction coefficients.
# Derive a new class from modMethanol2Hydrocarbons and add extra data.
# Nota Bene:
#   modMethanol2Hydrocarbons_Opt inherits all parameters/variables from the base class.
class modMethanol2Hydrocarbons_Opt(modMethanol2Hydrocarbons):
    def __init__(self, Name, Parent = None, Description = ""):
        modMethanol2Hydrocarbons.__init__(self, Name, Parent, Description)

        # Observed values at the specific time interval
        self.y1_obs = daeVariable("y1_obs", no_t, self, "Observed value 1 at the specified time interval")
        self.y2_obs = daeVariable("y2_obs", no_t, self, "Observed value 2 at the specified time interval")
        self.y3_obs = daeVariable("y3_obs", no_t, self, "Observed value 3 at the specified time interval")

        # This L2 norm sums all L2 norms in the previous time intervals
        self.L2      = daeVariable("L2",      L2_t, self, "Current L2 norm: ||yi(t) - yi_obs(t)||^2")
        self.L2_prev = daeVariable("L2_prev", L2_t, self, "L2 norm in previous time intrvals")

    def DeclareEquations(self):
        modMethanol2Hydrocarbons.DeclareEquations(self)

        # L2-norm ||yi(t) - yi_obs(t)||^2
        # L2 norm is a sum of the L2 norm in the previous time steps (L2_prev)
        # and the current norm: s1 + s2.
        # L2_prev will be reset after every time interval where we have observed values.
        s1 = (self.y1() - self.y1_obs())**2
        s2 = (self.y2() - self.y2_obs())**2
        s3 = (self.y3() - self.y3_obs())**2
        eq = self.CreateEquation("L2", "")
        eq.Residual = self.L2() - (self.L2_prev() + s1 + s2 + s3)
        eq.CheckUnitsConsistency = False

# Simulation class that will be used by the optimisation.
class simMethanol2Hydrocarbons_opt(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modMethanol2Hydrocarbons_Opt("tutorial_che_opt_5")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        # modMethanol2Hydrocarbons part
        self.m.p1.AssignValue(2.69)
        self.m.p2.AssignValue(0.5)
        self.m.p3.AssignValue(3.02)
        self.m.p4.AssignValue(0.5)
        self.m.p5.AssignValue(0.5)

        self.m.y1.SetInitialCondition(y1_t0)
        self.m.y2.SetInitialCondition(y2_t0)
        self.m.y3.SetInitialCondition(y3_t0)

        # Initialise variables required for parameter estimation.
        # Notate bene:
        #   Observed values should match initial conditions at t = 0
        #   L2_prev should be 0.0 initially
        self.m.y1_obs.AssignValue(y1_t0)
        self.m.y2_obs.AssignValue(y2_t0)
        self.m.y3_obs.AssignValue(y3_t0)
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
            self.m.y3_obs.ReAssignValue(y3_obs[t])

            # Reinitialise the DAE system after all changes made above
            self.Reinitialize()

            # Integrate, report data and set progress
            self.Log.Message('Integrating from %f to %f ...' % (self.CurrentTime, tn), 0)
            self.IntegrateUntilTime(tn, eDoNotStopAtDiscontinuity)
            self.ReportData(self.CurrentTime)
            self.Log.SetProgress(int(100.0 * self.CurrentTime/self.TimeHorizon))

    def SetUpOptimization(self):
        # Minimise L2-norm ||yi(t) - yi_obs(t)||^2
        self.ObjectiveFunction.Residual = self.m.L2() * 1e4 # scale Fobj
        self.ObjectiveFunction.AbsTolerance = 1e-6

        p_lb   =  0.0
        p_ub   = 20.0
        p_init =  1.0

        p1 = self.SetContinuousOptimizationVariable(self.m.p1, p_lb, p_ub, p_init)
        p2 = self.SetContinuousOptimizationVariable(self.m.p2, p_lb, p_ub, p_init)
        p3 = self.SetContinuousOptimizationVariable(self.m.p3, p_lb, p_ub, p_init)
        p4 = self.SetContinuousOptimizationVariable(self.m.p4, p_lb, p_ub, p_init)
        p5 = self.SetContinuousOptimizationVariable(self.m.p5, p_lb, p_ub, p_init)

# Experimental data (20 measurements, skip t=0) generated by the simulation below
times  = numpy.array([0.071875, 0.143750, 0.215625, 0.287500, 0.359375, 0.431250,
                      0.503125, 0.575000, 0.646875, 0.718750, 0.790625, 0.862500,
                      0.934375, 1.006250, 1.078125, 1.150000])
y1_obs = numpy.array([0.552208, 0.300598, 0.196879, 0.101175, 0.065684, 0.045096,
                      0.028880, 0.018433, 0.011509, 0.006215, 0.004278, 0.002698,
                      0.001944, 0.001116, 0.000732, 0.000426])
y2_obs = numpy.array([0.187768, 0.262406, 0.350412, 0.325110, 0.367181, 0.348264,
                      0.325085, 0.355673, 0.361805, 0.363117, 0.327266, 0.330211,
                      0.385798, 0.358132, 0.380497, 0.383051])
y3_obs = numpy.array([0.117684, 0.175074, 0.236679, 0.234442, 0.270303, 0.272637,
                      0.274075, 0.278981, 0.297151, 0.297797, 0.298722, 0.326645,
                      0.303198, 0.277822, 0.284194, 0.301471])
# Initial conditions
y1_t0 = 1.0
y2_t0 = 0.0
y3_t0 = 0.0

def setOptions(nlpsolver):
    nlpsolver.xtol_rel = 1e-6
    nlpsolver.xtol_abs = 1e-6
    nlpsolver.ftol_rel = 1e-6
    nlpsolver.ftol_abs = 1e-6

def consoleSimulation():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeDelegateDataReporter()
    simulation   = simMethanol2Hydrocarbons_opt()

    dr_tcpip     = daeTCPIPDataReporter()
    dr_data      = daeNoOpDataReporter()
    datareporter.AddDataReporter(dr_tcpip)
    datareporter.AddDataReporter(dr_data)

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 1.15/16
    simulation.TimeHorizon       = 1.15

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
    ts = dvars['tutorial_che_opt_5.y1'].TimeValues[1:]
    y1 = dvars['tutorial_che_opt_5.y1'].Values[1:]
    y2 = dvars['tutorial_che_opt_5.y2'].Values[1:]
    y3 = dvars['tutorial_che_opt_5.y3'].Values[1:]
    nt = len(ts)
    y1_exp = numpy.array([y + numpy.random.uniform(-y*0.1, y*0.1) for y in y1])
    y2_exp = numpy.array([y + numpy.random.uniform(-y*0.1, y*0.1) for y in y2])
    y3_exp = numpy.array([y + numpy.random.uniform(-y*0.1, y*0.1) for y in y3])

    float_formatter = lambda x: "%.6f" % x
    numpy.set_printoptions(formatter={'float_kind':float_formatter})
    print('times  = numpy.%s' % repr(ts))
    print('y1     = numpy.%s' % repr(y1))
    print('y1_obs = numpy.%s' % repr(y1_exp))
    print('y2     = numpy.%s' % repr(y2))
    print('y2_obs = numpy.%s' % repr(y2_exp))
    print('y3     = numpy.%s' % repr(y3))
    print('y3_obs = numpy.%s' % repr(y3_exp))

def run(**kwargs):
    simulation = simMethanol2Hydrocarbons_opt()
    nlpsolver  = pyNLOPT.daeNLOPT('NLOPT_LD_SLSQP')
    lasolver   = pyTrilinos.daeCreateTrilinosSolver("Amesos_Klu", "")
    relativeTolerance = 1e-6
    reportingTimes = times.tolist()
    daeActivity.optimize(simulation, reportingInterval       = 1, 
                                     timeHorizon             = 1,
                                     reportingTimes          = reportingTimes,
                                     lasolver                = lasolver,
                                     nlpsolver               = nlpsolver,
                                     nlpsolver_setoptions_fn = setOptions,
                                     relativeTolerance       = relativeTolerance,
                                     **kwargs)

if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'simulation'):
        consoleSimulation()
    else:
        guiRun = False if (len(sys.argv) > 1 and sys.argv[1] == 'console') else True
        run(guiRun = guiRun)
