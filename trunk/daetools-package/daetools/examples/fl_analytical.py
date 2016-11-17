#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           fl_analytical.py
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
import sys, numpy, scipy.interpolate, math
from daetools.pyDAE import *
from daetools.pyDAE.data_reporters import *
from time import localtime, strftime
import matplotlib.pyplot

# Standard variable types are defined in variable_types.py
from pyUnits import m, g, kg, s, K, mol, kmol, J, um

pbm_number_density_t = daeVariableType("pbm_number_density_t", m**(-1), 0.0, 1.0e+40,  0.0, 1e-0)

class extfnTranslateInTime(daeScalarExternalFunction):
    def __init__(self, Name, Model, units, L, ni0, G, i, Time):
        arguments = {}
        arguments["t"]  = Time

        self.L   = L   # quantity ndarray
        self.ni0 = ni0 # initial ni ndarray
        self.G   = G   # quantity
        self.i   = i   # index of the point

        self.interp = scipy.interpolate.interp1d([qL.value for qL in self.L], [ni0 for ni0 in self.ni0])

        # During the solver iterations, the function is called very often with the same arguments
        # Therefore, cache the last interpolated value to speed up a simulation
        self.cache = None

        # Counters for performance (just an info; not really needed)
        self.counter       = 0
        self.cache_counter = 0

        daeScalarExternalFunction.__init__(self, Name, Model, units, arguments)

    def Calculate(self, values):
        # Increase the call counter every time the function is called
        self.counter += 1

        # Get the argument from the dictionary of arguments' values.
        time = values["t"].Value

        # Here we do not need to return a derivative for it is not a function of variables.

        # First check if an interpolated value was already calculated during the previous call
        # If it was return the cached value (derivative part is always equal to zero in this case)
        if self.cache:
            if self.cache[0] == time:
                self.cache_counter += 1
                return adouble(self.cache[1])

        # The time received is not in the cache and has to be calculated.
        new_L = self.L[self.i].value - self.G.value * time # um/s * s -> um
        if new_L < self.L[0].value:
            res = adouble(self.ni0[0])
        else:
            interp_value = float(self.interp(new_L))
            res = adouble(interp_value, 0)

        #print 'Point %d' % self.i
        #print '    time   = %.12f' % (time)
        #print '    L      = %f' % (self.L[self.i].value)
        #print '    new_L  = %.20f' % (new_L)
        #print '    new_ni = %f' % (res.Value)

        # Save it in the cache for later use
        self.cache = (time, res.Value)

        return res

class modelAnalyticalSolution(daeModel):
    def __init__(self, Name, G, ni_0, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.G    = G
        self.ni_0 = ni_0

        self.L = daeDomain("L",  self, um, "Characteristic dimension (size) of crystals")

        self.ni_analytical = daeVariable("ni", pbm_number_density_t, self, "Analytical solution", [self.L])
        self.t             = daeVariable("t",  no_t,                 self, "Time elapsed in the process")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("Time", "The time elapsed in the process.")
        eq.Residual = self.t.dt() - Constant(1.0 * 1/s)

        L  = self.L.Points
        qL = L * self.L.Units
        nL = self.L.NumberOfPoints

        # Analytical solution
        self.extfns = [None for n in range(0, nL)]
        for i in range(0, nL):
            self.extfns[i] = extfnTranslateInTime("translate", self, pbm_number_density_t.Units, qL, self.ni_0, self.G, i, Time())
            eq = self.CreateEquation("ni_analytical(%d)" % i, "")
            eq.Residual = self.ni_analytical(i) - self.extfns[i]()

class simFluxLimiter(daeSimulation):
    def __init__(self, modelName, N, L, G, ni_0):
        daeSimulation.__init__(self)
        self.m = modelAnalyticalSolution(modelName, G, ni_0)

        self.N    = N
        self.L    = L
        self.G    = G
        self.ni_0 = ni_0

    def SetUpParametersAndDomains(self):
        self.m.L.CreateStructuredGrid(self.N, min(self.L), max(self.L))
        self.m.L.Points = self.L

    def SetUpVariables(self):
        # Initial conditions
        self.m.t.SetInitialCondition(0.0)

def run_analytical(simulationPrefix, modelName, N, L, G, ni_0, reportingInterval, timeHorizon):
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeDelegateDataReporter()
    dr_tcpip     = daeTCPIPDataReporter()
    dr_plot      = daePlotDataReporter()
    datareporter.AddDataReporter(dr_tcpip)
    datareporter.AddDataReporter(dr_plot)
    simulation   = simFluxLimiter(modelName, N, L, G, ni_0)

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = reportingInterval
    simulation.TimeHorizon       = timeHorizon

    # Connect data reporter
    simName = simulationPrefix + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(dr_tcpip.Connect("", simName) == False):
        sys.exit()

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)

    # Save the model report and the runtime model report
    #simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    #simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    # Run
    simulation.Run()
    simulation.Finalize()

    print('\n\nTranslateInTime[0] statistics:')
    print('  called %d times (cache value used %d times)' % (simulation.m.extfns[0].counter,
                                                             simulation.m.extfns[0].cache_counter))

    return dr_plot.Process

if __name__ == "__main__":
    # Create an initial CSD array and growth rate
    modelName = 'dpb_analytical(1)'
    N    = 100
    L_lb = 0.0
    L_ub = 100.0
    G    = 1 * um/s
    ni_0 = numpy.zeros(N+1)
    L    = numpy.linspace(L_lb, L_ub, N+1)
    for i in range(0, N+1):
        if L[i] > 10 and L[i] < 20:
            ni_0[i] = 1e10
    reportingInterval = 5
    timeHorizon       = 100

    process_analytical = run_analytical(modelName, N, L_lb, L_ub, G, ni_0, reportingInterval, timeHorizon)
    print(process_analytical.dictDomains)
    print(process_analytical.dictVariables)

