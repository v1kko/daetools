#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_che_5.py
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
Similar to the chem. eng. example 4, this example shows a comparison between the analytical
results and the discretised population balance equations results solved using the cell
centered finite volume method employing the high resolution upwind scheme (Van Leer
k-interpolation with k = 1/3) and a range of flux limiters.

This tutorial can be run from the console only.

The problem is from the section 4.1.2 Size-independent growth II of the following article:

- Nikolic D.D., Frawley P.J. Application of the Lagrangian Meshfree Approach to
  Modelling of Batch Crystallisation: Part II – An Efficient Solution of Integrated CFD
  and Population Balance Equations. Preprints 2016, 20161100128.
  `doi:10.20944/preprints201611.0012.v1 <http://dx.doi.org/10.20944/preprints201611.0012.v1>`_

and also from the section 3.2 Size-independent growth of the following article:

- Qamar S., Elsner M.P., Angelov I.A., Warnecke G., Seidel-Morgenstern A. (2006)
  A comparative study of high resolution schemes for solving population balances in crystallization.
  Computers and Chemical Engineering 30(6-7):1119-1131.
  `doi:10.1016/j.compchemeng.2006.02.012 <http://dx.doi.org/10.1016/j.compchemeng.2006.02.012>`_

Again, the growth-only crystallisation process was considered with the constant growth rate
of 0.1μm/s and with the different initial number density function:

.. code-block:: none

   n(L,0):                      0, if        L <= 2.0μm
                             1E10, if  2μm < L <= 10μm (region I)
                                0, if 10μm < L <= 18μm
         1E10*cos^2(pi*(L-26)/64), if 18μm < L <= 34μm (region II)
                                0, if 34μm < L <= 42μm
         1E10*sqrt(1-(L-50)^2/64), if 42μm < L <= 58μm (region III)
                                0, if 58μm < L <= 66μm
   1E10*exp(-(L-70)^2/(2sigma^2)), if 66μm < L <= 74μm (region IV)
                                0, if 74μm < L

The crystal size in the range of [0, 100]μm was discretised into 200 elements.
The analytical solution in this case is equal to the initial profile translated right
in time by a distance Gt (the growth rate multiplied by the time elapsed in the process).

Comparison of L1- and L2-norms (ni_HR - ni_analytical):

.. code-block:: none

   -------------------------------------
           Scheme  L1         L2
   -------------------------------------
         superbee  4.464e+10  1.015e+10
            smart  4.727e+10  1.120e+10
            Koren  4.861e+10  1.141e+10
            Sweby  5.435e+10  1.142e+10
               MC  5.129e+10  1.162e+10
           HQUICK  5.531e+10  1.194e+10
             HCUS  5.528e+10  1.194e+10
    vanLeerMinmod  5.600e+10  1.202e+10
          vanLeer  5.814e+10  1.225e+10
            ospre  6.131e+10  1.252e+10
            UMIST  6.181e+10  1.259e+10
            Osher  6.690e+10  1.275e+10
       vanAlbada1  6.600e+10  1.281e+10
           minmod  7.751e+10  1.360e+10
       vanAlbada2  7.901e+10  1.413e+10
   -------------------------------------

The comparison of number density functions between the analytical solution and the
solution obtained using high-resolution scheme with the Superbee flux limiter at t=100s:

.. image:: _static/tutorial_che_5-results.png
   :width: 500px

The comparison of number density functions between the analytical solution and the
solution obtained using high-resolution scheme with the Koren flux limiter at t=100s:

.. image:: _static/tutorial_che_5-results2.png
   :width: 500px
"""

import sys, numpy
from daetools.pyDAE import *
from daetools.pyDAE.data_reporters import *
from time import localtime, strftime
try:
    from .fl_analytical import run_analytical
    from .flux_limiters import HRFluxLimiter, supported_schemes
except:
    from fl_analytical import run_analytical
    from flux_limiters import HRFluxLimiter, supported_schemes

# Standard variable types are defined in variable_types.py
from pyUnits import m, g, kg, s, K, mol, kmol, J, um

pbm_number_density_t = daeVariableType("pbm_number_density_t", m**(-1), 0.0, 1.0e+40,  0.0, 1e-0)

class modelMoC(daeModel):
    def __init__(self, Name, G, Phi_callable, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.G  = G
        self.fl = HRFluxLimiter(Phi_callable, Constant(1e-10 * pbm_number_density_t.Units))

        self.L = daeDomain("L",  self, um, "Characteristic dimension (size) of crystals")

        self.ni = daeVariable("ni", pbm_number_density_t, self, "Van Leer k-interpolation scheme (k = 1/3)", [self.L])

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        L     = self.L.Points
        nL    = self.L.NumberOfPoints

        # k-interpolation (van Leer 1985)
        for i in range(0, nL):
            if i == 0:
                eq = self.CreateEquation("ni(%d)" % i, "")
                eq.Residual = self.ni(0) # Boundary condition: here G*ni = 0
            else:
                eq = self.CreateEquation("ni(%d)" % i, "")
                eq.Residual = self.ni.dt(i) + Constant(G) * (self.fl.ni_edge_plus(i,self.ni,nL) - self.fl.ni_edge_plus(i-1,self.ni,nL)) / (self.L(i) - self.L(i-1))

class simBatchReactor(daeSimulation):
    def __init__(self, modelName, N, L, G, ni_0, Phi):
        daeSimulation.__init__(self)
        self.m = modelMoC(modelName, G, Phi)
        self.m.Description = __doc__

        self.N    = N
        self.L    = L
        self.ni_0 = ni_0

    def SetUpParametersAndDomains(self):
        self.m.L.CreateStructuredGrid(self.N, min(self.L), max(self.L))
        self.m.L.Points = self.L

    def SetUpVariables(self):
        # Initial conditions
        L = self.m.L.Points
        nL = self.m.L.NumberOfPoints

        # Initial conditions (the first item is not differential)
        self.ni_0[0] = None
        self.m.ni.SetInitialConditions(self.ni_0)

def run_simulation(simPrefix, modelName, N, L, G, ni_0, reportingInterval, timeHorizon, Phi):
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeDelegateDataReporter()
    dr_tcpip     = daeTCPIPDataReporter()
    dr_data      = daeNoOpDataReporter()
    datareporter.AddDataReporter(dr_tcpip)
    datareporter.AddDataReporter(dr_data)
    simulation   = simBatchReactor(modelName, N, L, G, ni_0, Phi)

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = reportingInterval
    simulation.TimeHorizon       = timeHorizon

    # Connect data reporter
    simName = simPrefix + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
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
    return dr_data.Process

if __name__ == "__main__":
    # Create an initial CSD array and growth rate
    N    = 200
    L_lb = 0.0
    L_ub = 100.0
    G    = 0.1 * um/s
    ni_0 = numpy.zeros(N+1)
    L    = numpy.linspace(L_lb, L_ub, N+1)
    sigma = 0.778 * (L[1] - L[0])
    for i in range(0, N+1):
        if L[i] > 0 and L[i] < 2:
            ni_0[i] = 0
        elif L[i] > 2 and L[i] <= 10:
            ni_0[i] = 1e10
        elif L[i] > 10 and L[i] <= 18:
            ni_0[i] = 0
        elif L[i] > 18 and L[i] <= 34:
            ni_0[i] = 1e10 * numpy.cos(numpy.pi * (L[i] - 26)/16)**2
        elif L[i] > 34 and L[i] <= 42:
            ni_0[i] = 0
        elif L[i] > 42 and L[i] <= 58:
            ni_0[i] = 1e10 * numpy.sqrt(1 - ((L[i] - 50)**2)/64)
        elif L[i] > 58 and L[i] <= 66:
            ni_0[i] = 0
        elif L[i] > 66 and L[i] <= 77:
            ni_0[i] = 1e10 * numpy.exp(-((L[i] - 70)**2) / (2 * (sigma**2)))
        else:
            ni_0[i] = 0

    reportingInterval = 5
    timeHorizon       = 100
    timeIndex         = 20

    # First find analytical solution
    process_analytical = run_analytical('Analytical', 'Analytical', N, L, G, ni_0, reportingInterval, timeHorizon)
    ni_analytical = process_analytical.dictVariables['Analytical.ni']

    L_report = []
    try:
        for scheme in supported_schemes:
            scheme_name = scheme.__doc__
            process_dpb = run_simulation(scheme_name, scheme_name, N, L, G, ni_0, reportingInterval, timeHorizon, scheme)
            ni_dpb      = process_dpb.dictVariables['%s.ni' % scheme_name]

            ni_anal = ni_analytical.Values[timeIndex]
            ni_dpb  = ni_dpb.Values[timeIndex]

            L1 = numpy.linalg.norm(ni_dpb-ni_anal, ord = 1)
            L2 = numpy.linalg.norm(ni_dpb-ni_anal, ord = 2)
            print('L1 = %e, L2 = %e' % (L1, L2))

            L_report.append((scheme_name, L1, L2))

    finally:
        # Sort by L2
        L_report.sort(key = lambda t: t[2])
        print('   -------------------------------------')
        print('           Scheme  L1         L2        ')
        print('   -------------------------------------')
        for scheme, L1, L2 in L_report:
            print('  %15s  %.3e  %.3e' % (scheme, L1, L2))
        print('   -------------------------------------')
