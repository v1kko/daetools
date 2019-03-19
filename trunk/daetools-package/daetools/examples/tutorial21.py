#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial21.py
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
This tutorial introduces different methods for evaluation of equations in parallel.
Equations residuals, Jacobian matrix and sensitivity residuals can be evaluated 
in parallel using two methods

1. The Evaluation Tree approach (default)
   
   OpenMP API is used for evaluation in parallel.
   This method is specified by setting daetools.core.equations.evaluationMode option 
   in daetools.cfg to "evaluationTree_OpenMP" or setting the simulation property:
   
   simulation.EvaluationMode = eEvaluationTree_OpenMP
   
   numThreads controls the number of OpenMP threads in a team.
   If numThreads is 0 the default number of threads is used (the number of cores in the system). 
   Sequential evaluation is achieved by setting numThreads to 1.
   
2. The Compute Stack approach

   Equations can be evaluated in parallel using:

   a) OpenMP API for general purpose processors and manycore devices.
      
      This method is specified by setting daetools.core.equations.evaluationMode option 
      in daetools.cfg to "computeStack_OpenMP" or setting the simulation property:
   
      simulation.EvaluationMode = eComputeStack_OpenMP
   
      numThreads controls the number of OpenMP threads in a team.
      If numThreads is 0 the default number of threads is used (the number of cores in the system). 
      Sequential evaluation is achieved by setting numThreads to 1.

   b) OpenCL framework for streaming processors and heterogeneous systems.

      This type is implemented in an external Python module pyEvaluator_OpenCL. 
      It is up to one order of magnitude faster than the Evaluation Tree approach. 
      However, it does not support external functions nor thermo-physical packages.
      
      OpenCL evaluators can use a single or multiple OpenCL devices.
      It is required to install OpenCL drivers/runtime libraries.
      Intel: https://software.intel.com/en-us/articles/opencl-drivers
      AMD: https://support.amd.com/en-us/kb-articles/Pages/OpenCL2-Driver.aspx
      NVidia: https://developer.nvidia.com/opencl
"""

import os, sys, tempfile
from time import localtime, strftime
from daetools.pyDAE import *
from daetools.solvers.superlu import pySuperLU
from daetools.pyDAE.evaluator_opencl import pyEvaluator_OpenCL

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x = daeDomain("x", self, m, "X axis domain")
        self.y = daeDomain("y", self, m, "Y axis domain")

        self.Qb  = daeParameter("Q_b",         W/(m**2), self, "Heat flux at the bottom edge of the plate")
        self.Tt  = daeParameter("T_t",                K, self, "Temperature at the top edge of the plate")
        self.rho = daeParameter("&rho;",      kg/(m**3), self, "Density of the plate")
        self.cp  = daeParameter("c_p",         J/(kg*K), self, "Specific heat capacity of the plate")
        self.k   = daeParameter("&lambda;_p",   W/(m*K), self, "Thermal conductivity of the plate")
       
        self.T = daeVariable("T", temperature_t, self)
        self.T.DistributeOnDomain(self.x)
        self.T.DistributeOnDomain(self.y)
        self.T.Description = "Temperature of the plate"

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("HeatBalance", "Heat balance equation valid on open x and y domains")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = self.rho() * self.cp() * dt(self.T(x,y)) - \
                      self.k() * (d2(self.T(x,y), self.x, eCFDM) + d2(self.T(x,y), self.y, eCFDM))

        eq = self.CreateEquation("BC_bottom", "Neumann boundary conditions at the bottom edge (constant flux)")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eLowerBound)
        eq.Residual = - self.k() * d(self.T(x,y), self.y, eCFDM) - self.Qb()

        eq = self.CreateEquation("BC_top", "Dirichlet boundary conditions at the top edge (constant temperature)")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eUpperBound)
        eq.Residual = self.T(x,y) - self.Tt()

        eq = self.CreateEquation("BC_left", "Neumann boundary conditions at the left edge (insulated)")
        x = eq.DistributeOnDomain(self.x, eLowerBound)
        y = eq.DistributeOnDomain(self.y, eClosedClosed)
        eq.Residual = d(self.T(x,y), self.x, eCFDM)

        eq = self.CreateEquation("BC_right", "Neumann boundary conditions at the right edge (insulated)")
        x = eq.DistributeOnDomain(self.x, eUpperBound)
        y = eq.DistributeOnDomain(self.y, eClosedClosed)
        eq.Residual = d(self.T(x,y), self.x, eCFDM)

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial21")
        self.m.Description = __doc__
        
    def SetUpParametersAndDomains(self):
        self.m.x.CreateStructuredGrid(20, 0, 0.1)
        self.m.y.CreateStructuredGrid(20, 0, 0.1)

        self.m.k.SetValue(401 * W/(m*K))
        self.m.cp.SetValue(385 * J/(kg*K))
        self.m.rho.SetValue(8960 * kg/(m**3))
        self.m.Qb.SetValue(1.0e5 * W/(m**2))
        self.m.Tt.SetValue(300 * K)

    def SetUpVariables(self):
        for x in range(1, self.m.x.NumberOfPoints - 1):
            for y in range(1, self.m.y.NumberOfPoints - 1):
                self.m.T.SetInitialCondition(x, y, 300 * K)

def run(**kwargs):
    simulation = simTutorial()
    
    # Equation EvaluationMode can be one of:
    #  - eEvaluationTree_OpenMP
    #  - eComputeStack_OpenMP
    #simulation.EvaluationMode = eComputeStack_External

    # External compute stack evaluators can be set using SetComputeStackEvaluator function.
    # Here, the evaluation mode is set to eComputeStack_External.
    # Evaluators can be also set using the computeStackEvaluator argument of daeActivity.simulate function.
    # Available OpenCL platforms/devices can be obtained using the following functions: 
    openclPlatforms = pyEvaluator_OpenCL.AvailableOpenCLPlatforms()
    openclDevices   = pyEvaluator_OpenCL.AvailableOpenCLDevices()
    print('Available OpenCL platforms:')
    for platform in openclPlatforms:
        print('  Platform: %s' % platform.Name)
        print('    PlatformID: %d' % platform.PlatformID)
        print('    Vendor:     %s' % platform.Vendor)
        print('    Version:    %s' % platform.Version)
        #print('    Profile:    %s' % platform.Profile)
        #print('    Extensions: %s' % platform.Extensions)
        print('')
    print('Available OpenCL devices:')
    for device in openclDevices:
        print('  Device: %s' % device.Name)
        print('    PlatformID:      %d' % device.PlatformID)
        print('    DeviceID:        %d' % device.DeviceID)
        print('    DeviceVersion:   %s' % device.DeviceVersion)
        print('    DriverVersion:   %s' % device.DriverVersion)
        print('    OpenCLVersion:   %s' % device.OpenCLVersion)
        print('    MaxComputeUnits: %d' % device.MaxComputeUnits)
        print('')

    # OpenCL evaluators can use a single or multiple OpenCL devices.
    #   a) Single OpenCL device:
    computeStackEvaluator = pyEvaluator_OpenCL.CreateComputeStackEvaluator(platformID = 0, deviceID = 0)
    #   b) Multiple OpenCL devices (for heterogenous computing):
    #computeStackEvaluator = pyEvaluator_OpenCL.CreateComputeStackEvaluator( [(0, 0, 0.6), (1, 1, 0.4)] )
    simulation.SetComputeStackEvaluator(computeStackEvaluator)
    
    # Create LA solver
    lasolver = pySuperLU.daeCreateSuperLUSolver()
    
    return daeActivity.simulate(simulation, reportingInterval = 5, 
                                            timeHorizon       = 500,
                                            lasolver          = lasolver,
                                            **kwargs)

if __name__ == "__main__":
    guiRun = False if (len(sys.argv) > 1 and sys.argv[1] == 'console') else True
    run(guiRun = guiRun)    
