#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_opencs_ode_1.py
                DAE Tools: pyOpenCS module, www.daetools.com
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
Reimplementation of CVodes cvsRoberts_dns example.
The Roberts chemical kinetics problem with 3 rate equations::
    
    dy1/dt = -0.04*y1 + 1.e4*y2*y3
    dy2/dt =  0.04*y1 - 1.e4*y2*y3 - 3.e7*(y2)^2
    dy3/dt =  3.e7*(y2)^2

The problem is simulated for 4000 s, with the initial conditions::
    
    y1 = 1.0
    y2 = y3 = 0
    
The problem is stiff.
The original results are in tutorial_opencs_ode_1.csv file.
"""

import os, sys, json, numpy
from daetools.solvers.opencs import csModelBuilder_t, csNumber_t, csSimulate
from daetools.examples.tutorial_opencs_aux import compareResults

class Roberts:
    def __init__(self):
        self.Nequations = 3

    def GetInitialConditions(self):
        y0 = [0.0] * self.Nequations

        y0[0] = 1.0
        y0[1] = 0.0
        y0[2] = 0.0
        return y0

    def GetVariableNames(self):
        return ['y1', 'y2', 'y3']
    
    def CreateEquations(self, y):
        # y is a list of csNumber_t objects representing model variables
        y1,y2,y3 = y

        equations = [None] * self.Nequations
        equations[0] = -0.04 * y1 + 1.0e4 * y2 * y3
        equations[1] =  0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * numpy.power(y2, 2)
        equations[2] =  3.0e7 * numpy.power(y2, 2)
        
        return equations
    
def run(**kwargs):
    inputFilesDirectory = kwargs.get('inputFilesDirectory', os.path.splitext(os.path.basename(__file__))[0])

    # Instantiate the model being simulated.
    model = Roberts()
    
    # 1. Initialise the ODE system with the number of variables and other inputs.
    mb = csModelBuilder_t()
    mb.Initialize_ODE_System(model.Nequations, 0, defaultAbsoluteTolerance = 1e-7)
    
    # 2. Specify the OpenCS model.    
    # Create and set model equations using the provided time/variable/dof objects.
    # The ODE system is defined as:
    #     x' = f(x,y,t)
    # where x' are derivatives of state variables, x are state variables,
    # y are fixed variables (degrees of freedom) and t is the current simulation time.
    time            = mb.Time             # Current simulation time (t)
    variables       = mb.Variables        # State variables (x)
    dofs            = mb.DegreesOfFreedom # Fixed variables (y)
    mb.ModelEquations = model.CreateEquations(variables)    
    
    # Set initial conditions
    mb.VariableValues = model.GetInitialConditions()
    
    # Set variable names.
    mb.VariableNames  = model.GetVariableNames()
    
    # 3. Generate a model for single CPU simulations.    
    # Set simulation options (specified as a string in JSON format).
    # The default options are returned by SimulationOptions function as a string in JSON format.
    # The TimeHorizon and the ReportingInterval must be set.
    options = mb.SimulationOptions
    options['Simulation']['OutputDirectory']             = 'results'
    options['Simulation']['TimeHorizon']                 = 4000.0
    options['Simulation']['ReportingInterval']           =   10.0
    options['Solver']['Parameters']['RelativeTolerance'] =   1e-5
    mb.SimulationOptions = options
    
    # Partition the system to create the OpenCS model for a single CPU simulation.
    # In this case (Npe = 1) the graph partitioner is not required.
    Npe = 1
    graphPartitioner = None
    cs_models = mb.PartitionSystem(Npe, graphPartitioner)
    csModelBuilder_t.ExportModels(cs_models, inputFilesDirectory, mb.SimulationOptions)
    print("OpenCS model generated successfully!")

    # 4. Run simulation using the exported model from the specified directory.
    csSimulate(inputFilesDirectory)
    
    # Compare OpenCS and the original model results.
    compareResults(inputFilesDirectory, ['y1', 'y2', 'y3'])
    
if __name__ == "__main__":
    inputFilesDirectory = 'tutorial_opencs_ode_1'
    run(inputFilesDirectory = inputFilesDirectory)
