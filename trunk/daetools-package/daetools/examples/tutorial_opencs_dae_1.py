#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_opencs_dae_1.py
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
Reimplementation of IDAS idasAkzoNob_dns example.
The chemical kinetics problem with 6 non-linear diff. equations::
    
    dy1_dt + 2*r1 - r2 + r3 + r4 = 0
    dy2_dt + 0.5*r1 + r4 + 0.5*r5 - Fin = 0
    dy3_dt - r1 + r2 - r3 = 0
    dy4_dt + r2 - r3 + 2*r4 = 0
    dy5_dt - r2 + r3 - r5 = 0
             Ks*y1*y4 - y6 = 0

where::
    
    r1  = k1 * pow(y1,4) * sqrt(y2)
    r2  = k2 * y3 * y4
    r3  = k2/K * y1 * y5
    r4  = k3 * y1 * y4 * y4
    r5  = k4 * y6 * y6 * sqrt(y2)
    Fin = klA * (pCO2/H - y2)

The system is stiff.
The original results are in tutorial_opencs_dae_1.csv file.
"""

import os, sys, json, numpy
from daetools.solvers.opencs import csModelBuilder_t, csNumber_t, csSimulate
from daetools.examples.tutorial_opencs_aux import compareResults

# ChemicalKinetics class provides information for the OpenCS model:
#  - Variable names
#  - Initial conditions
#  - Model equations
# The same class can be used for specification of both OpenCS and DAE Tools models.
k1   =  18.70
k2   =   0.58
k3   =   0.09
k4   =   0.42
K    =  34.40
klA  =   3.30
Ks   = 115.83
pCO2 =   0.90
H    = 737.00
class ChemicalKinetics:
    def __init__(self):
        self.Nequations = 6

    def GetInitialConditions(self):
        y0 = [0.0] * self.Nequations
        y0[0] = 0.444
        y0[1] = 0.00123
        y0[2] = 0.00
        y0[3] = 0.007
        y0[4] = 0.0
        y0[5] = Ks * y0[0] * y0[3]
        return y0

    def GetVariableNames(self):
        return ['y1', 'y2', 'y3', 'y4', 'y5', 'y6']

    def CreateEquations(self, y, dydt):
        # y is a list of csNumber_t objects representing model variables
        # dydt is a list of csNumber_t objects representing time derivatives of model variables
        y1,y2,y3,y4,y5,y6 = y
        dy1_dt,dy2_dt,dy3_dt,dy4_dt,dy5_dt,dy6_dt = dydt

        r1  = k1 * numpy.power(y1,4) * numpy.sqrt(y2)
        r2  = k2 * y3 * y4
        r3  = k2/K * y1 * y5
        r4  = k3 * y1 * y4 * y4
        r5  = k4 * y6 * y6 * numpy.sqrt(y2)
        Fin = klA * ( pCO2/H - y2 )

        equations = [None] * self.Nequations
        equations[0] = dy1_dt + 2*r1 - r2 + r3 + r4
        equations[1] = dy2_dt + 0.5*r1 + r4 + 0.5*r5 - Fin
        equations[2] = dy3_dt - r1 + r2 - r3
        equations[3] = dy4_dt + r2 - r3 + 2*r4
        equations[4] = dy5_dt - r2 + r3 - r5
        equations[5] = Ks*y1*y4 - y6

        return equations

def run(**kwargs):
    inputFilesDirectory = kwargs.get('inputFilesDirectory', os.path.splitext(os.path.basename(__file__))[0])
    
    # Instantiate the model being simulated.
    model = ChemicalKinetics()
    
    # 1. Initialise the DAE system with the number of variables and other inputs.
    mb = csModelBuilder_t()
    mb.Initialize_DAE_System(model.Nequations, 0, defaultAbsoluteTolerance = 1e-10)
    
    # 2. Specify the OpenCS model.
    # Create and set model equations using the provided time/variable/timeDerivative/dof objects.
    # The DAE system is defined as:
    #     F(x',x,y,t) = 0
    # where x' are derivatives of state variables, x are state variables,
    # y are fixed variables (degrees of freedom) and t is the current simulation time.
    time            = mb.Time             # Current simulation time (t)
    variables       = mb.Variables        # State variables (x)
    timeDerivatives = mb.TimeDerivatives  # Derivatives of state variables (x')
    dofs            = mb.DegreesOfFreedom # Fixed variables (y)
    mb.ModelEquations = model.CreateEquations(variables, timeDerivatives)    
    
    # Set initial conditions.
    mb.VariableValues = model.GetInitialConditions()
    
    # Set variable names.
    mb.VariableNames  = model.GetVariableNames()
    
    # 3. Generate a model for single CPU simulations.    
    # Set simulation options (specified as a string in JSON format).
    # The default options are returned by SimulationOptions function as a string in JSON format.
    # The TimeHorizon and the ReportingInterval must be set.
    options = mb.SimulationOptions
    options['Simulation']['OutputDirectory']             = 'results'
    options['Simulation']['TimeHorizon']                 = 180.0
    options['Simulation']['ReportingInterval']           =   1.0
    options['Solver']['Parameters']['RelativeTolerance'] =  1e-8
    # Data reporter options
    #options['Simulation']['DataReporter']['Name']                       = 'CSV'
    #options['Simulation']['DataReporter']['Parameters']['precision']    = 14
    #options['Simulation']['DataReporter']['Parameters']['delimiter']    = ';'
    #options['Simulation']['DataReporter']['Parameters']['outputFormat'] = 'scientific'
    #options['Simulation']['DataReporter']['Name']                              = 'HDF5'
    #options['Simulation']['DataReporter']['Parameters']['fileNameResults']     = 'results.hdf5'
    #options['Simulation']['DataReporter']['Parameters']['fileNameDerivatives'] = 'derivatives.hdf5'
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
    compareResults(inputFilesDirectory, ['y1', 'y2', 'y3', 'y4', 'y5', 'y6'])
           
if __name__ == "__main__":
    inputFilesDirectory = 'tutorial_opencs_dae_1'
    run(inputFilesDirectory = inputFilesDirectory)
