"""
***********************************************************************************
                           opencs.py
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
import os, shutil, sys, numpy, json
from daetools.pyDAE import *
from .code_generator import daeCodeGenerator
import pyOpenCS
from pyOpenCS import csModelBuilder_t, csNumber_t, csSimulate

class daeCodeGenerator_OpenCS(daeCodeGenerator):
    """
    Limitations:
     - Discontinuous equations (STNs and IFs) are not supported
     - External functions are not supported
     - Thermo-physical property packages are not supported
    """
    def __init__(self):
        pass

    def generateSimulation(self, simulation,
                                 inputFilesDirectory,
                                 Npe                   = 1,
                                 graphPartitioner      = None,
                                 simulationOptions     = None,
                                 logPartitionResults   = False,
                                 balancingConstraints  = [], 
                                 unaryOperationsFlops  = {}, 
                                 binaryOperationsFlops = {}):
        """ Computational complexity of unary/binary mathematical operations.
        If operation does not exist in the dictionary 1 is assumed.
        - Unary functions:
            eSign, eSqrt, eExp, eLog, eLn, eAbs, eCeil, eFloor, eErf
            eSin, eCos, eTan, eArcSin, eArcCos, eArcTan, eSinh, eCosh, eTanh, eArcSinh, eArcCosh, eArcTanh.
        - Binary functions:
            ePlus eMinus eMulti eDivide ePower eMin eMax eArcTan2
        """ 
        # Check input arguments.
        if not simulation:
            raise RuntimeError('Invalid simulation object')
        if Npe <= 0:
            raise RuntimeError('Invalid number of processing elements')
        if Npe > 1 and graphPartitioner == None:
            raise RuntimeError('Graph partitioner must be specified for Npe > 1')
        for constraint in balancingConstraints:
            if constraint != 'Ncs' and constraint != 'Nflops' and constraint != 'Nnz' and constraint != 'Nflops_j':
                raise RuntimeError('Invalid balancing constraint: %s' % constraint)
        if simulationOptions:
            if not isinstance(simulationOptions, (dict, str)):
                raise RuntimeError('Invalid simulation options specified (must be a dictionary or a string)')
        
        # Create input files directory
        if not os.path.isdir(inputFilesDirectory):
            os.makedirs(inputFilesDirectory)
        
        # Instantiate Model Builder
        modelBuilder = csModelBuilder_t()
        
        # Collect the OpenCS model data from DAE Tools simulation and call 
        # DAE Tools-specific function in pyOpenCS that initialises the OpenCS model.
        mb_data = simulation.GetOpenCSModelData()
        modelBuilder.Initialise_DAETools_DAE_System(mb_data)
        
        # Set user-defined simulation options, otherwise the default ones will be used.
        # This will also check validity of the specified simulation options.
        if simulationOptions:
            if isinstance(simulationOptions, dict):
                options_s = simulationOptions
            else: # it is a string
                options_s = json.loads(simulationOptions)
            modelBuilder.SimulationOptions = options_s
        
        # Set mandatory simulation options.
        options = modelBuilder.SimulationOptions
        options['Simulation']['TimeHorizon']                 = simulation.TimeHorizon
        options['Simulation']['ReportingInterval']           = simulation.ReportingInterval
        options['Solver']['Parameters']['RelativeTolerance'] = simulation.RelativeTolerance
        modelBuilder.SimulationOptions = options
        
        # Partition the system and generate OpenCS models.
        cs_models = modelBuilder.PartitionSystem(Npe, 
                                                 graphPartitioner,
                                                 balancingConstraints  = balancingConstraints,
                                                 logPartitionResults   = False,
                                                 unaryOperationsFlops  = unaryOperationsFlops,
                                                 binaryOperationsFlops = binaryOperationsFlops)
    
        csModelBuilder_t.ExportModels(cs_models, 
                                      inputFilesDirectory, 
                                      modelBuilder.SimulationOptions)
        
        #csSimulate(inputFilesDirectory)

    @property
    def defaultSimulationOptions_DAE(self):
        return csModelBuilder_t.GetDefaultSimulationOptions_DAE()
    
    @property
    def defaultSimulationOptions_ODE(self):
        return csModelBuilder_t.GetDefaultSimulationOptions_ODE()
    
        
