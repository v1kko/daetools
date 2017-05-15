#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial20.py
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
This tutorial illustrates the sensitivity analysis features.

This model has one state variable (T) and one degree of freedom (Qin).
The sensitivity analysis is enabled and the directory for sensitivity matrix files
set using the SensitivityDataDirectory property (before a call to Initialize).
Qin is set as a parameter for sensitivity intergation.
The sensitivity matrix is saved in mmx coordinate format where the first
dimensions is Nparameters and second Nvariables: S[Np, Nvars].
"""

import os, sys, tempfile, numpy, scipy, scipy.io
from time import localtime, strftime
from daetools.pyDAE import *

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.m     = daeParameter("m",       kg,           self, "Mass of the copper plate")
        self.cp    = daeParameter("c_p",     J/(kg*K),     self, "Specific heat capacity of the plate")
        self.alpha = daeParameter("&alpha;", W/((m**2)*K), self, "Heat transfer coefficient")
        self.A     = daeParameter("A",       m**2,         self, "Area of the plate")
        self.Tsurr = daeParameter("T_surr",  K,            self, "Temperature of the surroundings")

        self.Qin  = daeVariable("Q_in",  power_t,       self, "Power of the heater")
        self.T    = daeVariable("T",     temperature_t, self, "Temperature of the plate")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("HeatBalance", "Integral heat balance equation")
        eq.BuildJacobianExpressions = True
        eq.Residual = self.m() * self.cp() * dt(self.T()) - self.Qin() + self.alpha() * self.A() * (self.T() - self.Tsurr())

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial20")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        self.m.cp.SetValue(385 * J/(kg*K))
        self.m.m.SetValue(1 * kg)
        self.m.alpha.SetValue(200 * W/((m**2)*K))
        self.m.A.SetValue(0.1 * m**2)
        self.m.Tsurr.SetValue(283 * K)

    def SetUpVariables(self):
        self.m.Qin.AssignValue(1000 * W)
        self.m.T.SetInitialCondition(283 * K)
        
    def SetUpSensitivityAnalysis(self):
        # SetSensitivityParameter is a handy alias for SetContinuousOptimizationVariable(variable, LB=0.0, UB=1.0, defaultValue=1.0)
        # In this scenario, the lower bound, the upper bound and the default value are unused.
        # If required, the optimisation functions can be added using the simulation.SetNumberOfObjectiveFunctions(n) function.
        self.SetSensitivityParameter(self.m.Qin)

def run():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()

    # Do no print progress
    log.PrintProgress = False

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Enable reporting of time derivatives for all reported variables
    simulation.ReportTimeDerivatives = True

    # Enable reporting of sensitivities for all reported variables
    simulation.ReportSensitivities = True

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 10
    simulation.TimeHorizon = 500

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # Initialize the simulation
    # The .mmx files with the sensitivity matrices will be saved in the temporary folder
    sensitivity_folder = tempfile.mkdtemp(suffix = '-sensitivities', prefix = 'tutorial20-')
    simulation.SensitivityDataDirectory = sensitivity_folder
    simulation.Initialize(daesolver, datareporter, log, calculateSensitivities = True)

    # Save the model report and the runtime model report
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()
    
    # Run
    simulation.Run()
    simulation.Finalize()

    # Alias for log.Message
    printf = lambda msg: log.Message(msg, 0)
    
    # Load the matrix from one of the sensitivity files (i.e. at time = 10s).
    # Filenames are in the "%06d-%.12f.mmx" % (counter, time) format.
    # The result is the matrix in the coordinate format (although it is actually a dense matrix).
    # Some binary file format might be used in the future (IO operations will be much faster).
    mf = os.path.join(sensitivity_folder, '000001-10.000000000000.mmx')
    coo_mat = scipy.io.mmread(mf)
    printf('Coordinate matrix from the .mmx file at time = 10.0s:')
    printf(str(coo_mat))

    # The sensitivity matrix has the following dimensions: S[Nparams, Nvars]
    nd_mat = coo_mat.toarray()
    
    # Unfortunately, using the raw sensitivity matrix requires some additional proccessing.
    # Every variable has an overall index used to identify it.
    # However, since some variables are fixed they are not part of the DAE system. 
    # Therefore, the DAE solver uses the block indexes: variable indexes within the DAE system ("block of equations").
    # Consequently, the sensitivity matrix also uses the block indexes.
    # Overall indexes can be obtained using the variable.OverallIndex property.
    # The corresponding block index can be obtained from the model.OverallIndex_BlockIndex_VariableNameMap
    # which is a dictionary {overallIndex : tuple(blockIndex,variableName)}.
    # Assigned variables (DOFs) have the block indexes with ULONG_MAX values (very large numbers).
    oi_bi_map = simulation.m.OverallIndex_BlockIndex_VariableNameMap
    printf('Dictionary OverallIndex_BlockIndex_VariableNameMap:')
    printf(str(oi_bi_map))
    
    # Toi is an overall index for variable T:
    Toi = simulation.m.T.OverallIndex
    # T_bi_name is a tuple containing (blockIndex, variableName) for the variable T:
    T_bi_name = oi_bi_map[Toi]
    # Tbi is the block index for the variable 'T'
    Tbi = T_bi_name[0]
    printf('Variable T: overallIndex = %d, blockIndex = %d' % (Toi, Tbi))
    # Qi is parameter index
    Qi = 0
    printf('dT/dQin   = %f' % nd_mat[Qi,Tbi])
    
    # The DAE solver has SensitivityMatrix property which contains a dense sensitivity matrix.
    # It can be called after Integrate... functions to obtain sensitivities at the current time.
    # daeDenseMatrix has operator __call__(row,column) to get the items and the functions
    # GetItem(row,column)/SetItem(row,column,newValue) to get/set the items.
    # Since we call it here at the end of simulation it returns sensitivities at time = TimeHorizon.
    sm = daesolver.SensitivityMatrix
    printf('Sensitivities at time = %f:' % simulation.TimeHorizon)
    printf(sm)
    printf('dT/dQin (at time = %fs) = %f (from the matrix)' % (simulation.TimeHorizon, sm(Qi,Tbi)))
    
    # The dense matrix object has npyValues property that returns a numpy array (the copy of the matrix).
    ndsm = sm.npyValues
    printf(ndsm)
    printf('dT/dQin (at time = %fs) = %f (from the numpy array)' % (simulation.TimeHorizon, ndsm[Qi,Tbi]))
    
if __name__ == "__main__":
    run()
