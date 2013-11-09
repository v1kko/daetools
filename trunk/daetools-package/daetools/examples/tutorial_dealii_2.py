#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                        tutorial_deal_II_2.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic, 2013
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
                                                      -->
    Sun (45 degrees), heat flux = 1 kW/m**2, direction n = (1, -1)
      \ \ \ \ \ \ \
       \ \ \ \ \ \ \         Inner tube: T = 300K
        \ \ \ \ \ \ \       /                              -->     
         \ \ \ \ \ \ \     /     Outer surface, flux = ∇q * n
          \ \ \ \ \ \ \   /     /
           \ \ \ \ \ ****/     /
            \ \ \ **    / **  /
             \ \**    **    **
              \ *    *  *    *
                **    **    **
                  **      **
                     ****
                

    dT                                             
   ---- - ∇κ∇Τ + ∇.(vT) = g, in Ω
    dt
"""

import os, sys, numpy, json, math
from daetools.pyDAE import *
from daetools.solvers.deal_II import *
from time import localtime, strftime
from daetools.solvers.superlu import pySuperLU
from daetools.solvers.trilinos import pyTrilinos
from daetools.solvers.aztecoo_options import daeAztecOptions

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

# Neumann BC use either value or gradient
# Dirichlet BC use vector_value with n_component = multiplicity of the equation
# Other functions use value
class GradientFunction_2D(Function_2D):
    def __init__(self, gradient, direction, n_components = 1):
        Function_2D.__init__(self, n_components)
        self.m_gradient = Tensor_1_2D()
        self.m_gradient[0] = gradient * direction[0]
        self.m_gradient[1] = gradient * direction[1]
        
    def gradient(self, point, component = 1):
        if point.x < 0 and point.y > 0:
            return self.m_gradient
        else:
            return Tensor_1_2D()

class BottomGradientFunction_2D(Function_2D):
    def __init__(self, gradient, n_components = 1):
        Function_2D.__init__(self, n_components)
        self.m_gradient = gradient
        
    def value(self, point, component = 1):
        if point.y < -0.5:
            return self.m_gradient
        else:
            return 0.0
            
class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)
        
        rho = 8960.0  # kg/m**3
        cp  =  385.0  # J/(kg*K)
        k   =  401.0  # W/(m*K)
        
        flux_above   = 2.0E3/(rho*cp) # (W/m**2)/((kg/m**3) * (J/(kg*K))) = 
        flux_beneath = 2.0E3/(rho*cp)  # (W/m**2)/((kg/m**3) * (J/(kg*K))) = 
        diffusivity  = k / (rho*cp)    # m**2/s
        
        print 'Thermal diffusivity = %f' % diffusivity
        print 'Beneath source flux = %f' % flux_beneath
        print 'Sbove source flux = %f x (1,-1)' % flux_above
        # Achtung, Achtung!!
        # Diffusivity, velocity, generation, dirichletBC and neumannBC must not go out of scope
        # for deal.II FE model keeps only weak references to them.
        self.functions    = {}
        self.functions['Diffusivity'] = ConstantFunction_2D(diffusivity)
        self.functions['Generation']  = ConstantFunction_2D(0.0)
        
        self.neumannBC      = {}
        self.neumannBC[0]   = (GradientFunction_2D(flux_above, direction = (-1, 1)),  eGradientFlux)
        self.neumannBC[1]   = (BottomGradientFunction_2D(flux_beneath),               eConstantFlux)
        
        self.dirichletBC    = {}
        self.dirichletBC[2] = ConstantFunction_2D(273) # Kelvins
        
        meshes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')
        meshFilename = os.path.join(meshes_dir, 'pipe.msh')
        
        # Achtung, Achtung!!
        # Finite element equations must not go out of scope for deal.II FE model keeps only weak references to them.
        self.cdr1 = dealiiFiniteElementEquation_2D.ConvectionDiffusionEquation('T', 'Temperature', self.dirichletBC, self.neumannBC)
        equations = [self.cdr1] #, self.cdr2]
        
        self.fe_dealII = dealiiFiniteElementSystem_2D( 
                                                       meshFilename,     # path to mesh
                                                       1,                # polinomial order
                                                       QGauss_2D(3),     # quadrature formula
                                                       QGauss_1D(3),     # face quadrature formula
                                                       self.functions,   # dictionary of space dependant functions {'Name':Function<dim>}
                                                       equations         # Finite Element equations (contributions to the cell_matrix, cell_matrix_dt, cell_rhs, BCs etc.)
                                                     )
                                
        self.fe = daeFiniteElementModel('HeatConduction', self, 'Transient heat conduction through a pipe wall with an external heat flux', self.fe_dealII)
       
    def DeclareEquations(self):
        daeModel.DeclareEquations(self)
        
class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_deal_II_2")
        self.m.Description = __doc__
        
    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        m_dt = self.m.fe_dealII.SystemMatrix_dt()
        T    = self.m.fe.dictVariables['T']
        
        # dofIndexesMap relates global DOF indexes to points within daetools variables
        dofIndexesMap = {}
        for variable in self.m.fe.Variables:
            if variable.Name == 'T':
                ic = 273
            for i in xrange(variable.NumberOfPoints):
                dofIndexesMap[variable.OverallIndex + i] = (variable, i, ic)
        
        for row in xrange(m_dt.n):
            # Iterate over columns and set initial conditions.
            # If an item in the dt matrix is zero skip it (it is at the boundary - not a diff. variable).
            for column in self.m.fe_dealII.RowIndices(row):
                if m_dt(row, column) != 0:
                    variable, index, ic = dofIndexesMap[column]
                    variable.SetInitialCondition(index, ic)
                    #print '%s(%d) initial condition = %f' % (variable.Name, column, ic)
    
# Use daeSimulator class
def guiRun(app):
    simulation = simTutorial()
    datareporter = daeDelegateDataReporter()
    tcpipDataReporter = daeTCPIPDataReporter()
    feDataReporter    = self.fe_dealII.CreateDataReporter()
    datareporter.AddDataReporter(tcpipDataReporter)
    datareporter.AddDataReporter(feDataReporter)

    # Connect datareporters
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(tcpipDataReporter.Connect("", simName) == False):
        sys.exit()
    feDataReporter.Connect(os.path.join(os.path.dirname(__file__), 'results'), simName)

    simulation.m.SetReportingOn(True)
    simulation.ReportingInterval = 60    # 1 minute
    simulation.TimeHorizon       = 60*60 # 1 hour
    simulator  = daeSimulator(app, simulation=simulation, datareporter = datareporter)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeDelegateDataReporter()
    simulation   = simTutorial()
    lasolver = pySuperLU.daeCreateSuperLUSolver()
    daesolver.SetLASolver(lasolver)
    """
    lasolver = pyTrilinos.daeCreateTrilinosSolver("AztecOO", "")
    lasolver.NumIters  = 500
    lasolver.Tolerance = 1e-6
    paramListAztec = lasolver.AztecOOOptions
    paramListAztec.set_int("AZ_solver",    daeAztecOptions.AZ_gmres)
    paramListAztec.set_int("AZ_kspace",    500)
    paramListAztec.set_int("AZ_scaling",   daeAztecOptions.AZ_none)
    paramListAztec.set_int("AZ_reorder",   0)
    paramListAztec.set_int("AZ_conv",      daeAztecOptions.AZ_r0)
    paramListAztec.set_int("AZ_keep_info", 1)
    paramListAztec.set_int("AZ_output",    daeAztecOptions.AZ_all) # {AZ_all, AZ_none, AZ_last, AZ_summary, AZ_warnings}
    paramListAztec.set_int("AZ_precond",         daeAztecOptions.AZ_dom_decomp)
    paramListAztec.set_int("AZ_subdomain_solve", daeAztecOptions.AZ_ilut)
    paramListAztec.set_int("AZ_overlap",         daeAztecOptions.AZ_none)
    paramListAztec.set_int("AZ_graph_fill",      1)
    paramListAztec.Print()
    daesolver.SetLASolver(lasolver)
    """

    # Create two data reporters: TCP/IP and DealII
    tcpipDataReporter = daeTCPIPDataReporter()
    feDataReporter    = simulation.m.fe_dealII.CreateDataReporter()
    datareporter.AddDataReporter(tcpipDataReporter)
    datareporter.AddDataReporter(feDataReporter)

    # Connect datareporters
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(tcpipDataReporter.Connect("", simName) == False):
        sys.exit()
    feDataReporter.Connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'), simName)

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 60    # 1 minute
    simulation.TimeHorizon       = 60*60 # 1 hour

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)
    
    # Save the model report and the runtime model report
    #simulation.m.fe.SaveModelReport(simulation.m.fe.Name + ".xml")
    #simulation.m.fe.SaveRuntimeModelReport(simulation.m.fe.Name + "-rt.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    # Run
    simulation.Run()
    simulation.Finalize()

if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        consoleRun()
    else:
        from PyQt4 import QtCore, QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
