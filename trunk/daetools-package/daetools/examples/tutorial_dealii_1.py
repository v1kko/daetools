#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                        tutorial_deal_II_1.py
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
deal.II classes provided by the python wrapper: 
 - Tensors of rank 1 up to three dimensions (Tensor<rank,dim,double> template): 
    * Tensor_1_1D
    * Tensor_1_2D
    * Tensor_1_3D
 - Points up to three dimensions (Point<dim,double> template):
    * Point_1D
    * Point_2D
    * Point_3D
 - Functions up to three dimensions (Function<dim> template):
    * Function_1D
    * Function_2D
    * Function_3D    
 - Solvers for scalar transport equations:
    * Convection-Diffusion Equation up to three dimensions (daeConvectionDiffusion<dim> template):
       + daeConvectionDiffusion_1D
       + daeConvectionDiffusion_2D
       + daeConvectionDiffusion_3D
    * Laplace Equation up to three dimensions
    * Poisson Equation up to three dimensions
    * Helmholtz Equation up to three dimensions
"""

import os, sys, numpy, json, tempfile
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
class fnConstantFunction(Function_2D):
    def __init__(self, val, n_components = 1):
        Function_2D.__init__(self, n_components)
        self.m_value = float(val)
        
    def value(self, point, component = 1):
        #print('Point%s = %f' % (point, self.m_value))
        return self.m_value
    
    def gradient(self, point, component = 1):
        return 0.0
    
    def vector_value(self, point):
        return [self.m_value]
       
    def vector_gradient(self, point):
        return [0.0]
       
class feObject(dealiiFiniteElementSystem_2D):
    def __init__(self, meshFilename, polynomialOrder, quadratureFormula, 
                       faceQuadratureFormula, functions, equations):
        dealiiFiniteElementSystem_2D.__init__(self, meshFilename, 
                                                    polynomialOrder,
                                                    quadratureFormula,
                                                    faceQuadratureFormula,
                                                    functions,
                                                    equations)
        
    def AssembleSystem(self):
        dealiiFiniteElementSystem_2D.AssembleSystem(self)
    
    def NeedsReAssembling(self):
        return False
        
class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)
        
        # Achtung, Achtung!!
        # Diffusivity, velocity, generation, dirichletBC and neumannBC must not go out of scope
        # for deal.II FE model keeps only weak references to them.
        self.functions    = {}
        self.functions['Diffusivity'] = ConstantFunction_2D(401.0/(8960*385))
        self.functions['Generation']  = ConstantFunction_2D(0.0)
        
        self.neumannBC      = {}
        #self.neumannBC[0]   = (ConstantFunction_2D(0.0),            eConstantFlux)
        self.neumannBC[1]   = (ConstantFunction_2D(2E6/(8960*385)), eConstantFlux)
        self.neumannBC[2]   = (ConstantFunction_2D(3E6/(8960*385)), eConstantFlux)
        
        self.dirichletBC    = {}
        self.dirichletBC[0] = fnConstantFunction(200)
        #self.dirichletBC[1] = fnConstantFunction(200)
        #self.dirichletBC[2] = fnConstantFunction(250)
        
        meshes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')
        meshFilename = os.path.join(meshes_dir, 'step-49.msh')
        
        # Achtung, Achtung!!
        # Finite element equations must not go out of scope for deal.II FE model keeps only weak references to them.
        self.cdr1 = dealiiFiniteElementEquation_2D.ConvectionDiffusionEquation('U', 'U description', self.dirichletBC, self.neumannBC)
        print('Convection-Diffusion equation:')
        print('    VariableName         =', self.cdr1.VariableName)
        print('    VariableDescription  =', self.cdr1.VariableDescription)
        print('    Multiplicity         =', self.cdr1.Multiplicity)
        print('    ElementMatrix        =', self.cdr1.Alocal)
        print('    ElementMatrix_dt     =', self.cdr1.Mlocal)
        print('    ElementRHS           =', self.cdr1.Flocal)
        #print('    FunctionsDirichletBC =', self.cdr1.FunctionsDirichletBC)
        #print('    FunctionsNeumannBC   =', self.cdr1.FunctionsNeumannBC)
        equations = [self.cdr1]
        
        self.fe_dealII = feObject(meshFilename,     # path to mesh
                                  1,                # polinomial order
                                  QGauss_2D(3),     # quadrature formula
                                  QGauss_1D(3),     # face quadrature formula
                                  self.functions,   # dictionary {'Name':Function<dim>} used during assemble
                                  equations         # Equations (contributions to the cell_matrix, cell_matrix_dt, cell_rhs, BCs etc.)
                                 )
          
        self.fe = daeFiniteElementModel('Helmholtz', self, 'Modified deal.II step-7 example (s-s Helmholtz equation)', self.fe_dealII)
       
    def DeclareEquations(self):
        daeModel.DeclareEquations(self)
        
class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_deal_II_1")
        self.m.Description = __doc__
        
    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        m_dt = self.m.fe_dealII.Msystem()
        
        # Vector where every item marks the boundar
        #dof_to_boundary = self.m.fe_dealII.GetDOFtoBoundaryMap()
        #print list(dof_to_boundary)
        
        # dofIndexesMap relates global DOF indexes to points within daetools variables
        dofIndexesMap = {}
        for variable in self.m.fe.Variables:
            if variable.Name == 'U':
                ic = 300
            else:
                raise RuntimeError('Unknown variable [%s] found' % variable.Name)
            
            for i in range(variable.NumberOfPoints):
                dofIndexesMap[variable.OverallIndex + i] = (variable, i, ic)
        
        for row in range(m_dt.n):
            # Iterate over columns and set initial conditions.
            # If an item in the dt matrix is zero skip it (it is at the boundary - not a diff. variable).
            for column in self.m.fe_dealII.RowIndices(row):
                if m_dt(row, column) != 0:
                    variable, index, ic = dofIndexesMap[column]
                    variable.SetInitialCondition(index, ic)
                    print('%s(%d) initial condition = %f' % (variable.Name, column, ic))
    
# Use daeSimulator class
def guiRun(app):
    simulation = simTutorial()
    datareporter = daeDelegateDataReporter()
    tcpipDataReporter = daeTCPIPDataReporter()
    feDataReporter    = simulation.m.fe_dealII.CreateDataReporter()
    datareporter.AddDataReporter(tcpipDataReporter)
    datareporter.AddDataReporter(feDataReporter)

    # Connect datareporters
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(tcpipDataReporter.Connect("", simName) == False):
        sys.exit()        
    results_folder = tempfile.mkdtemp(suffix = '-results', prefix = 'daetools-deal.II-')
    feDataReporter.Connect(results_folder, simName)
    try:
        from PyQt4 import QtCore, QtGui
        QtGui.QMessageBox.warning(None, "deal.II", "The simulation results will be located in: %s" % results_folder)
    except Exception as e:
        print(str(e))
    
    simulation.m.SetReportingOn(True)
    simulation.ReportingInterval = 10
    simulation.TimeHorizon       = 500
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
    simulation.ReportingInterval = 10
    simulation.TimeHorizon = 500

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)
    
    # Save the model report and the runtime model report
    #simulation.m.fe.SaveModelReport(simulation.m.fe.Name + ".xml")
    #simulation.m.fe.SaveRuntimeModelReport(simulation.m.fe.Name + "-rt.xml")
    
    from daetools.dae_simulator.simulation_explorer import simulate
    explorer = simulate(simulation)

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
