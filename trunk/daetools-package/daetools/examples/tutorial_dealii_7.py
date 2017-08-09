#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_dealii_7.py
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
In this example 2D transient Stokes flow driven by the differences in buoyancy caused
by the temperature differences in the fluid is solved 
(`deal.II step-31 <https://www.dealii.org/8.4.0/doxygen/deal.II/step_31.html>`_).

The differences to the original problem are that the grid is not adaptive and no 
stabilisation method is used.

The problem can be described using the following equations (the Boussinesq equations):

.. code-block:: none

   -div(2 * eta * eps(u)) + nabla(p) = -rho * beta * g * T in Omega
   -div(u) = 0 in Omega
   dT/dt + div(u * T) - div(k * nabla(T)) = gamma in Omega

The mesh is a simple square (0,1)x(0,1):

.. image:: _static/square(0,1)x(0,1)-50x50.png
   :width: 300 px

The temperature and the velocity vectors plot:

.. image:: _static/tutorial_dealii_7-results.png
   :height: 400 px

Animation:
    
.. image:: _static/tutorial_dealii_7-animation.gif
   :height: 400 px
"""

import os, sys, numpy, json, tempfile, random
from time import localtime, strftime
from daetools.pyDAE import *
from daetools.solvers.deal_II import *
from daetools.solvers.superlu import pySuperLU as superlu

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

class VelocityFunction_2D(adoubleFunction_2D):
    def __init__(self, n_components = 1):
        adoubleFunction_2D.__init__(self, n_components)
        
        self.n_components = n_components

    def vector_value(self, point):
        values = [adouble(0.0)] * self.n_components
        values[0] = adouble(0.0) # ux component
        values[1] = adouble(0.0) # uy component
        return values

class TemperatureSource_2D(Function_2D):
    def __init__(self, n_components = 1):
        Function_2D.__init__(self, n_components)

        # Centers of the heat source objects
        self.centers = [Point_2D(0.3, 0.1), 
                        Point_2D(0.45, 0.1), 
                        Point_2D(0.75, 0.1)]
        # Radius of the heat source objects
        self.radius  = 1.0 / 32.0

    def value(self, point, component = 0):
        if (self.centers[0].distance(point) < self.radius) or \
           (self.centers[1].distance(point) < self.radius) or \
           (self.centers[2].distance(point) < self.radius):
            return 1.0
        else:
            return 0.0

    def vector_value(self, point):
        return [self.value(point, c) for c in range(self.n_components)]

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        dofs = [dealiiFiniteElementDOF_2D(name='u',
                                          description='Velocity',
                                          fe = FE_Q_2D(1),
                                          multiplicity=2),
                dealiiFiniteElementDOF_2D(name='p',
                                          description='Pressure',
                                          fe = FE_Q_2D(1),
                                          multiplicity=1),
                dealiiFiniteElementDOF_2D(name='T',
                                          description='Temperature',
                                          fe = FE_Q_2D(1),
                                          multiplicity=1)]
        self.n_components = int(numpy.sum([dof.Multiplicity for dof in dofs]))

        meshes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')
        mesh_file  = os.path.join(meshes_dir, 'square(0,1)x(0,1)-50x50.msh')

        # Store the object so it does not go out of scope while still in use by daetools
        self.fe_system = dealiiFiniteElementSystem_2D(meshFilename    = mesh_file,     # path to mesh
                                                      quadrature      = QGauss_2D(3),  # quadrature formula
                                                      faceQuadrature  = QGauss_1D(3),  # face quadrature formula
                                                      dofs            = dofs)          # degrees of freedom

        self.fe_model = daeFiniteElementModel('Boussinesq', self, 'The Boussinesq equation', self.fe_system)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # Boundary IDs
        left_edge   = 0
        top_edge    = 1
        right_edge  = 2
        bottom_edge = 3

        # Create some auxiliary objects for readability
        phi_p_i         =  phi_2D('p', fe_i, fe_q)
        phi_p_j         =  phi_2D('p', fe_j, fe_q)
        dphi_p_i        = dphi_2D('p', fe_i, fe_q)
        dphi_p_j        = dphi_2D('p', fe_j, fe_q)
        
        phi_vector_u_i  =         phi_vector_2D('u', fe_i, fe_q)
        phi_vector_u_j  =         phi_vector_2D('u', fe_j, fe_q)
        div_phi_u_i     =            div_phi_2D('u', fe_i, fe_q)
        div_phi_u_j     =            div_phi_2D('u', fe_j, fe_q)
        sym_grad_u_i    = symmetric_gradient_2D('u', fe_i, fe_q)
        sym_grad_u_j    = symmetric_gradient_2D('u', fe_j, fe_q)
        
        phi_T_i         =  phi_2D('T', fe_i, fe_q)
        phi_T_j         =  phi_2D('T', fe_j, fe_q)
        dphi_T_i        = dphi_2D('T', fe_i, fe_q)
        dphi_T_j        = dphi_2D('T', fe_j, fe_q)
        
        # FE approximation of T at the specified quadrature point (adouble object)
        T_dof = dof_approximation_2D('T', fe_q)
        
        # FE approximation of u at the specified quadrature point (SUM(Tensor<1>*u(j)) object
        u_dof = vector_dof_approximation_2D('u', fe_q)
        
        normal  = normal_2D(fe_q)
        xyz     = xyz_2D(fe_q)
        JxW     = JxW_2D(fe_q)
        
        self.gammaFun = TemperatureSource_2D()
        gamma = function_value_2D("gamma", self.gammaFun, xyz)
        
        kappa = 2E-05
        beta  = 10.0
        eta   = 1.0
        rho   = 1.0
        g     = -Point_2D(0.0, 1.0)
        gravity = tensor1_2D(g)

        dirichletBC = {}
        dirichletBC[left_edge]   = [('p', adoubleConstantFunction_2D(adouble(0.0))),
                                    ('u', VelocityFunction_2D(self.n_components))]
        dirichletBC[top_edge]    = [('p', adoubleConstantFunction_2D(adouble(0.0))),
                                    ('u', VelocityFunction_2D(self.n_components))]
        dirichletBC[right_edge]  = [('p', adoubleConstantFunction_2D(adouble(0.0))),
                                    ('u', VelocityFunction_2D(self.n_components))]
        dirichletBC[bottom_edge] = [('p', adoubleConstantFunction_2D(adouble(0.0))),
                                    ('u', VelocityFunction_2D(self.n_components))]
        
        # Mass and stiffness matrix contributions can also be in the form of a three item tuple:
        #  (q_loop_expression, i_loop_expression, j_loop_expression)
        # which is evaluated as q_loop_expression*i_loop_expression*j_loop_expression
        # while load vector contributions can also be in the form of a two item tuple:
        #  (q_loop_expression, i_loop_expression)
        # The advantage of splitting loops is faster assembly (i.e. q_loop_expressions are evaluated only once per quadrature point)
        # and much simpler evaluation trees resulting in low memory requirements.

        # Contributions from the Stokes equation:
        Aij_u_gradient     = 2 * eta * (sym_grad_u_i * sym_grad_u_j) * JxW
        Aij_p_gradient     = -(div_phi_u_i * phi_p_j) * JxW
        Fi_buoyancy        = (T_dof, -rho * beta * (gravity * phi_vector_u_i) * JxW)  # -> float * adouble
        
        # Contributions from the continuity equation:
        Aij_continuity     = -(phi_p_i * div_phi_u_j) * JxW

        # Contributions from the heat convection-diffusion equation:
        Mij_T_accumulation = phi_T_i * phi_T_j * JxW
        Aij_T_convection   = (u_dof, phi_T_i, dphi_T_j * JxW)
        #Aij_T_convection   = u_dof*(phi_T_i*dphi_T_j * JxW) # -> Tensor<rank=1,float> * Tensor<rank=1,adouble> 
        Aij_T_diffusion    = kappa * (dphi_T_i * dphi_T_j) * JxW
        Fi_T_source        = phi_T_i * gamma * JxW

        # Total contributions:
        Mij = Mij_T_accumulation
        Aij = [Aij_u_gradient + Aij_p_gradient + Aij_continuity + Aij_T_diffusion,  Aij_T_convection]
        Fi  = [Fi_T_source, Fi_buoyancy]
        
        weakForm = dealiiFiniteElementWeakForm_2D(Aij = Aij,
                                                  Mij = Mij,
                                                  Fi  = Fi,
                                                  functionsDirichletBC = dirichletBC)

        self.fe_system.WeakForm = weakForm

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_dealii_7")
        self.m.Description = __doc__
        self.m.fe_model.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        setFEInitialConditions(self.m.fe_model, self.m.fe_system, 'T', 0.0)
        
# Use daeSimulator class
def guiRun(app):
    datareporter = daeDelegateDataReporter()
    simulation   = simTutorial()
    lasolver     = superlu.daeCreateSuperLUSolver()

    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    results_folder = tempfile.mkdtemp(suffix = '-results', prefix = 'tutorial_deal_II_7-')

    # Create two data reporters:
    # 1. deal.II (exports only FE DOFs in .vtk format to the specified directory)
    feDataReporter = simulation.m.fe_system.CreateDataReporter()
    datareporter.AddDataReporter(feDataReporter)
    if not feDataReporter.Connect(results_folder, simName):
        sys.exit()

    # 2. TCP/IP
    tcpipDataReporter = daeTCPIPDataReporter()
    datareporter.AddDataReporter(tcpipDataReporter)
    if not tcpipDataReporter.Connect("", simName):
        sys.exit()

    daeQtMessage("deal.II", "The simulation results will be located in: %s" % results_folder)

    simulation.m.SetReportingOn(True)
    simulation.ReportingInterval = 1
    simulation.TimeHorizon       = 50
    simulator  = daeSimulator(app, simulation=simulation, datareporter = datareporter, lasolver=lasolver)
    simulator.exec_()


# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeDelegateDataReporter()
    simulation   = simTutorial()

    lasolver = superlu.daeCreateSuperLUSolver()
    #lasolver = pyTrilinos.daeCreateTrilinosSolver("Amesos_Umfpack", "")
    #lasolver = pyIntelPardiso.daeCreateIntelPardisoSolver()
    daesolver.SetLASolver(lasolver)

    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tutorial_deal_II_7-results')

    # Create two data reporters:
    # 1. deal.II (exports only FE DOFs in .vtk format to the specified directory)
    feDataReporter = simulation.m.fe_system.CreateDataReporter()
    datareporter.AddDataReporter(feDataReporter)
    if not feDataReporter.Connect(results_folder, simName):
        sys.exit()

    # 2. TCP/IP
    #tcpipDataReporter = daeTCPIPDataReporter()
    #datareporter.AddDataReporter(tcpipDataReporter)
    #if not tcpipDataReporter.Connect("", simName):
    #    sys.exit()

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 0.5
    simulation.TimeHorizon = 50

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)
    
    # Save the model report and the runtime model report
    simulation.m.fe_model.SaveModelReport(simulation.m.Name + ".xml")
    #simulation.m.fe_model.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")
    
    from daetools.pyDAE.eval_node_graph import daeNodeGraph
    """
    A = simulation.m.fe_system.Asystem()
    row = 443
    cols = simulation.m.fe_system.RowIndices(row)
    print(cols)
    for col in cols:
        ad = A.GetItem(row, col)
        if ad.Node:
            print('Processing item (%04d,%04d)...' % (row,col))
            ng = daeNodeGraph('item', ad.Node)
            ng.SaveGraph('deal.ii-7-item(%04d,%04d).png' % (row,col))
    sys.exit()
    """
    """
    eeis = simulation.m.fe_model.Equations[0].EquationExecutionInfos
    for i,eei in enumerate(eeis):
        if (i in range(4000, 4005)) or (i in range(1200, 1210)):
            print('Processing node %d...' % i)
            ng = daeNodeGraph('residual', eei.Node)
            ng.SaveGraph('deal.ii-7-T_equation[%04d].png' % i)
    sys.exit()
    """
    
    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    # Run
    simulation.Run()
    simulation.Finalize()

if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        consoleRun()
    else:
        app = daeCreateQtApplication(sys.argv)
        guiRun(app)
