#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                        tutorial_dealii_1.py
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
__doc__ = """An introductory example of the support for Finite Elements in daetools.
The basic idea is to use an external library to perform all low-level tasks such as
management of mesh elements, degrees of freedom, matrix assembly, management of
boundary conditions etc. deal.II library (www.dealii.org) is employed for these tasks.
The mass and stiffness matrices and the load vector assembled in deal.II library are
used to generate a set of algebraic/differential equations in the following form:
[Mij]{dx/dt} + [Aij]{x} = {Fi}.
Specification of additional equations such as surface/volume integrals are also available.
The numerical solution of the resulting ODA/DAE system is performed in daetools
together with the rest of the model equations.

The unique feature of this approach is a capability to use daetools variables
to specify boundary conditions, time varying coefficients and non-linear terms,
and evaluate quantities such as surface/volume integrals.
This way, the finite element model is fully integrated with the rest of the model
and multiple FE systems can be created and coupled together.
In addition, non-linear and DAE finite element systems are automatically supported.

In this tutorial the simple transient heat conduction problem is solved using
the finite element method:

.. code-block:: none

   dT/dt - kappa/(rho*cp)*nabla^2(T) = g(T) in Omega

The mesh is rectangular with two holes, similar to the mesh in step-49 deal.II example:

.. image:: _static/step-49.png
   :alt:
   :width: 400 px

Dirichlet boundary conditions are set to 300 K on the outer rectangle,
350 K on the inner ellipse and 250 K on the inner diamond.

The temperature plot at t = 500s (generated in VisIt):

.. image:: _static/tutorial_dealii_1-results.png
   :width: 600 px

"""

import os, sys, numpy, json, tempfile
from time import localtime, strftime
from daetools.pyDAE import *
from daetools.solvers.deal_II import *
from daetools.solvers.superlu import pySuperLU

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

"""
daetools provide four main classes to support the deal.II library:
  1) dealiiFiniteElementDOF
     In deal.II represents a degree of freedom distributed on a finite element domain.
     In daetools represents a variable distributed on a finite element domain.
  2) dealiiFiniteElementSystem (implements daeFiniteElementObject)
     It is a wrapper around deal.II FESystem<dim> class and handles all finite element related details.
     It uses information about the mesh, quadrature and face quadrature formulas, degrees of freedom
     and the FE weak formulation to assemble the system's mass matrix (Mij), stiffness matrix (Aij)
     and the load vector (Fi).
  3) dealiiFiniteElementWeakForm
     Contains weak form expressions for the contribution of FE cells to the system/stiffness matrices,
     the load vector, boundary conditions and (optionally) surface/volume integrals (as an output).
  4) daeFiniteElementModel
     daeModel-derived class that use system matrices/vectors from the dealiiFiniteElementSystem object
     to generate a system of equations: [Mij]{dx/dt} + [Aij]{x} = {Fi}.
     This system is in a general case DAE system, although it can also be a system of linear/non-linear
     equations (if the mass matrix is zero).
"""
class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)
        
        self.T_outer = daeVariable("T_outer", temperature_t, self, "Temperature of the outer boundary with id=0 (Dirichlet BC)")

        # Some variables to store results of the FE surface/volume inegrals
        self.MeshSurface = daeVariable("MeshSurface", no_t, self, "Mesh outer surface area in 3d, circumference in 2D)")
        self.MeshVolume  = daeVariable("MeshVolume",  no_t, self, "Mesh volume in 3d, surface in 2D")

        self.Q0_total = daeVariable("Q0_total", no_t, self, "Total heat passing through the boundary with id=0")
        self.Q1_total = daeVariable("Q1_total", no_t, self, "Total heat passing through the boundary with id=1")
        self.Q2_total = daeVariable("Q2_total", no_t, self, "Total heat passing through the boundary with id=2")

        # 1. The starting point is a definition of the daeFiniteElementObject class (1D, 2D or 2D problem).

        #    1.1 Specification of the mesh file in one of the formats supported by deal.II:
        #         - UCD (unstructured cell data)
        #         - DB Mesh
        #         - XDA
        #         - Gmsh
        #         - Tecplot
        #         - NetCDF
        #         - UNV
        #         - VTK
        #         - Cubit
        #        Here the .msh format from Gmsh is used (generated from the step-49.geo file)
        meshes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')
        mesh_file  = os.path.join(meshes_dir, 'step-49.msh')

        #    1.2 Specification of the degrees of freedom (as a python list of dealiiFiniteElementDOF objects)
        #        Every dof has a name which will be also used to declare daetools variable ith the same name,
        #        description, finite element object (deal.II FiniteElement<dim> instance) and the multiplicity.
        #        The following finite elements are available:
        #          - Scalar finite elements: FE_Q, FE_Bernstein
        #          - Vector finite elements: FE_ABF, FE_BDM, FE_Nedelec, FE_RaviartThomas
        #          - Discontinuous Galerkin finite elements
        #            - scalar: FE_DGP, FE_DGQ
        #            - scalar, different shape functions: FE_DGPMonomial, FE_DGPNonparametric, FE_DGQArbitraryNodes
        #            - vector-valued: FE_DGBDM, FE_DGNedelec, FE_DGRaviartThomas
        #        For more information see the deal.II documentation
        #        Multiplicity > 1 defines vector DOFs (such as velocity).
        dofs = [dealiiFiniteElementDOF_2D(name='T',
                                          description='Temperature',
                                          fe = FE_Q_2D(1),
                                          multiplicity=1)]

        #    1.3 Specify quadrature formulas for elements and their faces.
        #        A large number of formulas is available in deal.II:
        #        QGaussLobatto, QGauss, QMidpoint, QSimpson, QTrapez, QMilne, QWeddle,
        #        QGaussLog, QGaussLogR, QGaussOneOverR, QGaussChebyshev, QGaussLobattoChebyshev
        #        each in 1D, 2D and 3D versions. Our example is 2D so we always use 2D versions.
        quadrature      = QGauss_2D(3) # quadrature formula
        faceQuadrature  = QGauss_1D(3) # face quadrature formula

        #    1.4 Create dealiiFiniteElementSystem object using the above information
        #        Nota bene:
        #        Store the object in python variable so it does not go out of scope while still in use by deal.II/daetools
        self.fe_system = dealiiFiniteElementSystem_2D(meshFilename    = mesh_file,      # path to mesh
                                                      quadrature      = quadrature,     # quadrature formula
                                                      faceQuadrature  = faceQuadrature, # face quadrature formula
                                                      dofs            = dofs)           # degrees of freedom

        # 2. Create daeFiniteElementModel object (similarly to the ordinary daetools model)
        #    with the finite element system object as the last argument.
        self.fe_model = daeFiniteElementModel('HeatConduction', self, 'Transient heat conduction FE problem', self.fe_system)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # 3. Define the weak form of the problem.

        #    3.1 First define some auxiliary variables (i.e thermo-physical properties)
        #        In this example copper is used.
        rho   = 8960.0  # kg/m**3
        cp    =  385.0  # J/(kg*K)
        kappa =  401.0  # W/(m*K)
        alpha = kappa / (rho * cp) # Thermal diffusivity (m**2/s)

        #    3.2 Diffusivity and the generation in the heat conduction equation can be:
        #         - constants
        #         - daetools expressions
        #         - deal.II Function<dim,Number> wrapper objects.
        #        In this example we use deal.II ConstantFunction<dim> class to specify a constant value.
        #        Since we have only one DOF we do not need to specify n_components in the constructor
        #        (the default value is 1) and we do not need to handle values of multiple components.
        #        Nota bene:
        #          Again, the function objects have to be stored in the python model
        #          since the weak form holds only the weak references to them.
        self.fun_Diffusivity = ConstantFunction_2D(alpha)
        self.fun_Generation  = ConstantFunction_2D(0.0)

        # The physical boundaries in the mesh file are specified using integers.
        # In our example they are the following:
        outerRectangle = 0 # outer boundary
        innerEllipse   = 1 # inner ellipse at the left side
        innerDiamond   = 2 # inner diamond at the right side

        #    3.3 Specify Dirichlet-type boundary conditions using deal.II Function<dim,Number> wrapper objects
        #        In this case, only the adouble versions of Function<dim,adouble> class can be used.
        #        The allowed values include constants and expressions on daetools variables/parameters.
        #        Here we use a time varying quantity for the outer boundary and constant values for the rest.
        #        The boundary conditions are given as a dictionary where the keys are boundary IDs (integers)
        #        and values are a list of tuples ('DOF name', Function<dim> object).
        dirichletBC = {}
        dirichletBC[outerRectangle] = [('T', adoubleConstantFunction_2D( self.T_outer() ))]
        dirichletBC[innerEllipse]   = [('T', adoubleConstantFunction_2D( adouble(350) ))]
        dirichletBC[innerDiamond]   = [('T', adoubleConstantFunction_2D( adouble(250) ))]

        #    3.4 Often, it is required to calculate surface or volume integrals.
        #        For instance, this is useful to obtain the total quantity flowing through a boundary
        #        in this example the total heat.
        #        Surface integrals are given as dictionaries
        #        {boundary ID : list of tuples (daetools variable, integral weak form)}
        #        This way, several surface integrals for a single boundary can be integrated.
        #        The variables are used to create equations in the following form:
        #          eq.Residual = variable() - integral_expression
        #        This way the results of integration are stored in the specified variable.
        #        More details about specification of the weak form is given below in section 3.6.
        #
        #        In this example we specify four boundary integrals:
        #          a) The simplest case is to use a surface integral to calculate the mesh surface
        #             at the given boundary (in 2D case it is equal to circumference).
        #             This is also useful for testing the validity of the results.
        #          b-d) The total heat passing through the outer rectangle, inner ellipse and inner diamond boundary.
        surfaceIntegrals = {}
        surfaceIntegrals[outerRectangle] = [None, None]
        #        a) Area of the mesh outer surface (a test, just to check the integration validity).
        #           In 2D circumference has to be 2*1.5 + 2*0.8 = 4.6.
        surfaceIntegrals[outerRectangle][0] = (self.MeshSurface(), (phi_2D('T', fe_i, fe_q) * JxW_2D(fe_q)))
        #        b) Total heat transferred through the outer rectangle boundary
        surfaceIntegrals[outerRectangle][1] = (self.Q0_total(), (-alpha * (dphi_2D('T', fe_i, fe_q) * normal_2D(fe_q)) * JxW_2D(fe_q)) * dof_2D('T', fe_i))
        #        c) Total heat transferred through the inner ellipse boundary
        surfaceIntegrals[innerEllipse] = [(self.Q1_total(), (-alpha * (dphi_2D('T', fe_i, fe_q) * normal_2D(fe_q)) * JxW_2D(fe_q)) * dof_2D('T', fe_i))]
        #        d) Total heat transferred through the inner diamond boundary
        surfaceIntegrals[innerDiamond] = [(self.Q2_total(), (-alpha * (dphi_2D('T', fe_i, fe_q) * normal_2D(fe_q)) * JxW_2D(fe_q)) * dof_2D('T', fe_i))]

        #    3.5 Volume integrals are specified similarly. The only difference is that they do not require boundary IDs.
        #        Nota bene:
        #          Care should be taken with the volume integrals since they include values at ALL cells
        #          and the resulting equations can be enormously long!!
        #
        #        In this example we specify only one volume integral: the simplest case used to calculate
        #        the mesh surface area (in 2D it is area; in 3D it is the volume).
        #        This is another test to check the integration validity.
        #        Area consists of the rectangle minus inner ellipse and diamond; here it should be 1.11732...
        #        (the total rectangle surface including holes is 1.5 * 0.8 = 1.2).
        volumeIntegrals = [ (self.MeshVolume(), (phi_2D('T', fe_i, fe_q) * phi_2D('T', fe_j, fe_q) * JxW_2D(fe_q))) ]

        #    3.6 Derive and specify the weak form of the problem.
        #        The weak form consists of the following contributions:
        #         a) Aij - cell contribution to the system stiffness matrix.
        #         b) Mij - cell contribution to the mass stiffness matrix.
        #         c) Fi  - cell contribution to the load vector.
        #         d) boundaryFaceAij  - boundary face contribution to the system stiffness matrix.
        #         e) boundaryFaceFi   - boundary face contribution to the load vector
        #         f) innerCellFaceAij - inner cell face contribution to the system stiffness matrix.
        #         g) innerCellFaceFi  - inner cell face contribution to the load vector
        #         h) functionsDirichletBC - Dirichlet boundary conditions (section 3.3)
        #         i) surfaceIntegrals - surface integrals (section 3.4)
        #         j) volumeIntegrals  - volume  integrals (section 3.5)
        #
        #         The weak form expressions are specified using the functions that wrap deal.II
        #         concepts used to assembly the matrices/vectors. The weak forms in daetools
        #         represent expressions as they would appear in typical nested for loops.
        #         In deal.II a typical cell assembly loop (in C++) would look like
        #         (i.e. a very simple example given in step-7):
        #
        #         typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
        #                                                        endc = dof_handler.end();
        #         for(; cell != endc; ++cell)
        #         {
        #             for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        #             {
        #                 for(unsigned int i = 0; i < dofs_per_cell; ++i)
        #                 {
        #                     for(unsigned int j = 0; j < dofs_per_cell; ++j)
        #                     {
        #                         cell_matrix(i,j) += ((fe_values.shape_grad(i,q_point) *
        #                                               fe_values.shape_grad(j,q_point)
        #                                               +
        #                                               fe_values.shape_value(i,q_point) *
        #                                               fe_values.shape_value(j,q_point)) *
        #                                               fe_values.JxW(q_point));
        #                     }
        #
        #                     cell_rhs(i) += (fe_values.shape_value(i,q_point) *
        #                                     rhs_values [q_point] * fe_values.JxW(q_point));
        #                 }
        #             }
        #         }
        #
        #         In daetools, the dealiiFiniteElementSystem class creates a context where these loops are executed
        #         and only weak form expressions are required - all other data are managed automatically, in a generic way.
        #         Obviously, the generic loops can be used to solve many FE problems but not all.
        #         However, they can support a large number of problems at the moment.
        #         In the future they will be expanded to support a broader class of problems.
        #
        #         Functions are developed for the most of the functionality provided by deal.II FEValues<dim> and
        #         FEFaceValues<dim> classes used for matrix assembly. The current list include (for 1D, 2D and 3D):
        #          - phi (variableName, shapeFunction, quadraturePoint): corresponds to shape_value in deal.II
        #          - dphi (variableName, shapeFunction, quadraturePoint): corresponds to shape_grad in deal.II
        #          - d2phi (variableName, shapeFunction, quadraturePoint): corresponds to shape_hessian in deal.II
        #          - phi_vector (variableName, shapeFunction, quadraturePoint): corresponds to shape_value of vector dofs in deal.II
        #          - dphi_vector (variableName, shapeFunction, quadraturePoint): corresponds to shape_grad of vector dofs in deal.II
        #          - d2phi_vector (variableName, shapeFunction, quadraturePoint): corresponds to shape_hessian of vector dofs in deal.II
        #          - div_phi (variableName, shapeFunction, quadraturePoint): corresponds to divergence in deal.II
        #          - JxW (quadraturePoint): corresponds to the mapped quadrature weight in deal.II
        #          - xyz (quadraturePoint): returns the point for the specified quadrature point in deal.II
        #          - normal (quadraturePoint): corresponds to the normal_vector in deal.II
        #          - function_value (functionName, function, point, component): wraps Function<dim> object that returns a value
        #          - function_gradient (functionName, function, point, component): wraps Function<dim> object that returns a gradient
        #          - function_adouble_value (functionName, function, point, component): wraps Function<dim> object that returns adouble value
        #          - function_adouble_gradient (functionName, function, point, component): wraps Function<dim> object that returns adouble gradient
        #          - dof (variableName, shapeFunction): returns daetools variable at the given index (adouble object)
        #          - dof_approximation (variableName, shapeFunction): returns FE approximation of a quantity as a daetools variable (adouble object)
        #          - dof_gradient_approximation (variableName, shapeFunction): returns FE gradient approximation of a quantity as a daetools variable (adouble object)
        #          - dof_hessian_approximation (variableName, shapeFunction): returns FE hessian approximation of a quantity as a daetools variable (adouble object)
        #          - vector_dof_approximation (variableName, shapeFunction): returns FE approximation of a vector quantity as a daetools variable (adouble object)
        #          - vector_dof_gradient_approximation (variableName, shapeFunction): returns FE approximation of a vector quantity as a daetools variable (adouble object)
        #          - adouble (ad): wraps any daetools expression to be used in matrix assembly
        #          - tensor1 (t): wraps deal.II Tensor<rank=1,double>
        #          - tensor2 (t): wraps deal.II Tensor<rank=2,double>
        #          - tensor3 (t): wraps deal.II Tensor<rank=3,double>
        #          - adouble_tensor1 (t): wraps deal.II Tensor<rank=1,adouble>
        #          - adouble_tensor2 (t): wraps deal.II Tensor<rank=1,adouble>
        #          - adouble_tensor3 (t): wraps deal.II Tensor<rank=1,adouble>

        #         First, we need to wrap Function<dim> objects to be used in the weak form using the function_value function:
        Diffusivity = function_value_2D('Diffusivity', self.fun_Diffusivity, xyz_2D(fe_q))
        Generation  = function_value_2D('Generation',  self.fun_Generation,  xyz_2D(fe_q))

        #         Heat conduction equation is a typical example of convection-diffusion equations.
        #         In the case of pure conduction it simplifies into a diffusion-only equation.
        #         Its weak formulation is simple and given as:
        #           - Cell contribution to the mass matrix:      Mij = phi_i * phi_j * JxW
        #           - Cell contribution to the stiffness matrix: Aij = (dphi_i * dphi_j) * Diffusivity * JxW
        #           - Cell contribution to the load vector:      Fi  = phi_i * Generation * JxW
        accumulation = (phi_2D('T', fe_i, fe_q) * phi_2D('T', fe_j, fe_q)) * JxW_2D(fe_q)
        diffusion    = (dphi_2D('T', fe_i, fe_q) * dphi_2D('T', fe_j, fe_q)) * Diffusivity * JxW_2D(fe_q)
        source       = phi_2D('T', fe_i, fe_q) * Generation * JxW_2D(fe_q)

        weakForm = dealiiFiniteElementWeakForm_2D(Aij = diffusion,
                                                  Mij = accumulation,
                                                  Fi  = source,
                                                  functionsDirichletBC = dirichletBC,
                                                  surfaceIntegrals = surfaceIntegrals,
                                                  volumeIntegrals = volumeIntegrals)

        # Print the assembled weak form as they will be seen from deal.II wrappers:
        print('Transient heat conduction equation:')
        print('    Aij = %s' % str(weakForm.Aij))
        print('    Mij = %s' % str(weakForm.Mij))
        print('    Fi  = %s' % str(weakForm.Fi))
        print('    boundaryFaceAij = %s' % str([item for item in enumerate(weakForm.boundaryFaceAij)]))
        print('    boundaryFaceFi  = %s' % str([item for item in enumerate(weakForm.boundaryFaceFi)]))
        print('    innerCellFaceAij = %s' % str(weakForm.innerCellFaceAij))
        print('    innerCellFaceFi  = %s' % str(weakForm.innerCellFaceFi))
        print('    surfaceIntegrals  = %s' % str([item for item in enumerate(weakForm.surfaceIntegrals)]))
        print('    volumeIntegrals  = %s' % str([item for item in enumerate(weakForm.volumeIntegrals)]))

        #    3.7 Finally, set the weak form of the FE system.
        #        This will declare a set of equations in the following form:
        #          [Mij]{dx/dt} + [Aij]{x} = {Fi}
        #        and the additional (optional) boundary integral equations.
        self.fe_system.WeakForm = weakForm

        # As an exercise, define a time varying quantity T_outer to be used as a Dirichlet boundary condition
        # to illistrate coupling between daetools and deal.II.
        # Here it is a simple constant, but it can be any valid daetools equation.
        eq = self.CreateEquation("T_outer", "Boundary conditions for the outer edge")
        eq.Residual = self.T_outer() - Constant(200 * K)

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_dealii_1")
        self.m.Description = __doc__
        self.m.fe_model.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        # 4. Set the initial conditions for differential variables using the setFEInitialConditions function.
        #    This function requires a dof name and either a float variable or a callable object to be called
        #    for every index in the variable. Here we use a simple constant value of 300 Kelvins.
        setFEInitialConditions(self.m.fe_model, self.m.fe_system, 'T', 300)

# 5. Set-up the main program
#    The only difference is that to obtain the results from the deal.II FE simulation
#    a special ata reporter object should be used (dealiiDataReporter).
#    It is created through the FE system object using the function dealiiFiniteElementSystem.CreateDataReporter()
#    The connect string represents a full path to the directory where the results will be stored in .vtk file format.
#    The data reporter will also generate vtk.visit file for easier animation in VisIt.
#    deal.II data reporter will create files only for FE dofs. The other canbe plotted in an usual fashion.
def guiRun(app):
    datareporter = daeDelegateDataReporter()
    simulation   = simTutorial()
    lasolver = pySuperLU.daeCreateSuperLUSolver()

    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    results_folder = tempfile.mkdtemp(suffix = '-results', prefix = 'tutorial_deal_II_1-')

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

    try:
        from PyQt4 import QtCore, QtGui
        QtGui.QMessageBox.warning(None, "deal.II", "The simulation results will be located in: %s" % results_folder)
    except Exception as e:
        print(str(e))
    
    simulation.m.SetReportingOn(True)
    simulation.ReportingInterval = 10
    simulation.TimeHorizon       = 500
    simulator  = daeSimulator(app, simulation=simulation, datareporter = datareporter, lasolver=lasolver)
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

    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tutorial_deal_II_1-results')

    # Create two data reporters:
    # 1. DealII
    feDataReporter = simulation.m.fe_system.CreateDataReporter()
    datareporter.AddDataReporter(feDataReporter)
    if not feDataReporter.Connect(results_folder, simName):
        sys.exit()
        
    # 2. TCP/IP
    tcpipDataReporter = daeTCPIPDataReporter()
    datareporter.AddDataReporter(tcpipDataReporter)
    if not tcpipDataReporter.Connect("", simName):
        sys.exit()

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 10
    simulation.TimeHorizon = 500

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)
    
    # Save the model report and the runtime model report
    simulation.m.fe_model.SaveModelReport(simulation.m.Name + ".xml")
    #simulation.m.fe_model.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")
    
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
