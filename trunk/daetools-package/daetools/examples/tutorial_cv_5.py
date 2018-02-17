#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_cv_5.py
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
Code verification using the Method of Manufactured Solutions.

This problem and its solution in `COMSOL Multiphysics <https://www.comsol.com>`_ software 
is described in the COMSOL blog:
`Verify Simulations with the Method of Manufactured Solutions (2015)
<https://www.comsol.com/blogs/verify-simulations-with-the-method-of-manufactured-solutions>`_.

Here, a 1D transient heat conduction problem in a bar of length L is solved using the FE method:

.. code-block:: none

   dT/dt - k/(rho*cp) * d2T/dx2 = 0, x in [0,L]

with the following boundary:
    
.. code-block:: none

   T(0,t) = 500 K
   T(L,t) = 500 K

and initial conditions:
    
.. code-block:: none

   T(x,0) = 500 K

The manufactured solution is given by function u(x):
    
.. code-block:: none

   u(x) = 500 + (x/L) * (x/L - 1) * (t/tau)
  
The new source term is:
    
.. code-block:: none

   g(x) = du/dt - k/(rho*cp) * d2u/dx2
   
The terms in the source g term are:
    
.. code-block:: none

   du_dt   = (x/L) * (x/L - 1) * (1/tau)
   d2u_dx2 = (2/(L**2)) * (t/tau)
      
Finally, the original problem with the new source term is:
    
.. code-block:: none

   dT/dt - k/(rho*cp) * d2T/dx2 = g(x), x in [0,L]

The mesh is linear (a bar) with a length of 100 m:

.. image:: _static/bar(0,100)-20.png
   :width: 500 px

The comparison plots for the coarse mesh and linear elements: 

.. image:: _static/tutorial_cv_5-results-5_elements-I_order.png
   :width: 400 px

The comparison plots for the coarse mesh and quadratic elements: 

.. image:: _static/tutorial_cv_5-results-5_elements-II_order.png
   :width: 400 px

The comparison plots for the fine mesh and linear elements: 

.. image:: _static/tutorial_cv_5-results-20_elements-I_order.png
   :width: 400 px

The comparison plots for the fine mesh and quadratic elements: 

.. image:: _static/tutorial_cv_5-results-20_elements-II_order.png
   :width: 400 px
"""

import os, sys, numpy, json, tempfile
from time import localtime, strftime
import matplotlib.pyplot as plt
from daetools.pyDAE import *
from daetools.solvers.deal_II import *
from daetools.solvers.superlu import pySuperLU

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

class TemperatureSource_1D(adoubleFunction_1D):
    def __init__(self, L, tau, t, alpha, n_components = 1):
        adoubleFunction_1D.__init__(self, n_components)

        self.L     = L
        self.tau   = tau
        self.t     = t
        self.alpha = alpha

    def value(self, point, component = 0):
        x     = point.x
        L     = self.L
        tau   = self.tau
        t     = Time()
        alpha = self.alpha

        u       = lambda x: 500 + (x/L) * (x/L - 1) * (t/tau)
        du_dt   = lambda x: (x/L) * (x/L - 1) * (1/tau)
        du_dx   = lambda x: (2*x/L**2 - 1/L) * (t/tau)
        d2u_dx2 = lambda x: (2/(L**2)) * (t/tau)
        Q       = lambda x: du_dt(x) - alpha * d2u_dx2(x)

        return Q(x)

    def vector_value(self, point):
        return [self.value(point, c) for c in range(self.n_components)]

class modTutorial(daeModel):
    def __init__(self, Name, mesh, quadratureFormulaOrder, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        dofs = [dealiiFiniteElementDOF_1D(name='T',
                                          description='Temperature',
                                          fe = FE_Q_1D(1),
                                          multiplicity=1)]
        self.n_components = int(numpy.sum([dof.Multiplicity for dof in dofs]))

        meshes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')
        mesh_file  = os.path.join(meshes_dir, mesh)

        # Store the object so it does not go out of scope while still in use by daetools
        self.fe_system = dealiiFiniteElementSystem_1D(meshFilename    = mesh_file,
                                                      quadrature      = QGauss_1D(quadratureFormulaOrder),
                                                      faceQuadrature  = QGauss_0D(quadratureFormulaOrder),
                                                      dofs            = dofs)

        self.fe_model = daeFiniteElementModel('HeatConduction', self, 'Transient heat conduction', self.fe_system)

        self.L = 100 # m
        
        self.x = daeDomain("x", self, m, "x domain")
        self.u = daeVariable("u", no_t, self, "", [self.x])

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # Thermo-physical properties of the metal.
        Ac    =    0.1  # m**2
        rho   = 2700.0  # kg/m**3
        cp    =  900.0  # J/(kg*K)
        kappa =  238.0  # W/(m*K)
        tau   = 3600.0  # seconds 
        L     =  self.L # m   
        t     = Time()
        # Thermal diffusivity (m**2/s)
        alpha = kappa/(rho * cp)

        # Create some auxiliary objects for readability
        phi_i  =  phi_1D('T', fe_i, fe_q)
        phi_j  =  phi_1D('T', fe_j, fe_q)
        dphi_i = dphi_1D('T', fe_i, fe_q)
        dphi_j = dphi_1D('T', fe_j, fe_q)
        xyz    = xyz_1D(fe_q)
        JxW    = JxW_1D(fe_q)

        # Boundary IDs
        left_edge   = 0
        right_edge  = 1

        dirichletBC = {}
        dirichletBC[left_edge]   = [
                                     ('T',  adoubleConstantFunction_1D(adouble(500), self.n_components)),
                                   ]
        dirichletBC[right_edge]  = [
                                     ('T',  adoubleConstantFunction_1D(adouble(500), self.n_components)),
                                   ]

        self.fun_Q = TemperatureSource_1D(L, tau, t, alpha)
        Q = function_adouble_value_1D('Q', self.fun_Q, xyz)
        
        # FE weak form terms
        accumulation = (phi_i * phi_j) * JxW
        diffusion    = (dphi_i * dphi_j) * alpha * JxW
        convection   = 0.0 * JxW
        source       = phi_i * Q * JxW

        weakForm = dealiiFiniteElementWeakForm_1D(Aij = diffusion + convection,
                                                  Mij = accumulation,
                                                  Fi  = source,
                                                  functionsDirichletBC = dirichletBC)

        # Setting the weak form of the FE system will declare a set of equations:
        # [Mij]{dx/dt} + [Aij]{x} = {Fi} and boundary integral equations
        self.fe_system.WeakForm = weakForm
        
        # Analytical solution
        eq = self.CreateEquation("u", "Analytical solution")
        x = eq.DistributeOnDomain(self.x, eClosedClosed)
        dx = L / (self.x.NumberOfPoints-1)
        x_ = (x() - 1) * dx
        eq.Residual = self.u(x) - (500 + (x_/L) * (x_/L - 1) * (t/tau))
        eq.CheckUnitsConsistency = False

class simTutorial(daeSimulation):
    def __init__(self, mesh, quadratureFormulaOrder):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_cv_5", mesh, quadratureFormulaOrder)
        self.m.Description = __doc__
        self.m.fe_model.Description = __doc__

    def SetUpParametersAndDomains(self):
        Nomega = self.m.fe_model.Domains[0].NumberOfPoints
        self.m.x.CreateArray(Nomega)

    def SetUpVariables(self):
        setFEInitialConditions(self.m.fe_model, self.m.fe_system, 'T', 500.0)

# Setup everything manually and run in a console
def simulate(mesh, quadratureFormulaOrder):
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeDelegateDataReporter()
    simulation   = simTutorial(mesh, quadratureFormulaOrder)

    # Do no print progress
    log.PrintProgress = False

    lasolver = pySuperLU.daeCreateSuperLUSolver()
    daesolver.SetLASolver(lasolver)

    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tutorial_cv_5-results')

    # Create three data reporters:
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

    # 3. Data
    dr = daeNoOpDataReporter()
    datareporter.AddDataReporter(dr)

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 3600
    simulation.TimeHorizon = 20*3600

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
    
    ###########################################
    #  Plots and data                         #
    ###########################################
    results = dr.Process.dictVariables
    Tvar = results[simulation.m.Name + '.HeatConduction.T']
    uvar = results[simulation.m.Name + '.u']
    Nx = simulation.m.x.NumberOfPoints
    L  = simulation.m.L
    times = numpy.linspace(0, L, Nx)
    T = Tvar.Values[-1,:] # 2D array [t,x]
    u = uvar.Values[-1,:] # 2D array [t,x]
   
    return times,T,u

def run(**kwargs):
    Nx1 = 5
    Nx2 = 20
    quadratureFormulaOrder = 2
    L = 100.0
    h1 = L / Nx1
    h2 = L / Nx2
    times1, T1, u1 = simulate('bar(0,100)-5.msh', quadratureFormulaOrder)
    times2, T2, u2 = simulate('bar(0,100)-20.msh', quadratureFormulaOrder)
    
    # The normalised global errors
    E1 = numpy.sqrt((1.0/Nx1) * numpy.sum((T1-u1)**2))
    E2 = numpy.sqrt((1.0/Nx2) * numpy.sum((T2-u2)**2))

    # Order of accuracy
    p = numpy.log(E1/E2) / numpy.log(h1/h2)
    C = E1 / h1**p
    
    print('\n\nOrder of Accuracy:')
    print('||E(h)|| is proportional to: C * (h**p)')
    print('||E(h1)|| = %e, ||E(h2)|| = %e' % (E1, E2))
    print('C = %e' % C)
    print('Order of accuracy (p) = %.2f' % p)
    
    fontsize = 14
    fontsize_legend = 11
    fig = plt.figure(figsize=(10,4), facecolor='white')
    title = 'Plots for coarse and fine grids (Order of accuracy = %.2f, quadrature order = %d) (cv_5)' % (p, quadratureFormulaOrder)
    fig.canvas.set_window_title(title)
    
    ax = plt.subplot(121)
    plt.plot(times1, T1, 'rs', label='T (FE)')
    plt.plot(times1, u1, 'b-', label='u (manufactured)')
    plt.xlabel('x, m', fontsize=fontsize)
    plt.ylabel('Temperature, K', fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)
    plt.xlim((0, 100))

    ax = plt.subplot(122)
    plt.plot(times2, T2, 'rs', label='T (FE)')
    plt.plot(times2, u2, 'b-', label='u (manufactured)')
    plt.xlabel('x, m', fontsize=fontsize)
    plt.ylabel('Temperature, K', fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)
    plt.xlim((0, 100))

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    run()
    
