#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_dealii_8.py
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
In this example a small parallel-plate reactor with an active surface is modelled.

This problem and its solution in `COMSOL Multiphysics <https://www.comsol.com>`_ software 
is described in the Application Gallery: `Transport and Adsorption (id=5)
<https://www.comsol.com/model/transport-and-adsorption-5>`_.

The transport in the bulk of the reactor is described by a convection-diffusion equation:
    
.. code-block:: none

   dc/dt - D*nabla^2(c) + div(uc) = 0 in Omega

The material balance for the surface, including surface diffusion and the reaction rate is:

.. code-block:: none

   dc_s/dt - Ds*nabla^2(c_s) = -k_ads * c * (Gamma_s - c_s) + k_des * c_s in Omega_s

For the bulk, the boundary condition at the active surface couples the rate of the reaction
at the surface with the flux of the reacting species and the concentration of the adsorbed
species and bulk species:
    
.. code-block:: none

    n⋅(-D*nabla(c) + c*u) = -k_ads*c*(Gamma_s - c_s) + k_des*c_s
    
The boundary conditions for the surface species are insulating conditions.

The problem is modelled using two coupled Finite Element systems: 
2D for bulk flow and 1D for the active surface. 
The linear interpolation is used to determine the bulk flow and active surface concentrations
at the interface.

The mesh is rectangular with the refined elements close to the left/right ends:

.. image:: _static/parallel_plate_reactor.png
   :height: 400 px

The cs plot at t = 2s:

.. image:: _static/tutorial_dealii_8-results.png
   :height: 400 px

The c plot at t = 2s:

.. image:: _static/tutorial_dealii_8-results2.png
   :height: 400 px

Animation:
    
.. image:: _static/tutorial_dealii_8-animation.gif
   :height: 400 px
"""

import os, sys, numpy, scipy.interpolate, operator, tempfile
from time import localtime, strftime
from daetools.pyDAE import *
from daetools.solvers.deal_II import *
from daetools.solvers.superlu import pySuperLU

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

# Inputs:
c0      = 1000  # [mol/m^3]     Initial concentration
k_ads   = 1e-6  # [m^3/(mol*s)] Forward rate constant
k_des   = 1e-9  # [1/s]	        Backward rate constant
Gamma_s = 1000  # [mol/m^2]     Active site concentration
Ds      = 1e-11 # [m^2/s]       Surface diffusivity
D       = 1e-9  # [m^2/s]       Gas diffusivity
v_max   = 1e-3  # [m/s]         Maximum velocity
delta   = 1e-4  # [m]           Channel width

# Boundary IDs (reactor)
left_edge      = 0
top_edge       = 1
right_edge     = 2 # excluding the active surface
bottom_edge    = 3
active_surface = 5

# Boundary IDs (active surface)
bottom_point = 0
top_point    = 1
      
dx = 1e-10
dy = 1e-10
def points_equal(x1, y1, x2, y2):
    return (x1 < x2+dx and x1 > x2-dx and y1 < y2+dy and y1 > y2-dy)

def coord_equal(x1, x2):
    #print(x1, x2, (x1 < x2+dx and x1 > x2-dx))
    return (x1 < x2+dx and x1 > x2-dx)

class VelocityFunction_2D(Function_2D):
    def __init__(self, n_components = 1):
        Function_2D.__init__(self, n_components)
        self.u = Tensor_1_2D()

    def gradient(self, point, component = 0):
        self.u[0] = 0.0
        self.u[1] = v_max * (1 - ((point.x-0.5*delta)/(0.5*delta))**2)
        return self.u

    def vector_gradient(self, point):
        return [self.value(point, c) for c in range(self.n_components)]

class c_interpolation(object):
    """
    Returns linear interpolation of the concentration based on y coordinates. 
    """
    def __init__(self, c_indexes):
        c_i = sorted(c_indexes.items(), key = operator.itemgetter(1))
        n = len(c_i)

        self.ci = numpy.empty((n), dtype=numpy.int_)    # array of indexes
        self.y  = numpy.empty((n), dtype=numpy.float_)  # array of y coordinates
        self.c  = numpy.empty((n), dtype=object)        # array of adouble objects

        for i, (ci, (y, c)) in enumerate(c_i):
            self.ci[i] = ci
            self.y[i]  = y
            self.c[i]  = c
            
    def get_c(self, y):
        if y < self.y[0] or y > self.y[-1]:
            print('y = %f out of bounds' % y)
            c = adouble(0.0)
        elif y == self.y[0]:
           c = self.c[0]
        elif y == self.y[-1]:
           c = self.c[-1]
        else:
            i0 = numpy.argwhere(self.y <= y)
            if len(i0):
                index = i0[-1][0]
                y0 = self.y[index]
                c0 = self.c[index]
            else:
                print('y = %f not found' % y)
                
            i1 = numpy.argwhere(self.y >= y)
            if len(i1):
                index = i1[0][0]
                y1 = self.y[index]
                c1 = self.c[index]
            else:
                print('y = %f not found' % y)
            
            c = (c0*(y1-y) + c1*(y-y0)) / (y1-y0)

        return c

class cs_interpolation_2D(c_interpolation, adoubleFunction_2D):
    def __init__(self, c_indexes, n_components = 1):
        c_interpolation.__init__(self, c_indexes)
        adoubleFunction_2D.__init__(self, n_components)
    
    def value(self, point, component = 0):
        return self.get_c(point.y)
            
    def vector_value(self, point):
        return [self.value(point, c) for c in range(self.n_components)]

class c_interpolation_1D(c_interpolation, adoubleFunction_1D):
    def __init__(self, c_indexes, n_components = 1):
        c_interpolation.__init__(self, c_indexes)
        adoubleFunction_1D.__init__(self, n_components)
    
    def value(self, point, component = 0):
        return self.get_c(point.x)
            
    def vector_value(self, point):
        return [self.value(point, c) for c in range(self.n_components)]


class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)
        
        self.FE_c_init()
        self.FE_cs_init()
        
    def FE_c_init(self):
        # FE system/model for the bulk gas concentration.
        dofs = [dealiiFiniteElementDOF_2D(name='c',
                                          description='Bulk gas concentration',
                                          fe = FE_Q_2D(1),
                                          multiplicity=1)]

        meshes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')
        mesh_file  = os.path.join(meshes_dir, 'parallel_plate_reactor.msh')

        self.fe_system_c = dealiiFiniteElementSystem_2D(meshFilename    = mesh_file,     # path to mesh
                                                        quadrature      = QGauss_2D(3),  # quadrature formula
                                                        faceQuadrature  = QGauss_1D(3),  # face quadrature formula
                                                        dofs            = dofs)          # degrees of freedom

        self.fe_model_c = daeFiniteElementModel('BulkFlow', self, 'Transient convection-diffusion-reaction equation', self.fe_system_c)
        
        self.c = self.fe_model_c.Variables[0]

    def FE_cs_init(self):
        # FE system/model for the active surface.
        dofs = [dealiiFiniteElementDOF_1D(name         = 'cs',
                                          description  = 'Surface concentration',
                                          fe           = FE_Q_1D(1),
                                          multiplicity = 1)]

        meshes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')
        mesh_file  = os.path.join(meshes_dir, 'active_surface.msh')

        self.fe_system_cs = dealiiFiniteElementSystem_1D(meshFilename    = mesh_file,     # path to mesh
                                                         quadrature      = QGauss_1D(3),  # quadrature formula
                                                         faceQuadrature  = QGauss_0D(3),  # face quadrature formula
                                                         dofs            = dofs)          # degrees of freedom

        self.fe_model_cs = daeFiniteElementModel('ActiveSurface', self, 'Transient diffusion-reaction equation', self.fe_system_cs)
        
        self.cs = self.fe_model_cs.Variables[0]
        
    def DeclareEquations(self):
        daeModel.DeclareEquations(self)
        
        # Populate dictionaries:
        # c_indexes = {overallIndex : (y-coord,  c(overallIndex))}
        # c_indexes = {overallIndex : (y-coord, cs(overallIndex))}
        # They will be used to get the concentration c/cs by linear interpolation.
        self.c_indexes = {}
        self.cs_indexes = {}

        c_sp  = list(self.fe_system_c.GetDOFSupportPoints())
        cs_sp = list(self.fe_system_cs.GetDOFSupportPoints()) 
        for ci, pcs in enumerate(cs_sp):
            self.cs_indexes[ci] = (pcs.x, self.cs(ci))
        for ci, pc in enumerate(c_sp):
            if pc.x >= delta - dx and pc.y >= 0.0 and pc.y <= delta + dy:
                self.c_indexes[ci] = (pc.y, self.c(ci))
        
        # Create weak formulations
        self.WeakForm_c()
        self.WeakForm_cs()

    def WeakForm_c(self):
        # Create some auxiliary objects for readability
        phi_i  =  phi_2D('c', fe_i, fe_q)
        phi_j  =  phi_2D('c', fe_j, fe_q)
        dphi_i = dphi_2D('c', fe_i, fe_q)
        dphi_j = dphi_2D('c', fe_j, fe_q)
        normal = normal_2D(fe_q)
        xyz    = xyz_2D(fe_q)
        JxW    = JxW_2D(fe_q)
        c_dof  = dof_approximation_2D('c', fe_q)

        dirichletBC = {}
        dirichletBC[bottom_edge] = [('c', adoubleConstantFunction_2D(adouble(c0)))]
        
        # Function<dim> wrapper
        self.fun_u_grad = VelocityFunction_2D()
        u_grad = function_gradient_2D("u", self.fun_u_grad, xyz)

        # adoubleFunction<dim> wrapper
        self.cs_fun = cs_interpolation_2D(self.cs_indexes)
        cs_fun = function_adouble_value_2D("cs_fun", self.cs_fun, xyz)

        # FE weak form terms
        c_accumulation = (phi_i * phi_j) * JxW
        c_diffusion    = (dphi_i * dphi_j) * D * JxW
        c_convection   = phi_i * (u_grad * dphi_j) * JxW
        c_source       = 0.0 * JxW

        c_faceFluxes = {}
        c_flux = -k_ads * c_dof * (Gamma_s - cs_fun) + k_des * cs_fun
        c_faceFluxes[active_surface] = phi_i * c_flux * JxW
        #c_faceFluxes[top_edge] = phi_i * (normal * u_grad) * c_dof * JxW

        weakForm = dealiiFiniteElementWeakForm_2D(Aij = c_diffusion + c_convection,
                                                  Mij = c_accumulation,
                                                  Fi  = c_source,
                                                  boundaryFaceFi = c_faceFluxes,
                                                  functionsDirichletBC = dirichletBC)

        self.fe_system_c.WeakForm = weakForm
        
    def WeakForm_cs(self):
        # Create some auxiliary objects for readability
        phi_i  =  phi_1D('cs', fe_i, fe_q)
        phi_j  =  phi_1D('cs', fe_j, fe_q)
        dphi_i = dphi_1D('cs', fe_i, fe_q)
        dphi_j = dphi_1D('cs', fe_j, fe_q)
        xyz    = xyz_1D(fe_q)
        JxW    = JxW_1D(fe_q)
        cs_dof = dof_approximation_1D('cs', fe_q)

        # adoubleFunction<dim> wrapper
        self.c_fun = c_interpolation_1D(self.c_indexes)
        c_fun = function_adouble_value_1D("c_fun", self.c_fun, xyz)

        # FE weak form terms
        cs_accumulation = (phi_i * phi_j) * JxW
        cs_diffusion    = (dphi_i * dphi_j) * Ds * JxW
        cs_flux         = k_ads * c_fun * (Gamma_s - cs_dof) - k_des * cs_dof
        cs_source       = phi_i * cs_flux * JxW

        weakForm = dealiiFiniteElementWeakForm_1D(Aij = cs_diffusion,
                                                  Mij = cs_accumulation,
                                                  Fi  = cs_source)

        self.fe_system_cs.WeakForm = weakForm

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_dealii_8")
        self.m.Description = __doc__
        self.m.fe_model_c.Description  = __doc__
        self.m.fe_model_cs.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        setFEInitialConditions(self.m.fe_model_c,  self.m.fe_system_c,  'c',  c0)
        setFEInitialConditions(self.m.fe_model_cs, self.m.fe_system_cs, 'cs', 0.0)
    
def run(**kwargs):
    guiRun = kwargs.get('guiRun', False)
    
    simulation = simTutorial()

    # Create SuperLU LA solver
    lasolver = pySuperLU.daeCreateSuperLUSolver()

    # Create and setup two data reporters:
    datareporter = daeDelegateDataReporter()
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if guiRun:
        results_folder_c  = tempfile.mkdtemp(suffix = '-results_c',  prefix = 'tutorial_deal_II_8-')
        results_folder_cs = tempfile.mkdtemp(suffix = '-results_cs', prefix = 'tutorial_deal_II_8-')
        daeQtMessage("deal.II", "The simulation results will be located in: %s and %s" % (results_folder_c, results_folder_cs))
    else:
        results_folder_c  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tutorial_deal_II_8-results-c')
        results_folder_cs = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tutorial_deal_II_8-results-cs')
        print("The simulation results will be located in: %s and %s" % (results_folder_c, results_folder_cs))
    
    # 1. deal.II for c
    feDataReporter_c = simulation.m.fe_system_c.CreateDataReporter()
    datareporter.AddDataReporter(feDataReporter_c)
    if not feDataReporter_c.Connect(results_folder_c, simName):
        sys.exit()

    # 2. deal.II for cs
    feDataReporter_cs = simulation.m.fe_system_cs.CreateDataReporter()
    datareporter.AddDataReporter(feDataReporter_cs)
    if not feDataReporter_cs.Connect(results_folder_cs, simName):
        sys.exit()

    # 3. TCP/IP
    tcpipDataReporter = daeTCPIPDataReporter()
    datareporter.AddDataReporter(tcpipDataReporter)
    if not tcpipDataReporter.Connect("", simName):
        sys.exit()

    return daeActivity.simulate(simulation, reportingInterval = 0.05, 
                                            timeHorizon       = 2,
                                            lasolver          = lasolver,
                                            datareporter      = datareporter,
                                            **kwargs)

if __name__ == "__main__":
    guiRun = False if (len(sys.argv) > 1 and sys.argv[1] == 'console') else True
    run(guiRun = guiRun)
