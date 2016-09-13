#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                        tutorial_dealii_2.py
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
__doc__ = """
.. code-block:: none

                                                                       -->
    ID=0: Sun (45 degrees), gradient heat flux = 2 kW/m**2 in direction n = (1,-1)
      \ \ \ \ \ \ \
       \ \ \ \ \ \ \         ID=2: Inner tube: constant temperature of 300 K
        \ \ \ \ \ \ \       /
         \ \ \ \ \ \ \     /
          \ \ \ \ \ \ \   /
           \ \ \ \ \ ****/
            \ \ \ **    / **
             \ \**    **    **
              \ *    *  *    *
    ____________**    **    **_____________
                  **      **                y = -0.5
                 /   ****
                /
               /
             ID=1: Outer surface below y=-0.5, constant flux of 2 kW/m**2

    dT                                             
   ---- - ∇κ∇Τ = g, in Ω
    dt

Mesh:

.. image:: _static/pipe.png
   :width: 400 px

Results at t = 3600s:

.. image:: _static/tutorial_dealii_2-results.png
   :width: 600 px
"""

import os, sys, numpy, json, math, tempfile
from time import localtime, strftime
from daetools.pyDAE import *
from daetools.solvers.deal_II import *
from daetools.solvers.superlu import pySuperLU

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

# Neumann BC use either value() or gradient() functions.
# Dirichlet BC use vector_value() with n_component = multiplicity of the equation.
# Other functions use just the value().
#
# Nota bene:
#   This function is derived from Function_2D class and returns "double" value/gradient
class GradientFunction_2D(Function_2D):
    def __init__(self, gradient, direction, n_components = 1):
        Function_2D.__init__(self, n_components)
        self.m_gradient = Tensor_1_2D()
        self.m_gradient[0] = gradient * direction[0]
        self.m_gradient[1] = gradient * direction[1]
        
    def gradient(self, point, component = 0):
        if point.x < 0 and point.y > 0:
            return self.m_gradient
        else:
            return Tensor_1_2D()

    def vector_gradient(self, point):
        return [self.gradient(point, c) for c in range(self.n_components)]

# Nota bene:
#   This function is derived from adoubleFunction_2D class and returns "adouble" value
#   In this case, it is a function of daetools parameters/variables
class BottomGradientFunction_2D(adoubleFunction_2D):
    def __init__(self, model, n_components = 1):
        adoubleFunction_2D.__init__(self, n_components)
        self.model = model
        
    def value(self, point, component = 0):
        # Actual adouble expression can only be evaluated here in value function
        # which is called during DeclareEquations phase
        gradient = self.model.flux1() + self.model.flux2()
        
        if point.y < -0.5:
            return gradient
        else:
            return adouble(0.0)

    def vector_value(self, point):
        return [self.value(point, c) for c in range(self.n_components)]

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)
        
        # Used only to set the Neumann BC at the bottom to illustrate the use of adouble functions 
        # and coupling between daetools and deal.ii
        self.flux1 = daeParameter("flux1", unit(), self, "Flux as a parameter (half of the gradient)")
        self.flux2 = daeVariable ("flux2",   no_t, self, "Flux as a variable (half of the gradient)")

        rho = 8960.0  # kg/m**3
        cp  =  385.0  # J/(kg*K)
        k   =  401.0  # W/(m*K)
        
        self.flux_above   = 2.0E3/(rho*cp) # (W/m**2)/((kg/m**3) * (J/(kg*K))) = 
        self.flux_beneath = 5.0E3/(rho*cp) # (W/m**2)/((kg/m**3) * (J/(kg*K))) =
        self.diffusivity  = k / (rho*cp)   # m**2/s
        
        print('Thermal diffusivity = %f' % self.diffusivity)
        print('Beneath source flux = %f' % self.flux_beneath)
        print('Above source flux = %f x (1,-1)' % self.flux_above)

        functions    = {}
        functions['Diffusivity'] = ConstantFunction_2D(self.diffusivity)
        functions['Generation']  = ConstantFunction_2D(0.0)
        functions['Flux_a']      = GradientFunction_2D(self.flux_above, direction = (-1, 1)) # Gradient flux at boundary id=0 (Sun)
        functions['Flux_b']      = BottomGradientFunction_2D(self)                           # Flux as a function of daetools variables at boundary id=1 (outer tube where y < -0.5)
        
        dirichletBC    = {}
        dirichletBC[2] = [('T', ConstantFunction_2D(300))] # at boundary id=2 (inner tube)
        
        meshes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')
        mesh_file  = os.path.join(meshes_dir, 'pipe.msh')

        dofs = [dealiiFiniteElementDOF_2D(name='T',
                                          description='Temperature',
                                          fe = FE_Q_2D(1),
                                          multiplicity=1)]

        weakForm = dealiiFiniteElementWeakForm_2D(Aij = (dphi_2D('T', fe_i, fe_q) * dphi_2D('T', fe_j, fe_q)) * function_value_2D("Diffusivity", xyz_2D(fe_q)) * JxW_2D(fe_q),
                                                  Mij = (phi_2D('T', fe_i, fe_q) * phi_2D('T', fe_j, fe_q)) * JxW_2D(fe_q),
                                                  Fi  = phi_2D('T', fe_i, fe_q) * function_value_2D("Generation", xyz_2D(fe_q)) * JxW_2D(fe_q),
                                                  faceAij = {},
                                                  faceFi  = {0: phi_2D('T', fe_i, fe_q) * (function_gradient_2D("Flux_a", xyz_2D(fe_q)) * normal_2D(fe_q)) * JxW_2D(fe_q),
                                                             1: phi_2D('T', fe_i, fe_q) * function_adouble_value_2D("Flux_b", xyz_2D(fe_q)) * JxW_2D(fe_q)},
                                                  functions = functions,
                                                  functionsDirichletBC = dirichletBC)

        print('Heat conduction equation:')
        print('    Aij = %s' % str(weakForm.Aij))
        print('    Mij = %s' % str(weakForm.Mij))
        print('    Fi  = %s' % str(weakForm.Fi))
        print('    faceAij = %s' % str([item for item in weakForm.faceAij]))
        print('    faceFi  = %s' % str([item for item in weakForm.faceFi]))

        # Store the object so it does not go out of scope while still in use by daetools
        self.fe_dealII = dealiiFiniteElementSystem_2D(meshFilename    = mesh_file,     # path to mesh
                                                      quadrature      = QGauss_2D(3),  # quadrature formula
                                                      faceQuadrature  = QGauss_1D(3),  # face quadrature formula
                                                      dofs            = dofs,          # degrees of freedom
                                                      weakForm        = weakForm)      # FE system in weak form

        self.fe = daeFiniteElementModel('HeatConduction', self, 'Transient heat conduction through a pipe wall with an external heat flux', self.fe_dealII)
       
    def DeclareEquations(self):
        daeModel.DeclareEquations(self)
        
class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_deal_II_2")
        self.m.Description = __doc__
        
    def SetUpParametersAndDomains(self):
        self.m.flux1.SetValue(self.m.flux_beneath / 2.0)

    def SetUpVariables(self):
        self.m.flux2.AssignValue(self.m.flux_beneath / 2.0)
        
        m_dt = self.m.fe_dealII.Msystem()
        T    = self.m.fe.dictVariables['T']
        
        # dofIndexesMap relates global DOF indexes to points within daetools variables
        dofIndexesMap = {}
        for variable in self.m.fe.Variables:
            if variable.Name == 'T':
                ic = 273
            for i in range(variable.NumberOfPoints):
                dofIndexesMap[variable.OverallIndex + i] = (variable, i, ic)
        
        for row in range(m_dt.n):
            # Iterate over columns and set initial conditions.
            # If an item in the dt matrix is zero skip it (it is at the boundary - not a diff. variable).
            for column in self.m.fe_dealII.RowIndices(row):
                if m_dt(row, column).Node or m_dt(row, column).Value != 0:
                    variable, index, ic = dofIndexesMap[column]
                    variable.SetInitialCondition(index, ic)
                    #print '%s(%d) initial condition = %f' % (variable.Name, column, ic)
    
# Use daeSimulator class
def guiRun(app):
    datareporter = daeDelegateDataReporter()
    simulation = simTutorial()
    feDataReporter = simulation.m.fe_dealII.CreateDataReporter()
    datareporter.AddDataReporter(feDataReporter)

    # Connect datareporters
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    results_folder = tempfile.mkdtemp(suffix = '-results', prefix = 'tutorial_deal_II_2-')
    feDataReporter.Connect(results_folder, simName)
    try:
        from PyQt4 import QtCore, QtGui
        QtGui.QMessageBox.warning(None, "deal.II", "The simulation results will be located in: %s" % results_folder)
    except Exception as e:
        print(str(e))

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

    # Create two data reporters: TCP/IP and DealII
    feDataReporter = simulation.m.fe_dealII.CreateDataReporter()
    datareporter.AddDataReporter(feDataReporter)
    # Connect datareporters
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    feDataReporter.Connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tutorial_deal_II_2-results'), simName)

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 60    # 1 minute
    simulation.TimeHorizon       = 60*60 # 1 hour

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)
    
    # Save the model report and the runtime model report
    simulation.m.fe.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.fe.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

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
