#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial16.py
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
This tutorial shows how to use DAE Tools objects with Numpy arrays to solve a simple
stationary heat conduction in one dimension (1D Poisson equation) using a simple 
Finite Elements method (with linear elements):

d2T(x)/dx2 = F(x);  for all x in: (0, Lx)

Linear finite elements discretization and simple FE matrix assembly:

.. code-block:: none

                   phi                 phi
                      (k-1)               (k)
                      
                     *                   *             
                   * | *               * | *            
                 *   |   *           *   |   *          
               *     |     *       *     |     *        
             *       |       *   *       |       *      
           *         |         *         |         *        
         *           |       *   *       |           *      
       *             |     *       *     |             *    
     *               |   *           *   |               *  
   *                 | *  element (k)  * |                 *
 *-------------------*+++++++++++++++++++*-------------------*-
                     x                   x
                      (k-i                (k)
                    
                     \_________ _________/
                               |
                               dx

"""

import sys, numpy
from time import localtime, strftime
from daetools.pyDAE import *

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x  = daeDomain("x", self, unit(), "x axis domain")
        self.L  = daeParameter("L", unit(), self, "Length")
        self.Ta = daeVariable("Ta", no_t, self, "Temperature - analytical solution", [self.x])
        self.T1 = daeVariable("T1", no_t, self, "Temperature - first way", [self.x])
        self.T2 = daeVariable("T2", no_t, self, "Temperature - second way", [self.x])

    def local_dof(self, i):
        return self.mapLocalToGlobalIndices

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        N = self.x.NumberOfPoints
        Nelem = N - 1
        Ndofs_per_elem = 2 # since we use linear elements
        
        numpy.set_printoptions(linewidth=1e10)
        ##################################################################################
        # Analytical solution
        ##################################################################################
        dx = self.L.GetValue() / Nelem
        m = [0., 5./2, 4., 9./2]
        for i in range(N):
            eq = self.CreateEquation("Poisson_Analytical(%d)" % i)
            eq.Residual = self.Ta(i) - m[i] * dx**2

        ##################################################################################
        # First way: use constant global stiffness matrix and load array
        ##################################################################################
        print('***************************************************************************')
        print('    First way')
        print('***************************************************************************')
        # Create global stiffness matrix and load vector (dtype = float):
        A = numpy.zeros((N,N))        
        F = numpy.zeros(N)

        # Maps local indices within an element to the indices in the system's matrix
        mapLocalToGlobalIndices = [(0, 1), (1, 2), (2, 3)]   
        # ∆x is equidistant and constant
        dx = self.L.GetValue() / Nelem
        for el in range(Nelem):
            # Get global indices for the current element
            dof_indices = mapLocalToGlobalIndices[el]
            
            # Element stiffness matrix and load vector:
            #       | 1 -1 |          | 1 |
            # Ael = |-1  1 |    Fel = | 1 |
            #       
            Ael = (1/dx) * numpy.array( [[1, -1], [-1, 1]] )
            Fel = (dx/2) * numpy.array( [1, 1] )
            
            # Loop over element DOFs and update the global matrix and vector 
            for i in range(Ndofs_per_elem):
                for j in range(Ndofs_per_elem):
                    A[dof_indices[i], dof_indices[j]] += Ael[i,j]
                F[dof_indices[i]] += Fel[i]
        
        print('The global stiffness matrix (A) before applying boundary conditions:')
        print(A)
        print('The global load vector (F):')
        print(F)
        
        # Boundary conditions:
        # at x = 0: T(0) = 0     (Dirichlet BC)
        # at x = 1: dT(1)/dx = 0 (Neumann BC)
        A[0, 1:-1] = 0
        F[0] = 0
        print('The global stiffness matrix (A) after applying boundary conditions:')
        print(A)
        print('The global load vector (F) after applying boundary conditions:')
        print(F)
        
        # Create a vector of temperatures:
        T = numpy.empty(N, dtype=object)
        T[:] = [self.T1(i) for i in range(N)]
        
        # Generate the system equations
        for i in range(N):
            eq = self.CreateEquation("Poisson_ConstantStiffnessMatrix(%d)" % i)
            eq.Residual = numpy.sum(A[i, :] * T[:]) - F[i]

        ##################################################################################
        # Second way: use global stiffness matrix and load array that depend on DAE Tools
        #             model parameters/variables (not constant in a general case).
        #             Obviously, they are constant here - this is only to show the concept
        ##################################################################################
        print('***************************************************************************')
        print('    Second way')
        print('***************************************************************************')
        # Create global stiffness matrix and load vector (dtype = object). This matrix and
        # load vector will be functions of model parameters/variables.  
        # In this simple example that is not a case; however, the procedure is analogous.
        A = numpy.zeros((N,N), dtype=object)
        #A[:] = adouble(0, 0, True)
        #print A
        
        F = numpy.zeros(N, dtype=object)
        #F[:] = adouble(0, 0, True)
        #print F
        
        # Maps local indices within an element to the indices in the system's matrix
        mapLocalToGlobalIndices = [(0, 1), (1, 2), (2, 3)]
        # ∆x is equidistant but not constant (it depends on the parameter 'L')
        dx = self.L() / Nelem
        for el in range(Nelem):
            # Get global indices for the current element
            dof_indices = mapLocalToGlobalIndices[el]
            
            # Element stiffness matrix and load vector are the same:
            #       | 1 -1 |          | 1 |
            # Ael = |-1  1 |    Fel = | 1 |
            #
            Ael = (1 / dx) * numpy.array( [[1, -1], [-1, 1]] )
            Fel = (dx / 2) * numpy.array([1, 1])
            
            # Loop over element DOFs and update the global matrix and vector 
            for i in range(Ndofs_per_elem):
                for j in range(Ndofs_per_elem):
                    A[dof_indices[i], dof_indices[j]] += Ael[i,j]
                F[dof_indices[i]] += Fel[i]
        
        print('The global stiffness matrix (A) before applying boundary conditions:')
        print(A)
        print('The global load vector (F):')
        print(F)
        
        # Boundary conditions:
        # at x = 0: T(0) = 0     (Dirichlet BC)
        # at x = 1: dT(1)/dx = 0 (Neumann BC)
        A[0, 1:-1] = 0
        F[0] = 0
        print('The global stiffness matrix (A) after applying boundary conditions:')
        print(A)
        print('The global load vector (F) after applying boundary conditions:')
        print(F)
        
        # Create a vector of temperatures:
        T = numpy.empty(N, dtype=object)
        T[:] = [self.T2(i) for i in range(N)]
        
        # Generate the system equations
        for i in range(N):
            eq = self.CreateEquation("Poisson_NonConstantStiffnexMatrix(%d)" % i)
            eq.Residual = numpy.sum(A[i, :] * T[:]) - F[i]
            
class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial16")
        self.m.Description = __doc__
        
    def SetUpParametersAndDomains(self):
        self.m.x.CreateArray(4)
        self.m.L.SetValue(1)

    def SetUpVariables(self):
        pass
    
# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 10
    sim.TimeHorizon       = 1000
    simulator  = daeSimulator(app, simulation=sim)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 10
    simulation.TimeHorizon = 1000

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)

    # Save the model report and the runtime model report
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

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
