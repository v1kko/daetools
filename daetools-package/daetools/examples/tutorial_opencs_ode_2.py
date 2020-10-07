#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_opencs_ode_2.py
                DAE Tools: pyOpenCS module, www.daetools.com
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
Reimplementation of CVodes cvsAdvDiff_bnd example.
The problem is simple advection-diffusion in 2-D::
    
    du/dt = d2u/dx2 + 0.5 du/dx + d2u/dy2

on the rectangle::
    
    0 <= x <= 2
    0 <= y <= 1
    
and simulated for 1 s.
Homogeneous Dirichlet boundary conditions are imposed, with the initial conditions::
    
    u(x,y,t=0) = x(2-x)y(1-y)exp(5xy)

The PDE is discretized on a uniform Nx+2 by Ny+2 grid with central differencing.
The boundary points are eliminated leaving an ODE system of size Nx*Ny.
The original results are in tutorial_opencs_ode_2.csv file.
"""

import os, sys, json, itertools, numpy
from daetools.solvers.opencs import csModelBuilder_t, csNumber_t, csSimulate
from daetools.examples.tutorial_opencs_aux import compareResults

class AdvectionDiffusion_2D:
    def __init__(self, Nx, Ny, u_bc):
        #In the CVode example cvsAdvDiff_bnd.c they only modelled interior points,
        #  excluded the boundaries from the ODE system, and assumed homogenous Dirichlet BCs (0.0).
        #There, they divided the 2D domain into (Nx+1) by (Ny+1) points and
        #  the points at x=0, x=Lx, y=0 and y=Ly are not used in the model.
        #Thus, x domain starts at x=1*dx, y domain starts at x=1*dy.
        self.Nx   = Nx
        self.Ny   = Ny
        self.u_bc = u_bc
        
        self.x0 = 0.0
        self.x1 = 2.0
        self.y0 = 0.0
        self.y1 = 1.0
        self.dx = (self.x1-self.x0) / (self.Nx+2-1)
        self.dy = (self.y1-self.y0) / (self.Ny+2-1)

        self.Nequations = self.Nx*self.Ny

    def GetInitialConditions(self):
        u0 = [0.0] * self.Nequations

        x0 = self.x0
        x1 = self.x1
        y0 = self.y0
        y1 = self.y1
        dx = self.dx
        dy = self.dy
        for ix in range(self.Nx):
            for iy in range(self.Ny):
                index = self.GetIndex(ix,iy)
                x = (ix+1) * dx
                y = (iy+1) * dy
                u0[index] = x*(x1 - x)*y*(y1 - y)*numpy.exp(5*x*y)
        return u0

    def GetVariableNames(self):
        return ['u(%d,%d)'%(x,y) for x,y in itertools.product(range(self.Nx), range(self.Ny))]

    def CreateEquations(self, y):
        # y is a list of csNumber_t objects representing model variables
        u_values = y
        dx = self.dx
        dy = self.dy
        Nx = self.Nx
        Ny = self.Ny

        def u(x, y):
            index = self.GetIndex(x,y)
            return u_values[index]
        
        # First order partial derivative per x.
        def du_dx(x, y):
            # If called for x == 0 or x == Nx-1 use the boundary value (u_bc = 0.0 in this example).
            ui1 = (self.u_bc if x == Nx-1 else u(x+1, y))
            ui2 = (self.u_bc if x == 0    else u(x-1, y))
            return (ui1 - ui2) / (2*dx)

        # First order partial derivative per y (not used in this example).
        def du_dy(x, y):
            # If called for y == 0 or y == Ny-1 use the boundary value (u_bc = 0.0 in this example).
            ui1 = (self.u_bc if y == Ny-1 else u(x, y+1))
            ui2 = (self.u_bc if y == 0    else u(x, y-1))
            return (ui1 - ui2) / (2*dy)

        # Second order partial derivative per x.
        def d2u_dx2(x, y):
            # If called for x == 0 or x == Nx-1 use the boundary value (u_bc = 0.0 in this example).
            ui1 = (self.u_bc if x == Nx-1 else u(x+1, y))
            ui  =                              u(x,   y)
            ui2 = (self.u_bc if x == 0    else u(x-1, y))
            return (ui1 - 2*ui + ui2) / (dx*dx)

        # Second order partial derivative per y.
        def d2u_dy2(x, y):
            # If called for y == 0 or y == Ny-1 use the boundary value (u_bc = 0.0 in this example).
            ui1 = (self.u_bc if y == Ny-1 else u(x, y+1))
            ui  =                              u(x,   y)
            ui2 = (self.u_bc if y == 0    else u(x, y-1))
            return (ui1 - 2*ui + ui2) / (dy*dy)

        eq = 0
        equations = [None] * self.Nequations
        for x in range(Nx):
            for y in range(Ny):
                equations[eq] = d2u_dx2(x,y) + 0.5 * du_dx(x,y) + d2u_dy2(x,y)
                eq += 1
        
        return equations
    
    def GetIndex(self, x, y):
        if x < 0 or x >= self.Nx:
            raise RuntimeError("Invalid x index")
        if y < 0 or y >= self.Ny:
            raise RuntimeError("Invalid y index")
        return self.Ny*x + y
    
def run(**kwargs):
    inputFilesDirectory = kwargs.get('inputFilesDirectory', os.path.splitext(os.path.basename(__file__))[0])
    Nx   = kwargs.get('Nx',   10)
    Ny   = kwargs.get('Ny',   5)
    u_bc = kwargs.get('u_bc', 0.0)
    
    # Instantiate the model being simulated.
    model = AdvectionDiffusion_2D(Nx, Ny, u_bc)
    
    # 1. Initialise the ODE system with the number of variables and other inputs.
    mb = csModelBuilder_t()
    mb.Initialize_ODE_System(model.Nequations, 0, defaultAbsoluteTolerance = 1e-6, defaultVariableName = 'u')
    
    # Create and set model equations using the provided time/variable/dof objects.
    # The ODE system is defined as:
    #     x' = f(x,y,t)
    # where x' are derivatives of state variables, x are state variables,
    # y are fixed variables (degrees of freedom) and t is the current simulation time.
    mb.ModelEquations = model.CreateEquations(mb.Variables)    
    # Set initial conditions
    mb.VariableValues = model.GetInitialConditions()
    # Set variable names.
    mb.VariableNames  = model.GetVariableNames()
    
    # 3. Generate a model for single CPU simulations.    
    # Set simulation options (specified as a string in JSON format).
    options = mb.SimulationOptions
    options['Simulation']['OutputDirectory']             = 'results'
    options['Simulation']['TimeHorizon']                 =  1.0
    options['Simulation']['ReportingInterval']           =  0.1
    options['Solver']['Parameters']['RelativeTolerance'] = 1e-5
    mb.SimulationOptions = options
    
    # Partition the system to create the OpenCS model for a single CPU simulation.
    # In this case (Npe = 1) the graph partitioner is not required.
    Npe = 1
    graphPartitioner = None
    cs_models = mb.PartitionSystem(Npe, graphPartitioner)
    csModelBuilder_t.ExportModels(cs_models, inputFilesDirectory, mb.SimulationOptions)
    print("OpenCS model generated successfully!")

    # 4. Run simulation using the exported model from the specified directory.
    csSimulate(inputFilesDirectory)
    
    # Compare OpenCS and the original model results.
    compareResults(inputFilesDirectory, ['u(0,0)', 'u(9,4)'])

if __name__ == "__main__":
    if len(sys.argv) == 1:
        Nx = 10
        Ny = 5
    elif len(sys.argv) == 3:
        Nx = int(sys.argv[1])
        Ny = int(sys.argv[2])
    else:
        print('Usage: python tutorial_opencs_ode_2.py Nx Ny')
        sys.exit()
        
    u_bc = 0.0
    inputFilesDirectory = 'tutorial_opencs_ode_2'
    run(Nx = Nx, Ny = Ny, u_bc = u_bc, inputFilesDirectory = inputFilesDirectory)
