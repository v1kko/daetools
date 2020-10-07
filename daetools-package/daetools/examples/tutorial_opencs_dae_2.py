#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_opencs_dae_2.py
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
Reimplementation of DAE Tools tutorial1.py example.
A simple heat conduction problem: conduction through a very thin, rectangular copper plate::
    
    rho * cp * dT(x,y)/dt = k * [d2T(x,y)/dx2 + d2T(x,y)/dy2];  x in (0, Lx), y in (0, Ly)

Two-dimensional Cartesian grid (x,y) of 20 x 20 elements.
The original results are in tutorial_opencs_dae_2.csv file.
"""

import os, sys, json, itertools
from daetools.solvers.opencs import csModelBuilder_t, csNumber_t, csSimulate, csGraphPartitioner_t
from daetools.examples.tutorial_opencs_aux import compareResults

rho = 8960 # density, kg/m^3
cp  =  385 # specific heat capacity, J/(kg.K)
k   =  401 # thermal conductivity, W/(m.K)
Qb  =  1E5 # flux at the bottom edge, W/m^2
Tt  =  300 # T at the top edge, K

class HeatConduction_2D:
    def __init__(self, Nx, Ny):
        self.Nx = Nx
        self.Ny = Ny
        
        self.Lx = 0.1 # m
        self.Ly = 0.1 # m

        self.dx = self.Lx / (Nx-1)
        self.dy = self.Ly / (Ny-1)

        self.Nequations = Nx*Ny

    def GetInitialConditions(self):
        y0 = [300.0] * self.Nequations
        return y0

    def GetVariableNames(self):
        return ['T(%d,%d)'%(x,y) for x,y in itertools.product(range(self.Nx), range(self.Ny))]

    def CreateEquations(self, y, dydt):
        # y is a list of csNumber_t objects representing model variables
        # dydt is a list of csNumber_t objects representing time derivatives of model variables
        T_values = y
        T_derivs = dydt
        dx = self.dx
        dy = self.dy
        Nx = self.Nx
        Ny = self.Ny
        
        def T(x, y):
            index = self.GetIndex(x,y)
            return T_values[index]

        def dT_dt(x, y):
            index = self.GetIndex(x,y)
            return T_derivs[index]

        # First order partial derivative per x.
        def dT_dx(x, y):
            if (x == 0): # left
                T0 = T(0, y)
                T1 = T(1, y)
                T2 = T(2, y)
                return (-3*T0 + 4*T1 - T2) / (2*dx)
            elif (x == Nx-1): # right
                Tn  = T(Nx-1,   y)
                Tn1 = T(Nx-1-1, y)
                Tn2 = T(Nx-1-2, y)
                return (3*Tn - 4*Tn1 + Tn2) / (2*dx)
            else:
                T1 = T(x+1, y)
                T2 = T(x-1, y)
                return (T1 - T2) / (2*dx)

        # First order partial derivative per y.
        def dT_dy(x, y):
            if (y == 0): # bottom
                T0 = T(x, 0)
                T1 = T(x, 1)
                T2 = T(x, 2)
                return (-3*T0 + 4*T1 - T2) / (2*dy)
            elif (y == Ny-1): # top
                Tn  = T(x, Ny-1  )
                Tn1 = T(x, Ny-1-1)
                Tn2 = T(x, Ny-1-2)
                return (3*Tn - 4*Tn1 + Tn2) / (2*dy)
            else:
                Ti1 = T(x, y+1)
                Ti2 = T(x, y-1)
                return (Ti1 - Ti2) / (2*dy)

        # Second order partial derivative per x.
        def d2T_dx2(x, y):
            # This function is typically called only for interior points.
            if (x == 0 or x == Nx-1):
                raise RuntimeError("d2T_dx2 called for boundary x point")

            Ti1 = T(x+1, y)
            Ti  = T(x,   y)
            Ti2 = T(x-1, y)
            return (Ti1 - 2*Ti + Ti2) / (dx*dx)

        # Second order partial derivative per y.
        def d2T_dy2(x, y):
            # This function is typically called only for interior points.
            if (y == 0 or y == Ny-1):
                raise RuntimeError("d2T_dy2 called for boundary y point")

            Ti1 = T(x, y+1)
            Ti  = T(x,   y)
            Ti2 = T(x, y-1)
            return (Ti1 - 2*Ti + Ti2) / (dy*dy)

        eq = 0
        equations = [None] * self.Nequations
        for x in range(Nx):
            for y in range(Ny):
                if (x == 0):                                # Left BC: zero flux
                    equations[eq] = dT_dx(x,y)
                elif (x == Nx-1):                           # Right BC: zero flux
                    equations[eq] = dT_dx(x,y)
                elif (x > 0 and x < Nx-1 and y == 0):       # Bottom BC: prescribed flux
                    equations[eq] = -k * dT_dy(x,y) - Qb
                elif (x > 0 and x < Nx-1 and y == Ny-1):     # Top BC: prescribed flux
                    equations[eq] = T(x,y) - Tt
                else:                                       # Inner region: diffusion equation
                    equations[eq] = rho * cp * dT_dt(x,y) - k * (d2T_dx2(x,y) + d2T_dy2(x,y))
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
    Nx = kwargs.get('Nx', 20)
    Ny = kwargs.get('Ny', 20)
    
    # Instantiate the model being simulated.
    model = HeatConduction_2D(Nx, Ny)
    
    # 1. Initialise the DAE system with the number of variables and other inputs.
    mb = csModelBuilder_t()
    mb.Initialize_DAE_System(model.Nequations, 0, defaultAbsoluteTolerance = 1e-10)
    
    # 2. Specify the OpenCS model.
    # Create and set model equations using the provided time/variable/timeDerivative/dof objects.
    # The DAE system is defined as:
    #     F(x',x,y,t) = 0
    # where x' are derivatives of state variables, x are state variables,
    # y are fixed variables (degrees of freedom) and t is the current simulation time.
    mb.ModelEquations = model.CreateEquations(mb.Variables, mb.TimeDerivatives)    
    # Set initial conditions.
    mb.VariableValues = model.GetInitialConditions()
    # Set variable names.
    mb.VariableNames  = model.GetVariableNames()
    
    # 3. Generate a model for single CPU simulations.    
    # Set simulation options (specified as a string in JSON format).
    options = mb.SimulationOptions
    options['Simulation']['OutputDirectory']             = 'results'
    options['Simulation']['TimeHorizon']                 = 500.0
    options['Simulation']['ReportingInterval']           =   5.0
    options['Solver']['Parameters']['RelativeTolerance'] =  1e-5
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
    compareResults(inputFilesDirectory, ['T(0,0)'])

if __name__ == "__main__":
    if len(sys.argv) == 1:
        Nx = 20
        Ny = 20
    elif len(sys.argv) == 3:
        Nx = int(sys.argv[1])
        Ny = int(sys.argv[2])
    else:
        print('Usage: python tutorial_opencs_dae_2.py Nx Ny')
        sys.exit()
        
    inputFilesDirectory = 'tutorial_opencs_dae_2'
    run(Nx = Nx, Ny = Ny, inputFilesDirectory = inputFilesDirectory)
