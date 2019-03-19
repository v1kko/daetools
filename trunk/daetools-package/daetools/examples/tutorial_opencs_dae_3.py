#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_opencs_dae_3.py
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
Reimplementation of IDAS idasBruss_kry_bbd_p example.
The PDE system is a two-species time-dependent PDE known as
Brusselator PDE and models a chemically reacting system::

    du/dt = eps1(d2u/dx2  + d2u/dy2) + u^2 v - (B+1)u + A
    dv/dt = eps2(d2v/dx2  + d2v/dy2) - u^2 v + Bu

Boundary conditions: Homogenous Neumann.
Initial Conditions::
    
    u(x,y,t0) = u0(x,y) =  1  - 0.5*cos(pi*y/L)
    v(x,y,t0) = v0(x,y) = 3.5 - 2.5*cos(pi*x/L)

The PDEs are discretized by central differencing on a uniform (Nx, Ny) grid.
The model is described in:

- R. Serban and A. C. Hindmarsh. CVODES, the sensitivity-enabled ODE solver in SUNDIALS.
  In Proceedings of the 5th International Conference on Multibody Systems,
  Nonlinear Dynamics and Control, Long Beach, CA, 2005. ASME.
- M. R. Wittman. Testing of PVODE, a Parallel ODE Solver.
  Technical Report UCRL-ID-125562, LLNL, August 1996. 

The original results are in tutorial_opencs_dae_3.csv file.
"""

import os, sys, json, itertools
from daetools.pyDAE import *
import pyOpenCS
from pyOpenCS import csModelBuilder_t, csNumber_t, csGraphPartitioner_t, createGraphPartitioner_2D_Npde, csSimulate
from tutorial_opencs_aux import compareResults

eps1 = 0.002
eps2 = 0.002
A    = 1.000
B    = 3.400

class Brusselator_2D:
    def __init__(self, Nx, Ny, u_flux_bc, v_flux_bc):
        self.Nx = Nx
        self.Ny = Ny
        self.u_flux_bc = u_flux_bc
        self.v_flux_bc = v_flux_bc
        
        self.x0 = 0.0
        self.x1 = 1.0
        self.y0 = 0.0
        self.y1 = 1.0
        self.dx = (self.x1-self.x0) / (Nx-1)
        self.dy = (self.y1-self.y0) / (Ny-1)

        self.x_domain = []
        self.y_domain = []
        for x in range(self.Nx):
            self.x_domain.append(self.x0 + x * self.dx)
        for y in range(self.Ny):
            self.y_domain.append(self.y0 + y * self.dy)

        self.u_start_index = 0*Nx*Ny
        self.v_start_index = 1*Nx*Ny

        self.Nequations = 2*Nx*Ny

    def GetInitialConditions(self):
        # Use numpy array so that setting u_0 and v_0 changes the original values
        uv0 = numpy.zeros(self.Nequations)
        u_0 = uv0[self.u_start_index : self.v_start_index]
        v_0 = uv0[self.v_start_index : self.Nequations]

        dx = self.dx
        dy = self.dy
        x0 = self.x0
        x1 = self.x1
        y0 = self.y0
        y1 = self.y1
        
        pi = 3.1415926535898
        Lx = x1 - x0
        Ly = y1 - y0
        
        for ix in range(self.Nx):
            for iy in range(self.Ny):
                index = self.GetIndex(ix,iy)
                x = self.x_domain[ix]
                y = self.y_domain[iy]

                u_0[index] = 1.0 - 0.5 * numpy.cos(pi * y / Ly)
                v_0[index] = 3.5 - 2.5 * numpy.cos(pi * x / Lx)

        return uv0.tolist()

    def GetVariableNames(self):
        x_y_inds = [(x,y) for x,y in itertools.product(range(self.Nx), range(self.Ny))]
        return ['u(%d,%d)'%(x,y) for x,y in x_y_inds] + ['v(%d,%d)'%(x,y) for x,y in x_y_inds]

    def CreateEquations(self, y, dydt):
        # y is a list of csNumber_t objects representing model variables
        # dydt is a list of csNumber_t objects representing time derivatives of model variables
        u_values    = y   [self.u_start_index : self.v_start_index]
        v_values    = y   [self.v_start_index : self.Nequations]
        dudt_values = dydt[self.u_start_index : self.v_start_index]
        dvdt_values = dydt[self.v_start_index : self.Nequations]
        
        u_flux_bc = self.u_flux_bc
        v_flux_bc = self.v_flux_bc
        
        dx = self.dx
        dy = self.dy
        Nx = self.Nx
        Ny = self.Ny
        x_domain = self.x_domain
        y_domain = self.y_domain

        def u(x, y):
            index = self.GetIndex(x, y)
            return u_values[index]
        
        def v(x, y):
            index = self.GetIndex(x, y)
            return v_values[index]

        def du_dt(x, y):
            index = self.GetIndex(x, y)
            return dudt_values[index]

        def dv_dt(x, y):
            index = self.GetIndex(x, y)
            return dvdt_values[index]

        # First order partial derivative per x.
        def du_dx(x, y):
            if(x == 0): # left
                u0 = u(0, y)
                u1 = u(1, y)
                u2 = u(2, y)
                return (-3*u0 + 4*u1 - u2) / (2*dx)
            elif(x == Nx-1): # right
                un  = u(Nx-1,   y)
                un1 = u(Nx-1-1, y)
                un2 = u(Nx-1-2, y)
                return (3*un - 4*un1 + un2) / (2*dx)
            else:
                u1 = u(x+1, y)
                u2 = u(x-1, y)
                return (u1 - u2) / (2*dx)

        def dv_dx(x, y):
            if(x == 0): # left
                u0 = v(0, y)
                u1 = v(1, y)
                u2 = v(2, y)
                return (-3*u0 + 4*u1 - u2) / (2*dx)
            elif(x == Nx-1): # right
                un  = v(Nx-1,   y)
                un1 = v(Nx-1-1, y)
                un2 = v(Nx-1-2, y)
                return (3*un - 4*un1 + un2) / (2*dx)
            else:
                u1 = v(x+1, y)
                u2 = v(x-1, y)
                return (u1 - u2) / (2*dx)

        # First order partial derivative per y.
        def du_dy(x, y):
            if(y == 0): # bottom
                u0 = u(x, 0)
                u1 = u(x, 1)
                u2 = u(x, 2)
                return (-3*u0 + 4*u1 - u2) / (2*dy)
            elif(y == Ny-1): # top
                un  = u(x, Ny-1  )
                un1 = u(x, Ny-1-1)
                un2 = u(x, Ny-1-2)
                return (3*un - 4*un1 + un2) / (2*dy)
            else:
                ui1 = u(x, y+1)
                ui2 = u(x, y-1)
                return (ui1 - ui2) / (2*dy)

        def dv_dy(x, y):
            if(y == 0): # bottom
                u0 = v(x, 0)
                u1 = v(x, 1)
                u2 = v(x, 2)
                return (-3*u0 + 4*u1 - u2) / (2*dy)
            elif(y == Ny-1): # top
                un  = v(x, Ny-1  )
                un1 = v(x, Ny-1-1)
                un2 = v(x, Ny-1-2)
                return (3*un - 4*un1 + un2) / (2*dy)
            else:
                ui1 = v(x, y+1)
                ui2 = v(x, y-1)
                return (ui1 - ui2) / (2*dy)

        # Second order partial derivative per x.
        def d2u_dx2(x, y):
            if(x == 0 or x == Nx-1):
                raise RuntimeError("d2u_dx2 called at the boundary")

            ui1 = u(x+1, y)
            ui  = u(x,   y)
            ui2 = u(x-1, y)
            return (ui1 - 2*ui + ui2) / (dx*dx)

        def d2v_dx2(x, y):
            if(x == 0 or x == Nx-1):
                raise RuntimeError("d2v_dx2 called at the boundary")

            vi1 = v(x+1, y)
            vi  = v(x,   y)
            vi2 = v(x-1, y)
            return (vi1 - 2*vi + vi2) / (dx*dx)

        # Second order partial derivative per y.
        def d2u_dy2(x, y):
            if(y == 0 or y == Ny-1):
                raise RuntimeError("d2u_dy2 called at the boundary")

            ui1 = u(x, y+1)
            ui  = u(x,   y)
            ui2 = u(x, y-1)
            return (ui1 - 2*ui + ui2) / (dy*dy)

        def d2v_dy2(x, y):
            if(y == 0 or y == Ny-1):
                raise RuntimeError("d2v_dy2 called at the boundary")

            vi1 = v(x, y+1)
            vi  = v(x,   y)
            vi2 = v(x, y-1)
            return (vi1 - 2*vi + vi2) / (dy*dy)

        eq = 0
        equations = [None] * self.Nequations
        # Component u:
        for x in range(Nx):
            for y in range(Ny):
                if(x == 0):       # Left BC: Neumann BCs
                    equations[eq] = du_dx(x,y) - u_flux_bc
                elif(x == Nx-1):  # Right BC: Neumann BCs
                    equations[eq] = du_dx(x,y) - u_flux_bc
                elif(y == 0):     # Bottom BC: Neumann BCs
                    equations[eq] = du_dy(x,y) - u_flux_bc
                elif(y == Ny-1):  # Top BC: Neumann BCs
                    equations[eq] = du_dy(x,y) - u_flux_bc
                else:
                    # Interior points
                    equations[eq] = du_dt(x,y) \
                                    - eps1 * (d2u_dx2(x,y) + d2u_dy2(x,y)) \
                                    - (u(x,y)*u(x,y)*v(x,y) - (B+1)*u(x,y) + A)
                eq += 1
        
        # Component v:
        for x in range(Nx):
            for y in range(Ny):
                if(x == 0):       # Left BC: Neumann BCs
                    equations[eq] = dv_dx(x,y) - v_flux_bc
                elif(x == Nx-1):  # Right BC: Neumann BCs
                    equations[eq] = dv_dx(x,y) - v_flux_bc
                elif(y == 0):     # Bottom BC: Neumann BCs
                    equations[eq] = dv_dy(x,y) - v_flux_bc
                elif(y == Ny-1):  # Top BC: Neumann BCs
                    equations[eq] = dv_dy(x,y) - v_flux_bc
                else:
                    # Interior points
                    equations[eq] = dv_dt(x,y) \
                                    - eps2 * (d2v_dx2(x,y) + d2v_dy2(x,y)) \
                                    + u(x,y)*u(x,y)*v(x,y) - B*u(x,y)
                eq += 1
        
        return equations
    
    def GetIndex(self, x, y):
        if x < 0 or x >= self.Nx:
            raise RuntimeError("Invalid x index")
        if y < 0 or y >= self.Ny:
            raise RuntimeError("Invalid y index")
        return self.Ny*x + y
    
    
class myGraphPartitioner(csGraphPartitioner_t):
    # Divides vertices into Npe partitions without an analysis 
    def __init__(self):
        csGraphPartitioner_t.__init__(self)

    def GetName(self):
        return 'myGraphPartitioner'
    
    def Partition(self, Npe, Nvertices, Nconstraints, rowIndices, colIndices, vertexWeights):
        if Npe > Nvertices:
            raise RuntimeError('Npe larger than Nvertices')
        
        Neq_pe     = int(Nvertices / Npe)
        equations  = list(range(Nvertices))
        partitions = []
        for i in range(Npe):
            start = i     * Neq_pe
            end   = (i+1) * Neq_pe
            if i == Npe-1:
                end = Nvertices
            partitions.append(equations[start:end])
        print('myGraphPartitioner partitions: %s' % partitions)
        return partitions

def run(**kwargs):
    inputFilesDirectory = kwargs.get('inputFilesDirectory', os.path.splitext(os.path.basename(__file__))[0])
    Nx        = kwargs.get('Nx',        82)
    Ny        = kwargs.get('Ny',        82)
    u_flux_bc = kwargs.get('u_flux_bc', 0.0)
    v_flux_bc = kwargs.get('v_flux_bc', 0.0)
    
    # Instantiate the model being simulated.
    model = Brusselator_2D(Nx, Ny, u_flux_bc, v_flux_bc)
    
    # 1. Initialise the DAE system with the number of variables and other inputs.
    mb = csModelBuilder_t()
    mb.Initialize_DAE_System(model.Nequations, 0, defaultAbsoluteTolerance = 1e-5)
    
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
    options['Simulation']['TimeHorizon']                 = 20.0
    options['Simulation']['ReportingInterval']           =  0.1
    options['Solver']['Parameters']['RelativeTolerance'] = 1e-5
    
    # ILU options for Ncpu = 1: k = 3, rho = 1.0, alpha = 1e-1, w = 0.5
    options['LinearSolver']['Preconditioner']['Parameters']['fact: level-of-fill']      =    3
    options['LinearSolver']['Preconditioner']['Parameters']['fact: relax value']        =  0.5
    options['LinearSolver']['Preconditioner']['Parameters']['fact: absolute threshold'] = 1e-1
    options['LinearSolver']['Preconditioner']['Parameters']['fact: relative threshold'] =  1.0
    mb.SimulationOptions = options
    
    # Partition the system to create the OpenCS model for a single CPU simulation.
    # In this case (Npe = 1) the graph partitioner is not required.
    Npe = 1
    graphPartitioner = None
    cs_models = mb.PartitionSystem(Npe, graphPartitioner)
    csModelBuilder_t.ExportModels(cs_models, inputFilesDirectory, mb.SimulationOptions)
    print("Single CPU OpenCS model generated successfully!")

    # 4. Generate models for parallel simulations on message-passing multi-processors.    
    # ILU options for Ncpu = 8: k = 1, rho = 1.0, alpha = 1e-1, w = 0.0
    options['LinearSolver']['Preconditioner']['Parameters']['fact: level-of-fill']      =    1
    options['LinearSolver']['Preconditioner']['Parameters']['fact: relax value']        =  0.0
    options['LinearSolver']['Preconditioner']['Parameters']['fact: absolute threshold'] = 1e-1
    options['LinearSolver']['Preconditioner']['Parameters']['fact: relative threshold'] =  1.0
    mb.SimulationOptions = options
    
    # For distributed memory systems a graph partitioner must be specified.
    # Available partitioners:
    #   1. csGraphPartitioner_Simple (splits the given set of equations into Npe parts with no analysis)
    #   2. csGraphPartitioner_2D_Npde (distributes Npde equations on an uniform 2D grid split into Npe quadrants)
    #   3. csGraphPartitioner_Metis (partitions the set of equations using the METIS graph partitioner)
    #      Two algorithms are avaiable:
    #       - PartGraphKway:      'Multilevel k-way partitioning' algorithm
    #       - PartGraphRecursive: 'Multilevel recursive bisectioning' algorithm
    #   4. User-defined graph partitioners
    # The Metis partitioner can additionally balance the specified quantities in all partitions 
    # using the following balancing constraints:
    #  - Ncs:      balance number of compute stack items in equations
    #  - Nnz:      balance number of non-zero items in the incidence matrix
    #  - Nflops:   balance number of FLOPS required to evaluate equations
    #  - Nflops_j: balance number of FLOPS required to evaluate derivatives (Jacobian) matrix
    # PartitionSystem arguments:
    #  - Npe: unsigned int specifying the number of processing elements (CPUs)
    #  - graphPartitioner: csGraphPartitioner_t instance
    #  - balancingConstraints: a list of balancing constraints as strings 
    #  - logPartitionResults: bool (default False); if true a log file is created
    #  - unaryOperationsFlops: dictionary [csUnaryFunctions : unsigned int])
    #  - binaryOperationsFlops: dictionary [csBinaryFunctions : unsigned int]
    Npe = 8
    graphPartitioner = createGraphPartitioner_2D_Npde(Nx = model.Nx, 
                                                      Ny = model.Ny, 
                                                      Npde = 2, 
                                                      Npex_Npey_ratio = 2.0)
    inputFilesDirectory_mpi = inputFilesDirectory + ('-Npe=%d-2D_Npde' % Npe)
    cs_models_mpi = mb.PartitionSystem(Npe, graphPartitioner, logPartitionResults = True)
    csModelBuilder_t.ExportModels(cs_models_mpi, inputFilesDirectory_mpi, mb.SimulationOptions)
    print("Npe = %d CPUs OpenCS model generated successfully!" % Npe)

    # 5. Run simulation using the exported model from the specified directory.
    csSimulate(inputFilesDirectory)
    
    # Compare OpenCS and the original model results.
    compareResults(inputFilesDirectory, ['u(0,0)', 'u(81,81)'])
    
if __name__ == "__main__":
    Nx = 82
    Ny = 82
    u_flux_bc = 0.0
    v_flux_bc = 0.0
    inputFilesDirectory = 'tutorial_opencs_dae_3'
    run(Nx = Nx, 
        Ny = Ny, 
        u_flux_bc = u_flux_bc, 
        v_flux_bc = v_flux_bc, 
        inputFilesDirectory = inputFilesDirectory)
