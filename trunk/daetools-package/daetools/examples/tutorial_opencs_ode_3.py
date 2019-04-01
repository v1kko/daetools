#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_opencs_ode_3.py
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
Reimplementation of CVodes cvsDiurnal_kry example.
2-species diurnal kinetics advection-diffusion PDE system in 2D::

  dc(i)/dt = Kh*(d/dx)^2 c(i) + V*dc(i)/dx + (d/dy)(Kv(y)*dc(i)/dy) + Ri(c1,c2,t), i = 1,2

where::
    
  R1(c1,c2,t) = -q1*c1*c3 - q2*c1*c2 + 2*q3(t)*c3 + q4(t)*c2
  R2(c1,c2,t) =  q1*c1*c3 - q2*c1*c2 - q4(t)*c2
  Kv(y) = Kv0*exp(y/5)

Kh, V, Kv0, q1, q2, and c3 are constants, and q3(t) and q4(t) vary diurnally.
The problem is posed on the square::
    
    0 <= x <= 20 (km)
    30 <= y <= 50 (km)

with homogeneous Neumann boundary conditions, and integrated for 86400 sec (1 day).
The PDE system is discretised using the central differences on a uniform 10 x 10 mesh.
The original results are in tutorial_opencs_ode_3.csv file.
"""

import os, sys, json, itertools, numpy
from daetools.solvers.opencs import pyOpenCS
from daetools.solvers.opencs import csModelBuilder_t, csNumber_t, csSimulate
from daetools.solvers.opencs import csGraphPartitioner_t, createGraphPartitioner_2D_Npde
from daetools.examples.tutorial_opencs_aux import compareResults

V   =  1.00E-03
Kh  =  4.00E-06
Kv0 =  1.00E-08
q1  =  1.63E-16
q2  =  4.66E-16
C3  =  3.70E+16
a3  = 22.62
a4  =  7.601

class DiurnalKinetics_2D:
    def __init__(self, Nx, Ny):
        self.Nx = Nx
        self.Ny = Ny
        
        self.x0 =  0.0
        self.x1 = 20.0
        self.y0 = 30.0
        self.y1 = 50.0
        self.dx = (self.x1-self.x0) / (Nx-1)
        self.dy = (self.y1-self.y0) / (Ny-1)

        self.x_domain = []
        self.y_domain = []
        for x in range(self.Nx):
            self.x_domain.append(self.x0 + x * self.dx)
        for y in range(self.Ny):
            self.y_domain.append(self.y0 + y * self.dy)

        self.C1_start_index = 0*Nx*Ny
        self.C2_start_index = 1*Nx*Ny

        self.Nequations = 2*Nx*Ny

    def GetInitialConditions(self):
        # Use numpy array so that setting C1_0 and C2_0 changes the original values
        C0 = numpy.zeros(self.Nequations)
        C1_0 = C0[self.C1_start_index : self.C2_start_index]
        C2_0 = C0[self.C2_start_index : self.Nequations]

        dx = self.dx
        dy = self.dy
        x0 = self.x0
        x1 = self.x1
        y0 = self.y0
        y1 = self.y1
        
        def alfa(x):
            xmid = (x1+x0)/2.0
            cx = 0.1 * (x - xmid)
            cx2 = cx*cx
            return 1 - cx2 + 0.5*cx2*cx2
        
        def beta(y):
            ymid = (y1+y0)/2.0
            cy = 0.1 * (y - ymid)
            cy2 = cy*cy
            return 1 - cy2 + 0.5*cy2*cy2
        
        for ix in range(self.Nx):
            for iy in range(self.Ny):
                index = self.GetIndex(ix,iy)
                x = self.x_domain[ix]
                y = self.y_domain[iy]

                C1_0[index] = 1E6  * alfa(x) * beta(y)
                C2_0[index] = 1E12 * alfa(x) * beta(y)
        
        return C0.tolist()

    def GetVariableNames(self):
        x_y_inds = [(x,y) for x,y in itertools.product(range(self.Nx), range(self.Ny))]
        return ['C1(%d,%d)'%(x,y) for x,y in x_y_inds] + ['C2(%d,%d)'%(x,y) for x,y in x_y_inds]

    def CreateEquations(self, y, time):
        # y is a list of csNumber_t objects representing model variables
        C1_values = y[self.C1_start_index : self.C2_start_index]
        C2_values = y[self.C2_start_index : self.Nequations]
        
        dx = self.dx
        dy = self.dy
        Nx = self.Nx
        Ny = self.Ny
        x_domain = self.x_domain
        y_domain = self.y_domain

        # Math. functions
        sin    = pyOpenCS.sin
        acos   = pyOpenCS.acos
        exp    = pyOpenCS.exp
        cs_max = pyOpenCS.max

        nonzero = csNumber_t(1E-30)
        
        def Kv(y):
            return Kv0 * numpy.exp(y_domain[y] / 5.0)

        def q3(time):
            w = numpy.arccos(-1.0) / 43200.0
            sinwt = cs_max(nonzero, sin(w*time))
            return exp(-a3 / sinwt)

        def q4(time):
            w = numpy.arccos(-1.0) / 43200.0
            sinwt = cs_max(nonzero, sin(w*time))
            return exp(-a4 / sinwt)

        def R1(c1, c2, time):
            return -q1*c1*C3 - q2*c1*c2 + 2*q3(time)*C3 + q4(time)*c2

        def R2(c1, c2, time):
            return q1*c1*C3 - q2*c1*c2 - q4(time)*c2

        def C1(x, y):
            index = self.GetIndex(x, y)
            return C1_values[index]
        
        def C2(x, y):
            index = self.GetIndex(x, y)
            return C2_values[index]

        # First order partial derivative per x.
        def dC1_dx(x, y):
            # If called for x == 0 or x == Nx-1 return 0.0 (zero flux through boundaries).
            if(x == 0 or x == Nx-1):
                return csNumber_t(0.0)

            ci1 = C1(x+1, y)
            ci2 = C1(x-1, y)
            return (ci1 - ci2) / (2*dx)

        def dC2_dx(x, y):
            # If called for x == 0 or x == Nx-1 return 0.0 (zero flux through boundaries).
            if(x == 0 or x == Nx-1):
                return csNumber_t(0.0)

            ci1 = C2(x+1, y)
            ci2 = C2(x-1, y)
            return (ci1 - ci2) / (2*dx)

        # First order partial derivative per y.
        def dC1_dy(x, y):
            # If called for y == 0 or y == Ny-1 return 0.0 (zero flux through boundaries).
            if(y == 0 or y == Ny-1):
                return csNumber_t(0.0)

            ci1 = C1(x, y+1)
            ci2 = C1(x, y-1)
            return (ci1 - ci2) / (2*dy)

        def dC2_dy(x, y):
            # If called for y == 0 or y == Ny-1  return 0.0 (zero flux through boundaries).
            if(y == 0 or y == Ny-1):
                return csNumber_t(0.0)

            ci1 = C2(x, y+1)
            ci2 = C2(x, y-1)
            return (ci1 - ci2) / (2*dy)

        # Second order partial derivative per x.
        def d2C1_dx2(x, y):
            # If called for x == 0 or x == Nx-1 return 0.0 (no diffusion through boundaries).
            if(x == 0 or x == Nx-1):
                return csNumber_t(0.0)

            ci1 = C1(x+1, y)
            ci  = C1(x,   y)
            ci2 = C1(x-1, y)
            return (ci1 - 2*ci + ci2) / (dx*dx)

        def d2C2_dx2(x, y):
            # If called for x == 0 or x == Nx-1 return 0.0 (no diffusion through boundaries).
            if(x == 0 or x == Nx-1):
                return csNumber_t(0.0)

            ci1 = C2(x+1, y)
            ci  = C2(x,   y)
            ci2 = C2(x-1, y)
            return (ci1 - 2*ci + ci2) / (dx*dx)

        # Second order partial derivative per y.
        def d2C1_dy2(x, y):
            # If called for y == 0 or y == Ny-1 return 0.0 (no diffusion through boundaries).
            if(y == 0 or y == Ny-1):
                return csNumber_t(0.0)

            ci1 = C1(x, y+1)
            ci  = C1(x,   y)
            ci2 = C1(x, y-1)
            return (ci1 - 2*ci + ci2) / (dy*dy)

        def d2C2_dy2(x, y):
            # If called for y == 0 or y == Ny-1 return 0.0 (no diffusion through boundaries).
            if(y == 0 or y == Ny-1):
                return csNumber_t(0.0)

            ci1 = C2(x, y+1)
            ci  = C2(x,   y)
            ci2 = C2(x, y-1)
            return (ci1 - 2*ci + ci2) / (dy*dy)

        eq = 0
        equations = [None] * self.Nequations
        # Component 2 (C1):
        for x in range(Nx):
            for y in range(Ny):
                equations[eq] = V  * dC1_dx(x,y) +  \
                                Kh * d2C1_dx2(x,y) + \
                                Kv(y) * (0.2 * dC1_dy(x,y) + d2C1_dy2(x,y)) + \
                                R1(C1(x,y), C2(x,y), time)                     
                eq += 1
        
        # Component 2 (C2):
        for x in range(Nx):
            for y in range(Ny):
                equations[eq] = V  * dC2_dx(x,y) + \
                                Kh * d2C2_dx2(x,y) + \
                                Kv(y) * (0.2 * dC2_dy(x,y) + d2C2_dy2(x,y)) + \
                                R2(C1(x,y), C2(x,y), time)
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
    Nx = kwargs.get('Nx', 80)
    Ny = kwargs.get('Ny', 80)
    
    # Instantiate the model being simulated.
    model = DiurnalKinetics_2D(Nx, Ny)
    
    # 1. Initialise the ODE system with the number of variables and other inputs.
    mb = csModelBuilder_t()
    mb.Initialize_ODE_System(model.Nequations, 0, defaultAbsoluteTolerance = 1e-5)
    
    # Create and set model equations using the provided time/variable/dof objects.
    # The ODE system is defined as:
    #     x' = f(x,y,t)
    # where x' are derivatives of state variables, x are state variables,
    # y are fixed variables (degrees of freedom) and t is the current simulation time.
    mb.ModelEquations = model.CreateEquations(mb.Variables, mb.Time)    
    # Set initial conditions
    mb.VariableValues = model.GetInitialConditions()
    # Set variable names.
    mb.VariableNames  = model.GetVariableNames()
    
    # 3. Generate a model for single CPU simulations.    
    # Set simulation options (specified as a string in JSON format).
    options = mb.SimulationOptions
    options['Simulation']['OutputDirectory']             = 'results'
    options['Simulation']['TimeHorizon']                 = 86400.0
    options['Simulation']['ReportingInterval']           =   100.0
    options['Solver']['Parameters']['RelativeTolerance'] =    1e-5
    
    # ILU options for Ncpu = 1: k = 1, rho = 1.0, alpha = 1e-5, w = 0.0
    options['LinearSolver']['Preconditioner']['Parameters']['fact: level-of-fill']      =    1
    options['LinearSolver']['Preconditioner']['Parameters']['fact: relax value']        =  0.0
    options['LinearSolver']['Preconditioner']['Parameters']['fact: absolute threshold'] = 1e-5
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
                                                      Npex_Npey_ratio = 0.5)
    inputFilesDirectory_mpi = inputFilesDirectory + '-Npe=%d-2D_Npde' % Npe
    cs_models_mpi = mb.PartitionSystem(Npe, graphPartitioner, logPartitionResults = True)
    csModelBuilder_t.ExportModels(cs_models_mpi, inputFilesDirectory_mpi, mb.SimulationOptions)
    print("Npe = %d CPUs OpenCS model generated successfully!" % Npe)

    # 5. Run simulation using the exported model from the specified directory.
    csSimulate(inputFilesDirectory)
    
    # Compare OpenCS and the original model results.
    compareResults(inputFilesDirectory, ['C1(0,0)'])
    
if __name__ == "__main__":
    if len(sys.argv) == 1:
        Nx = 80
        Ny = 80
    elif len(sys.argv) == 3:
        Nx = int(sys.argv[1])
        Ny = int(sys.argv[2])
    else:
        print('Usage: python tutorial_opencs_ode_3.py Nx Ny')
        sys.exit()
        
    inputFilesDirectory = 'tutorial_opencs_ode_3'
    run(Nx = Nx, Ny = Ny, inputFilesDirectory = inputFilesDirectory)
