#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial_adv_4.py
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
This tutorial illustrates the OpenCS code generator.
For the given DAE Tools simulation it generates input files for OpenCS simulation,
either for a single CPU or for a parallel simulation using MPI.
The model is identical to the model in the tutorial 11.

The OpenCS framework currently does not support:
    
- Discontinuous equations (STNs and IFs)
- External functions
- Thermo-physical property packages

The temperature plot (at t=100s, x=0.5128, y=*):

.. image:: _static/tutorial_adv_4-results.png
   :width: 500px
"""

import sys, numpy, itertools, json
from time import localtime, strftime
from daetools.pyDAE import *

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

#from daetools.solvers.superlu import pySuperLU
from daetools.solvers.trilinos import pyTrilinos
from daetools.solvers.aztecoo_options import daeAztecOptions

# The linear solver used is iterative (GMRES); therefore decrease the abs.tol.
temperature_t.AbsoluteTolerance = 1e-2

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x = daeDomain("x", self, m, "X axis domain")
        self.y = daeDomain("y", self, m, "Y axis domain")

        self.Qb  = daeParameter("Q_b",         W/(m**2), self, "Heat flux at the bottom edge of the plate")
        self.Qt  = daeParameter("Q_t",         W/(m**2), self, "Heat flux at the top edge of the plate")
        self.rho = daeParameter("&rho;",      kg/(m**3), self, "Density of the plate")
        self.cp  = daeParameter("c_p",         J/(kg*K), self, "Specific heat capacity of the plate")
        self.k   = daeParameter("&lambda;_p",   W/(m*K), self, "Thermal conductivity of the plate")

        self.T = daeVariable("T", temperature_t, self)
        self.T.DistributeOnDomain(self.x)
        self.T.DistributeOnDomain(self.y)
        self.T.Description = "Temperature of the plate"

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # For readibility, get the adouble objects from parameters/variables
        # and create numpy arrays for T and its derivatives in tim and space
        # This will also save a lot of memory (no duplicate adouble objects in equations)
        Nx  = self.x.NumberOfPoints
        Ny  = self.y.NumberOfPoints
        rho = self.rho()
        cp  = self.cp()
        k   = self.k()
        Qb  = self.Qb()
        Qt  = self.Qt()

        T      = numpy.empty((Nx,Ny), dtype=object)
        dTdt   = numpy.empty((Nx,Ny), dtype=object)
        dTdx   = numpy.empty((Nx,Ny), dtype=object)
        dTdy   = numpy.empty((Nx,Ny), dtype=object)
        d2Tdx2 = numpy.empty((Nx,Ny), dtype=object)
        d2Tdy2 = numpy.empty((Nx,Ny), dtype=object)
        for x in range(Nx):
            for y in range(Ny):
                T[x,y]      = self.T(x,y)
                dTdt[x,y]   = dt(self.T(x,y))
                dTdx[x,y]   = d (self.T(x,y), self.x, eCFDM)
                dTdy[x,y]   = d (self.T(x,y), self.y, eCFDM)
                d2Tdx2[x,y] = d2(self.T(x,y), self.x, eCFDM)
                d2Tdy2[x,y] = d2(self.T(x,y), self.y, eCFDM)

        # Get the flat list of indexes
        indexes = [(x,y) for x,y in itertools.product(range(Nx), range(Ny))]
        eq_types = numpy.empty((Nx,Ny), dtype=object)
        eq_types[ : , : ] = 'i' # inner region
        eq_types[ : ,  0] = 'B' # bottom boundary
        eq_types[ : , -1] = 'T' # top boundary
        eq_types[  0, : ] = 'L' # left boundary
        eq_types[ -1, : ] = 'R' # right boundary
        print(eq_types.T) # print it transposed to visualize it more easily
        for x,y in indexes:
            eq_type = eq_types[x,y]
            eq = self.CreateEquation("HeatBalance", "")
            if eq_type == 'i':
                eq.Residual = rho*cp*dTdt[x,y] - k*(d2Tdx2[x,y] + d2Tdy2[x,y])

            elif eq_type == 'L':
                eq.Residual = dTdx[x,y]

            elif eq_type == 'R':
                eq.Residual = dTdx[x,y]

            elif eq_type == 'T':
                eq.Residual = -k*dTdy[x,y] - Qt

            elif eq_type == 'B':
                eq.Residual = -k*dTdy[x,y] - Qb

            else:
                raise RuntimeError('Invalid equation type: %s' % eq_type)

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_adv_4")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        self.m.x.CreateStructuredGrid(99, 0, 10.0)
        self.m.y.CreateStructuredGrid(99, 0, 10.0)

        self.m.k.SetValue(401 * W/(m*K))
        self.m.cp.SetValue(385 * J/(kg*K))
        self.m.rho.SetValue(8960 * kg/(m**3))
        self.m.Qb.SetValue(1e6 * W/(m**2))
        self.m.Qt.SetValue(0 * W/(m**2))

    def SetUpVariables(self):
        for x in range(1, self.m.x.NumberOfPoints - 1):
            for y in range(1, self.m.y.NumberOfPoints - 1):
                self.m.T.SetInitialCondition(x, y, 300 * K)

def run_code_generators(simulation, log):
    # Demonstration of daetools c++/MPI code-generator:
    import tempfile
    tmp_folder1 = tempfile.mkdtemp(prefix = 'daetools-code_generator-opencs-1cpu-')
    tmp_folder4 = tempfile.mkdtemp(prefix = 'daetools-code_generator-opencs-4cpu-')
    msg = 'Generated input files for the csSimulator will be located in: \n%s and: \n%s' % (tmp_folder1, tmp_folder4)
    log.Message(msg, 0)

    try:
        daeQtMessage("tutorial_adv_4", msg)
    except Exception as e:
        log.Message(str(e), 0)

    from daetools.code_generators.opencs import daeCodeGenerator_OpenCS
    from pyOpenCS import createGraphPartitioner_Metis, createGraphPartitioner_Simple
    
    cg = daeCodeGenerator_OpenCS()

    # Get default simulation options for DAE systems as a dictionary and
    # set the linear solver parameters.
    options = cg.defaultSimulationOptions_DAE
    options['LinearSolver']['Preconditioner']['Name']       = 'Amesos'
    options['LinearSolver']['Preconditioner']['Parameters'] = {"amesos: solver type": "Amesos_Klu"}

    # Generate input files for simulation on a single CPU
    cg.generateSimulation(simulation, 
                          tmp_folder1,
                          1,
                          simulationOptions = options)

    # Generate input files for parallel simulation on 4 CPUs
    gp = createGraphPartitioner_Metis('PartGraphRecursive')
    constraints = ['Ncs','Nnz','Nflops','Nflops_j'] # use all available constraints
    cg.generateSimulation(simulation,
                          tmp_folder4,
                          4, 
                          graphPartitioner = gp, 
                          simulationOptions = options, 
                          balancingConstraints = constraints)
    
def setupLASolver():
    lasolver = pyTrilinos.daeCreateTrilinosSolver("AztecOO", "")

    parameterList = lasolver.ParameterList
    
    lasolver.NumIters  = 1000
    lasolver.Tolerance = 1e-3
    
    parameterList.set_int("AZ_solver",    daeAztecOptions.AZ_gmres)
    parameterList.set_int("AZ_kspace",    500)
    parameterList.set_int("AZ_scaling",   daeAztecOptions.AZ_none)
    parameterList.set_int("AZ_reorder",   0)
    parameterList.set_int("AZ_conv",      daeAztecOptions.AZ_r0)
    parameterList.set_int("AZ_keep_info", 1)
    parameterList.set_int("AZ_output",    daeAztecOptions.AZ_none) # {AZ_all, AZ_none, AZ_last, AZ_summary, AZ_warnings}

    # Preconditioner options
    parameterList.set_int  ("AZ_precond",         daeAztecOptions.AZ_dom_decomp)
    parameterList.set_int  ("AZ_subdomain_solve", daeAztecOptions.AZ_ilu)
    parameterList.set_int  ("AZ_orthog",          daeAztecOptions.AZ_modified)
    parameterList.set_int  ("AZ_graph_fill",      1)    # default: 0
    parameterList.set_float("AZ_athresh",         1E-5) # default: 0.0
    parameterList.set_float("AZ_rthresh",         1.0)  # default: 0.0

    parameterList.Print()

    return lasolver

def run(**kwargs):
    # Prevent nodes being deleted after they are needed no longer.
    cfg = daeGetConfig()
    cfg.SetInteger('daetools.core.nodes.deleteNodesThreshold', 1000000)

    simulation = simTutorial()
    lasolver = setupLASolver()
    return daeActivity.simulate(simulation, reportingInterval        = 10, 
                                            timeHorizon              = 100,
                                            lasolver                 = lasolver,
                                            relativeTolerance        = 1e-3,
                                            run_before_simulation_fn = run_code_generators,
                                            **kwargs)

if __name__ == "__main__":
    guiRun = False if (len(sys.argv) > 1 and sys.argv[1] == 'console') else True
    run(guiRun = guiRun)
