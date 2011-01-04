"""
daetools.pyDAE.solver module implements several types of numerical solvers:
 - For solution of systems of equations:
    - System of linear equations (LA solver)
    - System of nonlinear equations (NLA solver)
    - System of differential and algebraic equations (DAE solver)
 - For optimization of:
    - Steady-state problems
    - Dynamic problems
Currently only DAE solvers are implemented.

Integer constants defined in the module:
    daeeIDALASolverType
     - eSundialsLU
     - eSundialsGMRES
     - eThirdParty
"""
  
class daeDAESolver_t:
    """
    DAE Solver interface.
    PROPERTIES:
     - Log: daeLog object
     - RelativeTolerance: float
     - InitialConditionMode: daeeInitialConditionMode
    """

class daeIDAS(daeDAESolver_t):
    """
    Sundials IDAS DAE solver wrapper.
    https://computation.llnl.gov/casc/sundials/main.html
    """
    def SetLASolver(self, LASolver):
        """
        Currently the following third part LA solvers are available:
         - Dense Lapack (Intel MKL, AMD ACML, Atlas) [serial]
         - Sparse Trilinos Amesos (Lapack, KLU, SuperLU, Umfpack) [serial or SMP]
         - Sparse Intel Pardiso [SMP]
        ARGUMENTS:
         - LASolver: daeIDALASolver object
        RETURNS:
           Nothing
        """
        pass

    def SaveMatrixAsXPM(self, Filename):
        """
        Saves the sparse matrix structure in .xpm image format.
        ARGUMENTS:
         - Filename: string
        RETURNS:
           Nothing
        """
        pass

class daeNLPSolver_t:
    """
    NLP Solver interface.
    """
    def Initialize(self, Simulation, DAESolver, DataReporter, Log):
        """
        (Abstract)
        ARGUMENTS:
         - Simulation: daeSimulation_t object
         - DAESolver: daeDAESolver_t object
         - DataReporter: daeDataReporter_t object
         - Log: daeLog_t object
        RETURNS:
           Nothing
        """
        pass

    def Solve(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           Nothing
        """
        pass

class daeBONMIN(daeNLPSolver_t):
    """
    BONMIN MINLP solver wrapper.
    https://projects.coin-or.org/Bonmin
    """
    def Initialize(self, Simulation, DAESolver, DataReporter, Log):
        """
        ARGUMENTS:
         - Simulation: daeSimulation_t object
         - DAESolver: daeDAESolver_t object
         - DataReporter: daeDataReporter_t object
         - Log: daeLog_t object
        RETURNS:
           Nothing
        """
        pass

    def Solve(self):
        """
        ARGUMENTS:
           None
        RETURNS:
           Nothing
        """
        pass
