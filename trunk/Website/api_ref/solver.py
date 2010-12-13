"""
dae.solver module implements several types of numerical solvers:
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
    PROPERTIES:
     - Log: daeLog object
     - RelativeTolerance: float
     - InitialConditionMode: daeeInitialConditionMode
    """
    def Initialize(self, Block, Log, InitialConditionMode):
        """
        ARGUMENTS:
         - Block: daeBlock object
         - Log: daeLog object
         - InitialConditionMode: daeeInitialConditionMode
         RETURNS:
           Nothing
        """
        pass

    def Reinitialize(self, CopyDataFromBlock):
        """
        ARGUMENTS:
         - CopyDataFromBlock: bool
         RETURNS:
           Nothing
        """
        pass

    def Solve(self, Time, StopCriterion):
        """
        ARGUMENTS:
         - Time: float
         - StopCriterion: daeeStopCriterion
        RETURNS:
           Current time: float
        """
        pass

class daeIDASolver(daeDAESolver_t):
    """
    Implementation of Sundials IDA DAE solver.
    https://computation.llnl.gov/casc/sundials/main.html
    """
    def SetLASolver(self, SolverType, LASolver = None):
        """
        Currently the following third part LA solvers are available:
         - Dense Lapack (Intel MKL, AMD ACMl, custom) [serial]
         - Sparse Trilinos Amesos [serial]
         - Sparse Intel Pardiso [serial or SMP]
        ARGUMENTS:
         - SolverType: daeeIDALASolverType
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

    def SaveMatrixAsPBM(self, Filename):
        """
        Saves sparse matrix structure in .pbm image format.
        ARGUMENTS:
         - Filename: string
        RETURNS:
           Nothing
        """
        pass
