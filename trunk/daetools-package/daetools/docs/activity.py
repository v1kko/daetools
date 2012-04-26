"""
daetools.pyDAE.activity module is used to perform various types of activities. 
Currently steady state and dynamic simulation and optimization are supported.

Integer constants defined in the module:
    daeeStopCriterion
     - eStopAtGlobalDiscontinuity
     - eStopAtModelDiscontinuity
     - eDoNotStopAtDiscontinuity
"""

class daeSimulation_t:
    """
    The base simulation class (abstract).
    """
    def GetModel(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           daeModel object
        """
        pass
    
    def SetModel(self, Model):
        """
        (Abstract)
        ARGUMENTS:
         - Model: daeModel
        RETURNS:
           Nothing
        """
        pass
    
    def GetDataReporter(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           daeDataReporter_t object
        """
        pass
    
    def GetLog(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           daeLog_t object
        """
        pass
    
    def Run(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           Nothing
        """
        pass
    
    def Pause(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           Nothing
        """
        pass
    
    def Resume(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           Nothing
        """
        pass
    
    def GetTimeHorizon(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           float
        """
        pass
    
    def SetTimeHorizon(self, TimeHorizon):
        """
        (Abstract)
        ARGUMENTS:
         - TimeHorizon: float
        RETURNS:
           Nothing
        """
        pass
    
    def GetReportingInterval(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           float
        """
        pass
    
    def SetReportingInterval(self, ReportingInterval):
        """
        (Abstract)
        ARGUMENTS:
         - ReportingInterval: float
        RETURNS:
           Nothing
        """
        pass
    
    def ReportData(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           Nothing
        """
        pass
    
    def Initialize(self, DAESolver, DataReporter, Log, CalculateSensitivities = False, NumberOfObjectiveFunctions = 1):
        """
        (Abstract)
        ARGUMENTS:
         - DAESolver: daeDAESolver_t
         - DataReporter: daeDataReporter_t
         - Log: daeLog_t
         - CalculateSensitivities: bool
         - NumberOfObjectiveFunctions: unsigned integer
        RETURNS:
           Nothing
        """
        pass
    
    def Finalize(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           Nothing
        """
        pass
    
    def SolveInitial(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           Nothing
        """
        pass
    
    def GetDAESolver(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           daeDAESolver_t
        """
        pass
    
    def Reinitialize(self):
        """
        (Abstract)
        ARGUMENTS:
           Nothing
        RETURNS:
           None
        """
        pass
    
    def Reset(self):
        """
        (Abstract)
        ARGUMENTS:
           Nothing
        RETURNS:
           None
        """
        pass
    
    def ReRun(self):
        """
        (Abstract)
        ARGUMENTS:
           Nothing
        RETURNS:
           None
        """
        pass
    
    def Integrate(self, StopCriterion):
        """
        (Abstract)
        ARGUMENTS:
         - StopCriterion: daeeStopCriterion
        RETURNS:
           float
        """
        pass
    
    def IntegrateForTimeInterval(self, TimeInterval):
        """
        (Abstract)
        ARGUMENTS:
         - TimeInterval: float
        RETURNS:
           float
        """
        pass
    
    def IntegrateUntilTime(self, Time, StopCriterion):
        """
        (Abstract)
        ARGUMENTS:
         - Time: float
         - StopCriterion: daeeStopCriterion
        RETURNS:
           float
        """
        pass

    def SetUpParametersAndDomains(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           Nothing
        """
        pass
    
    def SetUpVariables(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           Nothing
        """
        pass
    
    def SetUpOptimization(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           Nothing
        """
        pass

    def StoreInitializationValues(self, fileName):
        """
        (Abstract)
        ARGUMENTS:
         - fileName: string
        RETURNS:
           Nothing
        """
        pass

    def LoadInitializationValues(self, fileName):
        """
        (Abstract)
        ARGUMENTS:
         - fileName: string
        RETURNS:
           Nothing
        """
        pass

class daeSimulation(daeSimulation_t):
    """
    An implementation of the daeSimulation_t.
    Properties:
     - OptimizationConstraints: list of daeOptimizationConstraint objects
     - OptimizationVariables: list of daeOptimizationVariable objects
     - ObjectiveFunction: daeObjectiveFunction object
    The user has to implement the following methods:
     - SetUpParametersAndDomains
     - SetUpVariables
     - SetUpOptimization (for optimization only)
    Also, the user can reimplement the Run() method to provide a custom operating procedure.  
    """
    def CreateEqualityConstraint(self, Description):
        """
        ARGUMENTS:
         - Description: string
        RETURNS:
           daeOptimizationConstraint object
        """
        pass

    def CreateInequalityConstraint(self, Description):
        """
        ARGUMENTS:
           None
        RETURNS:
           daeOptimizationConstraint object
        """
        pass

    def SetContinuousOptimizationVariable(self, Variable, LB, UB, defaultValue):
        """
        ARGUMENTS:
         - Variable: daeVariable|adouble
         - LB: float
         - UB: float
         - defaultValue: float
        RETURNS:
           Nothing
        """
        pass

    def SetIntegerOptimizationVariable(self, Variable, LB, UB, defaultValue):
        """
        ARGUMENTS:
         - Variable: daeVariable|adouble
         - LB: int
         - UB: int
         - defaultValue: int
        RETURNS:
           Nothing
        """
        pass

    def SetBinaryOptimizationVariable(self, Variable, defaultValue):
        """
        ARGUMENTS:
         - Variable: daeVariable|adouble
         - defaultValue: bool
        RETURNS:
           Nothing
        """
        pass

class daeOptimization_t:
    """
    The optimization class (abstract).
    """
    def Run(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           Nothing
        """
        pass
    
    def Initialize(self, Simulation, MINLPSolver, DAESolver, DataReporter, Log, NumberOfObjectiveFunctions = 1):
        """
        (Abstract)
        ARGUMENTS:
         - Simulation: daeSimulation_t object
         - MINLPSolver: daeMINLPSolver_t object
         - DAESolver: daeDAESolver_t object
         - DataReporter: daeDataReporter_t object
         - Log: daeLog_t object
         - NumberOfObjectiveFunctions: unsigned integer
        RETURNS:
           Nothing
        """
        pass
    
    def Finalize(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           Nothing
        """
        pass
    
class daeOptimization(daeOptimization_t):
    """
    An implementation of the daeOptimization_t class.
    """
