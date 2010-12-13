"""
dae.activity module is used to perform various types of activities. Currently only steady state and dynamic simulation are supported.

Integer constants defined in the module:
    daeeStopCriterion
     - eStopAtGlobalDiscontinuity
     - eStopAtModelDiscontinuity
     - eDoNotStopAtDiscontinuity
"""

class daeActivity_t:
    """
    The base activity class (abstract).
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
    
    def Stop(self):
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

    
class daeDynamicActivity_t(daeActivity_t):
    """
    The base dynamic activity class (abstract).
    """
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
        ARGUMENTS:
           None
        RETURNS:
           Nothing
        """
        pass

class daeDynamicSimulation_t(daeDynamicActivity_t):
    """
    The base dynamic simulation class (abstract).
    """
    def Initialize(self, DAESolver, DataReporter, Log):
        """
        (Abstract)
        ARGUMENTS:
         - DAESolver: daeDAESolver_t
         - DataReporter: daeDataReporter_t
         - Log: daeLog_t
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

class daeDynamicSimulation(daeDynamicSimulation_t):
    """
    Default implementation of the dynamic simulation.
    The user has to implement the following methods:
     - SetUpParametersAndDomains
     - SetUpVariables
    Also, the user can reimplement the Run() method to provide a custom operating procedure of the process.  
    """
