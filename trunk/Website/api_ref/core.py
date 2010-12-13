"""
dae.core module contains classes necessary to build models. 
Every model consists of the following information:
 - Domains
 - Parameters
 - Variables
 - Equations
 - Ports
 - State transition networks
 - Child models

Integer constants defined in the module:
    daeeDomainType (Defines a type of a domain; it can be arrray or distributed)
     - eArray
     - eDistributed

    daeeParameterType (Defines a type of a parameter; it can be real, integer or boolean)
     - eReal
     - eInteger
     - eBool

    daeePortType (Defines a type of a port; it can be inlet or outlet)
     - eInletPort
     - eOutletPort

    daeeDiscretizationMethod (Discretization methods available; these can be center, foward or backward finite difference method)
     - eCFDM
     - eFFDM
     - eBFDM

    daeeDomainBounds (Defines points of the domain that the equation is distributed on)
     - eOpenOpen (The whole domain except the left and the right bound; equivalent to (LB,UB) )
     - eOpenClosed (The whole domain except the left bound; equivalent to (LB,UB] )
     - eClosedOpen (The whole domain except the right bound; equivalent to [LB,UB) )
     - eClosedClosed (The whole domain; equivalent to [LB,UB] )
     - eLowerBound (The lower bound; LB )
     - eUpperBound (The upper bound; UB )

    daeeInitialConditionType (Defines a type of IC; it can be algebraic or differential)
     - eAlgebraicIC
     - eDifferentialIC

    daeeDomainIndexType
     - eConstantIndex
     - eDomainIterator

    daeeRangeType
     - eRangeConstantIndex
     - eRangeDomainIterator
     - eRange

    daeIndexRangeType
     - eAllPointsInDomain
     - eRangeOfIndexes
     - eCustomRange
"""

class daeObject:
    """
    A base class for all other classes in the module.
    PROPERTIES:
     - Name: string
     - CanonicalName: string
     - Description: string
    """
    pass
    
def daeSaveModel(Model, Filename):
    """
    Saves the model definition into .xml file
    ARGUMENTS:
     - Model: daeModel (model object)
     - Filename: string (.xml file name)
    """
    pass

class daeVariableType:
    """
    PROPERTIES:
     - Name: string
     - Units: string
     - LowerBound: float
     - UpperBound: float
     - InitialGuess: float
     - AbsoluteTolerance: float
    """
    def __init__(self, Name, Units, LowerBound, UpperBound, InitialGuess, AbsoluteTolerance):
        """
        ARGUMENTS:
        - Name: string
        - Units: string
        - LowerBound: float
        - UpperBound: float
        - InitialGuess: float
        - AbsoluteTolerance: float
        """
        pass
   
class daeDomain(daeObject):
    """
    PROPERTIES:
     - Type: daeeDomainType (read-only)
     - NumberOfIntervals: unsigned int (read-only)
     - NumberOfPoints: unsigned int (read-only)
     - Points: list of floats
     - DiscretizationMethod: daeeDiscretizationMethod (read-only)
     - DiscretizationOrder: unsigned int (read-only)
     - LowerBound: float (read-only)
     - UpperBound: float (read-only)
    """
    def __init__(self, Model):
        """ 
        ARGUMENTS:
         - Model: daeModel object
        """
        pass

    def __init__(self, Port):
        """ 
        ARGUMENTS:
         - Port: daePort object
        """
        pass

    def CreateArray(self, NoIntervals):
        """ 
        A function called from within simulation object to create a discrete domain (that is an array). 
        ARGUMENTS:
         - NoIntervals: unsigned int
        RETURNS:
           Nothing
        """
        pass
    
    def CreateDistributed(self, DiscretizationMethod, Order, NoIntervals, LowerBound, UpperBound):
        """
        A function called from within simulation object to create a distributed domain. 
        ARGUMENTS:
         - DiscretizationMethod: daeeDiscretizationMethod 
         - Order: unsigned int (currently only 2nd order is implemented)
         - NoIntervals: unsigned int
         - LowerBound: float
         - UpperBound: float
        RETURNS:
           Nothing
        """
        pass
    
    def __call__(self, Indexes = None):
        """
        Function call operator () used to create an index range object.
        Index ranges are used as arguments in functions array(), dt_array(), ...
        ARGUMENTS:
         - Indexes
           There are three overloads:
            - domain()
              If called with no arguments, the function will return daeIndexRange object including all points within the domain
            - domain(Indexes)
              If argument Indexes is a list of unsigned ints, the function will return daeIndexRange object including the points given in the list
            - domain(start, end, step)
              If argument Indexes is a slice: start,end,step, the function will return daeIndexRange object including the points defined by the slice object
        RETURNS:
           daeIndexRange object
        """
        pass

    def __getitem__(self, Index):
        """
        Operator [] used to get the value at the specified point within the domain.
        ARGUMENTS:
         - Index: unsigned int
        RETURNS:
           adouble
        """
        pass
    
    def GetPoint(self, Index):
        """
        Function to access the raw data (to get the value at the specified point within the domain).
        ARGUMENTS:
         - Index: unsigned int
        RETURNS:
           float
        """
        pass

    def GetNumPyArray(self):
        """
        Operator [] used to get the value at the specified point within the domain.
        ARGUMENTS:
           None
        RETURNS:
           one-dimensional numpy array of length NumberOfPoints
        """
        pass

class daeDomainIndex:
    """
    Used as an argument of the operator () in daeParameter/daeVariable classes to obtain the value of the variable for specified domain indexes.
    Usually created automatically by the framework. It contains either Index (unsigned int) or DEDI (daeDEDI object) property depending on its type.
    PROPERTIES:
     - Type: daeeDomainIndexType (read-only)
     - Index: unsigned int (read-only), valid if the Type = eConstantIndex
     - DEDI: daeDEDI object  (read-only), valid if the Type = eDomainIterator
    """
    def _init_(self, Index):
        """ 
        ARGUMENTS:
         - Index: unsigned int
        """
        pass
    
    def _init_(self, DEDI):
        """ 
        ARGUMENTS:
         - DEDI: daeDEDI object
        """
        pass

class daeIndexRange:
    """
    Used as an argument of 'array' functions in daeParameter/daeVariable classes to create an array of parameters/variables.
    PROPERTIES:
     - Domain: daeDomain object (read-only)
     - NoPoints: number of points (read-only)
     - Type: daeIndexRangeType (read-only)
     - StartIndex: unsigned int (read-only), valid if the Type = eRangeOfIndexes
     - EndIndex: unsigned int (read-only), valid if the Type = eRangeOfIndexes
     - Step: unsigned int (read-only), valid if the Type = eRangeOfIndexes
    """
    def _init_(self, Domain):
        """ 
        ARGUMENTS:
         - Domain: daeDomain object
        """
        pass
    
    def _init_(self, Domain, Indexes):
        """ 
        ARGUMENTS:
         - Domain: daeDomain object
         - Indexes: list of unsigned ints
        """
        pass

    def _init_(self, Domain, StartIndex, EndIndex, Step):
        """ 
        ARGUMENTS:
         - Domain: daeDomain object
         - StartIndex: int
         - EndIndex: int
         - Step: int
        """
        pass
   
class daeArrayRange:
    """
    Used as an argument of 'array' functions in daeParameter/daeVariable classes to create an array of parameters/variables.
    Usually created automatically by the framework. 
    PROPERTIES:
     - NoPoints: number of points (read-only)
     - Type: daeeRangeType (read-only)
     - Range: daeIndexRange object (read-only), valid if the Type = eRange
     - Index: unsigned int (read-only), valid if the Type = eRangeConstantIndex
     - DEDI: daeDEDI object (read-only), valid if the Type = eRangeDomainIterator
    """
    def _init_(self, Index):
        """ 
        ARGUMENTS:
         - Index: unsigned int
        """
        pass
    
    def _init_(self, DEDI):
        """ 
        ARGUMENTS:
         - DEDI: daeDEDI object
        """
        pass

    def _init_(self, IndexRange):
        """ 
        ARGUMENTS:
         - IndexRange: daeIndexRange object
        """
        pass

class daeDEDI(daeObject):
    """
    Also known as daeDistributedEquationDomainInfo. Used as an argument of 'operator ()' and 'array' functions in daeParameter/daeVariable classes.
    Always created automatically by the framework (returned by daeEquation::DistributeOnDomain)
    """
    def __call__(self):
        """ 
        ARGUMENTS:
           None
        RETURNS:
           adouble object
        """
        pass

class daeParameter(daeObject):
    """
    PROPERTIES:
     - Type: daeeParameterType
     - Domains: daeDomain list
    """
    def __init__(self, Name, Type, Parent):
        """
        ARGUMENTS:
         - Name: string
         - Type: daeeParameterType
         - Parent: daeModel | daePort object
        """
        pass
    
    def DistributeOnDomain(self, Domain):
        """
        ARGUMENTS:
         - Domain: daeDomain object
        RETURNS:
          Nothing
        """
        pass
    
    def SetValue(self, D1, D2, D3, D4, D5, D6, D7, D8, Value):
        """
        Overloaded function, with 0 to 8 arguments for (D1 - D8):
         - SetValue(Value)
         - SetValue(D1, Value)
         - SetValue(D1, D2, Value)
           ...
         - SetValue(D1, D2, D3, D4, D5, D6, D7, D8, Value)
        Used to set the value of the parameter (in SetUpParametersAndDomains() function).
        ARGUMENTS:
         - D1 to D8: unsigned int(s)
         - Value: float
        RETURNS:
          Nothing
        """
        pass
    
    def GetValue(self, D1, D2, D3, D4, D5, D6, D7, D8):
        """
        Overloaded function, with 0 to 8 arguments for (D1 - D8):
         - GetValue(Value)
         - GetValue(D1, Value)
         - GetValue(D1, D2, Value)
           ...
         - GetValue(D1, D2, D3, D4, D5, D6, D7, D8, Value)
        Used to get the value of the parameter. It is rarely used. 
        ARGUMENTS:
         - D1 to D8: unsigned int(s)
        RETURNS:
          float
        """
        pass
    
    def __call__(self, O1, O2, O3, O4, O5, O6, O7, O8):
        """
        Overloaded function call operator, with 0 to 8 arguments for (O1 - O8):
         - param()
         - param(O1)
         - param(O1, O2)
           ...
         - param(O1, O2, O3, O4, O5, O6, O7, O8)
        Used only for declaring equation residual (to get a value of the parameter).
        ARGUMENTS:
         - O1 to O8: daeDomainIndex | daeDEDI | unsigned int
        RETURNS:
          adouble object
        """
        pass
        
    def array(self, O1, O2, O3, O4, O5, O6, O7, O8):
        """
        Overloaded function, with 0 to 8 arguments for (O1 - O8):
         - array()
         - array(O1)
         - array(O1, O2)
           ...
         - array(O1, O2, O3, O4, O5, O6, O7, O8)
        Used only for declaring equation residual (to create an array of parameters).
        ARGUMENTS:
         - O1 to O8: daeIndexRange | daeDEDI | unsigned int
        RETURNS:
          adouble_array object
        """
        pass
    
    def GetNumPyArray(self):
        """
        Used to wrap parameter's values into the multi-dimensional numpy array.
        Dimensions are defined by the number of points in the domains that the parameter is distributed on.
        If the parameter is not distributed on any domains then one-dimensional numpy array of length 1 is returned.
        ARGUMENTS:
           None
        RETURNS:
           multi-dimensional numpy array
        """
        pass

class daeVariable(daeObject):
    """
    PROPERTIES:
     - Domains: daeDomain list
     - ReportingOn: boolean
    """
    def __init__(self, Name, VariableType, Parent):
        """
        ARGUMENTS:
         - Name: string
         - VariableType: daeVariableType object
         - Parent: daeModel | daePort object
        """
        pass
    
    def DistributeOnDomain(self, Domain):
        """
        ARGUMENTS:
         - Domain: daeDomain object
        RETURNS:
          Nothing
        """
        pass
    
    def SetValue(self, D1, D2, D3, D4, D5, D6, D7, D8, Value):
        """
        Overloaded function, with 0 to 8 arguments for (D1 - D8):
         - SetValue(Value)
         - SetValue(D1, Value)
         - SetValue(D1, D2, Value)
           ...
         - SetValue(D1, D2, D3, D4, D5, D6, D7, D8, Value)
        Used to set the value of the variable and can be used ONLY AFTER successful initialization (after the call to SolveInitial() function).
        IT SHOULD NOT BE USED DIRECTLY IN SIMULATION (YOU SHOULD KNOW EXACTLY WHAT YOU ARE DOING!) SINCE IT ACCESSES THE VARIABLE RAW DATA AND CAN AFFECT THE SOLVER!! 
        ARGUMENTS:
         - D1 to D8: unsigned int(s)
         - Value: float
        RETURNS:
          Nothing
        """
        pass
    
    def GetValue(self, D1, D2, D3, D4, D5, D6, D7, D8):
        """
        Overloaded function, with 0 to 8 arguments for (D1 - D8):
         - GetValue(Value)
         - GetValue(D1, Value)
         - GetValue(D1, D2, Value)
           ...
         - GetValue(D1, D2, D3, D4, D5, D6, D7, D8, Value)
        Used to get the value of the variable.
        It is rarely used. It access the variable raw data and can be used ONLY AFTER successful initialization (after the call to SolveInitial() function) 
        ARGUMENTS:
         - D1 to D8: unsigned int(s)
        RETURNS:
          float
        """
        pass
    
    def AssignValue(self, D1, D2, D3, D4, D5, D6, D7, D8, Value):
        """
        Overloaded function, with 0 to 8 arguments for (D1 - D8):
         - AssignValue(Value)
         - AssignValue(D1, Value)
         - AssignValue(D1, D2, Value)
           ...
         - AssignValue(D1, D2, D3, D4, D5, D6, D7, D8, Value)
        Used to set Degrees Of Freedom of the model (in SetUpVariables() function).
        ARGUMENTS:
         - D1 to D8: unsigned int(s)
         - Value: float
        RETURNS:
          Nothing
        """
        pass

    def ReAssignValue(self, D1, D2, D3, D4, D5, D6, D7, D8, Value):
        """
        Overloaded function, with 0 to 8 arguments for (D1 - D8):
         - ReAssignValue(Value)
         - ReAssignValue(D1, Value)
         - ReAssignValue(D1, D2, Value)
           ...
         - ReAssignValue(D1, D2, D3, D4, D5, D6, D7, D8, Value)
        Used to reset the value of the assigned variable previously set by AssignValue() function (in Run() function).
        NOTE: Once you are done with ReAssigning values/ReSetting initial conditions you must call the function Reinitialize() from the simulation class!
        ARGUMENTS:
         - D1 to D8: unsigned int(s)
         - Value: float
        RETURNS:
          Nothing
        """
        pass

    def __call__(self, O1, O2, O3, O4, O5, O6, O7, O8):
        """
        Overloaded function call operator, with 0 to 8 arguments for (O1 - O8):
         - var()
         - var(O1)
         - var(O1, O2)
           ...
         - var(O1, O2, O3, O4, O5, O6, O7, O8)
        Used only for declaring equation residual (to get a value of the variable).
        ARGUMENTS:
         - O1 to O8: daeDomainIndex | daeDEDI | unsigned int
        RETURNS:
          adouble object
        """
        pass
        
    def d(self, Domain, O1, O2, O3, O4, O5, O6, O7, O8):
        """
        Overloaded function, with 0 to 8 arguments for (O1 - O8):
         - d()
         - d(O1)
         - d(O1, O2)
           ...
         - d(O1, O2, O3, O4, O5, O6, O7, O8)
        Used only for declaring equation residual (to get a partial derivative of 1st order of the variable).
        ARGUMENTS:
         - Domain: daeDomain object
         - O1 to O8: daeDomainIndex | daeDEDI | unsigned int
        RETURNS:
          adouble object
        """
        pass

    def d2(self, Domain, O1, O2, O3, O4, O5, O6, O7, O8):
        """
        Overloaded function, with 0 to 8 arguments for (O1 - O8):
         - d2()
         - d2(O1)
         - d2(O1, O2)
           ...
         - d2(O1, O2, O3, O4, O5, O6, O7, O8)
        Used only for declaring equation residual (to get a partial derivative of 2nd order of the variable).
        ARGUMENTS:
         - Domain: daeDomain object
         - O1 to O8: daeDomainIndex | daeDEDI | unsigned int
        RETURNS:
          adouble object
        """
        pass

    def dt(self, O1, O2, O3, O4, O5, O6, O7, O8):
        """
        Overloaded function, with 0 to 8 arguments for (O1 - O8):
         - dt()
         - dt(O1)
         - dt(O1, O2)
           ...
         - dt(O1, O2, O3, O4, O5, O6, O7, O8)
        Used only for declaring equation residual (to get a time derivative of the variable).
        ARGUMENTS:
         - O1 to O8: daeDomainIndex | daeDEDI | unsigned int
        RETURNS:
          adouble object
        """
        pass

    def array(self, O1, O2, O3, O4, O5, O6, O7, O8):
        """
        Overloaded function, with 0 to 8 arguments for (O1 - O8):
         - array()
         - array(O1)
         - array(O1, O2)
           ...
         - array(O1, O2, O3, O4, O5, O6, O7, O8)
        Used only for declaring equation residual (to create an array of variables).
        ARGUMENTS:
         - O1 to O8: daeIndexRange | daeDEDI | unsigned int
        RETURNS:
          adouble_array object
        """
        pass
    
    def d_array(self, Domain, O1, O2, O3, O4, O5, O6, O7, O8):
        """
        Overloaded function, with 0 to 8 arguments for (O1 - O8):
         - d_array()
         - d_array(O1)
         - d_array(O1, O2)
           ...
         - d_array(O1, O2, O3, O4, O5, O6, O7, O8)
        Used only for declaring equation residual (to create an array of partial derivatives of 1st order).
        ARGUMENTS:
         - Domain: daeDomain object
         - O1 to O8: daeIndexRange | daeDEDI | unsigned int
        RETURNS:
          adouble_array object
        """
        pass
    
    def d2_array(self, Domain, O1, O2, O3, O4, O5, O6, O7, O8):
        """
        Overloaded function, with 0 to 8 arguments for (O1 - O8):
         - d2_array()
         - d2_array(O1)
         - d2_array(O1, O2)
           ...
         - d2_array(O1, O2, O3, O4, O5, O6, O7, O8)
        Used only for declaring equation residual (to create an array of partial derivatives of 2nd order).
        ARGUMENTS:
         - Domain: daeDomain object
         - O1 to O8: daeIndexRange | daeDEDI | unsigned int
        RETURNS:
          adouble_array object
        """
        pass

    def dt_array(self, O1, O2, O3, O4, O5, O6, O7, O8):
        """
        Overloaded function, with 0 to 8 arguments for (O1 - O8):
         - dt_array()
         - dt_array(O1)
         - dt_array(O1, O2)
           ...
         - dt_array(O1, O2, O3, O4, O5, O6, O7, O8)
        Used only for declaring equation residual (to create an array of time derivatives).
        ARGUMENTS:
         - O1 to O8: daeIndexRange | daeDEDI | unsigned int
        RETURNS:
          adouble_array object
        """
        pass

    def SetInitialGuess(self, D1, D2, D3, D4, D5, D6, D7, D8, InitialGuess):
        """
        Overloaded function, with 0 to 8 arguments for (D1 - D8):
         - SetInitialGuess(InitialGuess)
         - SetInitialGuess(D1, InitialGuess)
         - SetInitialGuess(D1, D2, InitialGuess)
           ...
         - SetInitialGuess(D1, D2, D3, D4, D5, D6, D7, D8, InitialGuess)
        Used to set the initial guess of the variable (in SetUpVariables() function).
        ARGUMENTS:
         - D1 to D8: unsigned int(s)
         - InitialGuess: float
        RETURNS:
          Nothing
        """
        pass

    def SetInitialCondition(self, D1, D2, D3, D4, D5, D6, D7, D8, InitialCondition):
        """
        Overloaded function, with 0 to 8 arguments for (D1 - D8):
         - SetInitialCondition(Value)
         - SetInitialCondition(D1, Value)
         - SetInitialCondition(D1, D2, Value)
           ...
         - SetInitialCondition(D1, D2, D3, D4, D5, D6, D7, D8, Value)
        Used to set the initial condition of the variable (in SetUpVariables() function).
        ARGUMENTS:
         - D1 to D8: unsigned int(s)
         - InitialCondition: float
        RETURNS:
          Nothing
        """
        pass

    def ReSetInitialCondition(self, D1, D2, D3, D4, D5, D6, D7, D8, InitialCondition):
        """
        Overloaded function, with 0 to 8 arguments for (D1 - D8):
         - ReSetInitialCondition(Value)
         - ReSetInitialCondition(D1, Value)
         - ReSetInitialCondition(D1, D2, Value)
           ...
         - ReSetInitialCondition(D1, D2, D3, D4, D5, D6, D7, D8, Value)
        Used to reset the initial condition of the variable (in Run() function).
        NOTE: Once you are done with ReAssigning values/ReSetting initial conditions you must call the function Reinitialize() from the simulation class!
        ARGUMENTS:
         - D1 to D8: unsigned int(s)
         - InitialCondition: float
        RETURNS:
          Nothing
        """
        pass

    def SetAbsoluteTolerances(self, AbsoluteTolerance):
        """
        Used to set the absolute tolerance for all points in all domains of the variable (in SetUpVariables() function).
        ARGUMENTS:
         - AbsoluteTolerance: float
        RETURNS:
          Nothing
        """
        pass
    
    def SetInitialGuesses(self, InitialGuess):
        """
        Used to set the initial guesses for all points in all domains of the variable (in SetUpVariables() function).
        ARGUMENTS:
         - InitialGuess: float
        RETURNS:
          Nothing
        """
        pass
    
    def GetNumPyArray(self):
        """
        Used to wrap parameter's values into the multi-dimensional numpy array.
        Dimensions are defined by the number of points in the domains that the variable is distributed on.
        If the variable is not distributed on any domains then one-dimensional numpy array of length 1 is returned.
        ARGUMENTS:
           None
        RETURNS:
           multi-dimensional numpy array
        """
        pass

class daePort(daeObject):
    """
    ATTRIBUTES:
     - Type: daeePortType
     - Domains: daeDomain list
     - Parameters: daeParameter list
     - Variables: daeVariable list
    """
    def __init__(self, Name, Type, Model):
        """ 
        ARGUMENTS:
         - Name: string
         - Type: daeePortType
         - Model: daeModel object
        """
        pass

    def SetReportingOn(self, On):
        """
        ARGUMENTS:
         - On: boolean
        RETURNS:
           Nothing
        """
        pass

class daeEquation(daeObject):
    """
    PROPERTIES:
     - Domains: daeDomain list
     - Residual: adouble
    """
    def DistributeOnDomain(self, Domain, Bounds):
        """
        ARGUMENTS:
         - Domain: daeDomain
         - Bounds: daeeDomainBounds | list of unsigned ints
        RETURNS:
           daeDEDI object
        """
        pass

class daeState(daeObject):
    """
     - NumberOfEquations: unsigned int
     - NumberOfStateTransitions: unsigned int
     - NumberOfNestedSTNs: unsigned int
     - Equations: daeEquation list
     - StateTransitions: daeStateTransition list
     - NestedSTNs: daeSTN list
    """
    pass

class daeStateTransition(daeObject):
    """
     - StateFrom: daeState
     - StateTo: daeState
     - Condition: daeCondition
    """
    pass

class daeSTN(daeObject):
    """
     - NumberOfStates: unsigned int
     - States: daeState list
     - ActiveState: daeState
     - ParentState: daeState
    """
    def SetActiveState(self, StateName):
        """
        ARGUMENTS:
         - StateName: string
        RETURNS:
           Nothing
        """
        pass

class daeIF(daeSTN):
    """
     - NumberOfStates: unsigned int
     - States: daeState list
     - ActiveState: daeState
     - ParentState: daeState
    """
    pass
  
class daeModel(daeObject):
    """
     - Domains: daeDomain list
     - Parameters: daeParameter list
     - Variables: daeVariable list
     - Equations: daeEquation list
     - STNs: daeSTN list
     - Ports: daePort list
     - ChildModels: daeModel list
     - PortArrays: daePortArray list
     - ChildModelArrays: daeModelArray list
     - InitialConditionMode: daeeInitialConditionMode
    """
    def __init__(self, Name, Parent = None):
        """
        (Abstract)
        ARGUMENTS:
         - Name: string
         - Parent: daeModel object
        """
        pass
    
    def DeclareEquations(self):
        """
        (Abstract)
        The function used to provide a definition of the model equations and state transition networks.
        ARGUMENTS:
           None
        RETURNS:
           Nothing
        """
        pass

    def CreateEquation(self, Name):
        """
        Creates a new daeEquation object
        ARGUMENTS:
         - Name: string
        RETURNS:
           daeEquation object
        """
        pass
    
    def ConnectPorts(self, PortFrom, PortTo):
        """
        ARGUMENTS:
         - PortFrom: daePort object
         - PortTo: daePort object
        RETURNS:
           Nothing
        """
        pass

    def SetReportingOn(self, On):
        """
        ARGUMENTS:
         - On: boolean
        RETURNS:
           Nothing
        """
        pass

    def dt(self, ad):
        """
        ARGUMENTS:
         - ad: adouble object
        RETURNS:
           adouble object
        """
        pass
    
    def sum(self, adarr):
        """
        ARGUMENTS:
         - adarr: adouble_array object
        RETURNS:
           adouble object
        """
        pass

    def product(self, adarr):
        """
        ARGUMENTS:
         - adarr: adouble_array object
        RETURNS:
           adouble object
        """
        pass

    def integral(self, adarr):
        """
        ARGUMENTS:
         - adarr: adouble_array object
         - Domain: daeDomain object
         - From: unsigned int
         - To: unsigned int
        RETURNS:
           adouble object
        """
        pass

    def min(self, adarr):
        """
        ARGUMENTS:
         - adarr: adouble_array object
        RETURNS:
           adouble object
        """
        pass
  
    def max(self, adarr):
        """
        ARGUMENTS:
         - adarr: adouble_array object
        RETURNS:
           adouble object
        """
        pass

    def average(self, adarr):
        """
        ARGUMENTS:
         - adarr: adouble_array object
        RETURNS:
           adouble object
        """
        pass

    def dt(self, ad):
        """
        Calculates a time derivative of an expression
        ARGUMENTS:
         - ad: adouble object
        RETURNS:
           adouble object
        """
        pass

    def d(self, ad, domain):
        """
        Calculates a partial derivative of an expression per domain 
        ARGUMENTS:
         - ad: adouble object
         - domain: daeDomain object
        RETURNS:
           adouble object
        """
        pass

    def IF(self, Condition, EventTolerance = 0):
        """
        ARGUMENTS:
         - Condition: daeCondition object
         - EventTolerance: real_t
        RETURNS:
           Nothing
        """
        pass

    def ELSE_IF(self, Condition, EventTolerance = 0):
        """
        ARGUMENTS:
         - Condition: daeCondition object
         - EventTolerance: real_t
        RETURNS:
           Nothing
        """
        pass

    def ELSE(self):
        """
        ARGUMENTS:
        RETURNS:
           Nothing
        """
        pass

    def END_IF(self):
        """
        ARGUMENTS:
        RETURNS:
           Nothing
        """
        pass

    def STN(self, Name):
        """
        ARGUMENTS:
         - Name: string
        RETURNS:
           daeSTN object
        """
        pass

    def STATE(self, Name):
        """
        ARGUMENTS:
         - Name: string
        RETURNS:
           daeState object
        """
        pass

    def END_STN(self):
        """
        ARGUMENTS:
        RETURNS:
           Nothing
        """
        pass

    def SWITCH_TO(self, StateName, Condition, EventTolerance = 0):
        """
        ARGUMENTS:
         - StateName: string
         - Condition: daeCondition object
         - EventTolerance: real_t
        RETURNS:
           Nothing
        """
        pass

class adouble:
    """
    ATTRIBUTES:
     - Value: float
     - Derivative: float
    """
    def __neg__(self):
        """
        Prefix operator -
        ARGUMENTS:
           None
        RETURNS:
           adouble
        """
        pass
    
    def __pos__(self):
        """
        Prefix operator +
        ARGUMENTS:
           None
        RETURNS:
           adouble
        """
        pass

    def __add__(self, adf):
        """
        Mathematical operator +
        ARGUMENTS:
         - adf: adouble | float
        RETURNS:
           adouble
        """
        pass
    
    def __sub__(self, adf):
        """
        Mathematical operator -
        ARGUMENTS:
         - adf: adouble | float
        RETURNS:
           adouble
        """
        pass
    
    def __mul__(self, adf):
        """
        Mathematical operator *
        ARGUMENTS:
         - adf: adouble | float
        RETURNS:
           adouble
        """
        pass
    
    def __div__(self, adf):
        """
        Mathematical operator /
        ARGUMENTS:
         - adf: adouble | float
        RETURNS:
           adouble
        """
        pass
    
    def __pow__(self, adf):
        """
        Mathematical operator ^
        ARGUMENTS:
         - adf: adouble | float
        RETURNS:
           adouble
        """
        pass
    
    def __eq__(self, adf):
        """
        Mathematical operator ==
        ARGUMENTS:
         - adf: adouble | float
        RETURNS:
           adouble
        """
        pass
    
    def __neq__(self, adf):
        """
        Mathematical operator !=
        ARGUMENTS:
         - adf: adouble | float
        RETURNS:
           adouble
        """
        pass
    
    def __le__(self, adf):
        """
        Mathematical operator <=
        ARGUMENTS:
         - adf: adouble | float
        RETURNS:
           adouble
        """
        pass
    
    def __gt__(self, adf):
        """
        Mathematical operator >
        ARGUMENTS:
         - adf: adouble | float
        RETURNS:
           adouble
        """
        pass
    
    def __ge__(self, adf):
        """
        Mathematical operator >=
        ARGUMENTS:
         - adf: adouble | float
        RETURNS:
           adouble
        """
        pass

def Min(adf1, adf2):
    """
    Mathematical function min
    ARGUMENTS:
     - adf1: adouble | float
     - adf2: adouble | float
    RETURNS:
       adouble
    """
    pass

def Max(adf1, adf2):
    """
    Mathematical function max
    ARGUMENTS:
     - adf1: adouble | float
     - adf2: adouble | float
    RETURNS:
       adouble
    """
    pass

def Pow(adf1, adf2):
    """
    Mathematical function pow
    ARGUMENTS:
     - adf1: adouble | float
     - adf2: adouble | float
    RETURNS:
       adouble
    """
    pass


class adouble_array:
    """
    """
    def __getitem__(self, Index):
        """
        Operator []
        ARGUMENTS:
         - Index: unsigned int
        RETURNS:
           adouble
        """
        pass

    def __neg__(self):
        """
        Prefix operator -
        ARGUMENTS:
           None
        RETURNS:
           adouble
        """
        pass

    def __add__(self, adarr):
        """
        Mathematical operator +
        ARGUMENTS:
         - adarr: adouble | adouble_array
        RETURNS:
           adouble
        """
        pass
    
    def __sub__(self, adarr):
        """
        Mathematical operator -
        ARGUMENTS:
         - adarr: adouble | adouble_array
        RETURNS:
           adouble
        """
        pass
    
    def __mul__(self, adarr):
        """
        Mathematical operator *
        ARGUMENTS:
         - adarr: adouble | adouble_array
        RETURNS:
           adouble
        """
        pass
    
    def __div__(self, adarr):
        """
        Mathematical operator /
        ARGUMENTS:
         - adarr: adouble | adouble_array
        RETURNS:
           adouble
        """
        pass
    
def Exp(adarr):
    """
    Calculates exponential of the argument adarr, which can be a constant adouble or adouble_array object
    ARGUMENTS:
     - adarr: adouble | adouble_array
    RETURNS: 
       adouble | adouble_array
    """
    pass

def Log(adarr):
    """
    ARGUMENTS:
     - adarr: adouble | adouble_array
    RETURNS: 
       adouble | adouble_array
    """
    pass

def Sqrt(adarr):
    """
    ARGUMENTS:
     - adarr: adouble | adouble_array
    RETURNS: 
       adouble | adouble_array
    """
    pass

def Sin(adarr):
    """
    ARGUMENTS:
     - adarr: adouble | adouble_array
    RETURNS: 
       adouble | adouble_array
    """
    pass

def Cos(adarr):
    """
    ARGUMENTS:
     - adarr: adouble | adouble_array
    RETURNS: 
       adouble | adouble_array
    """
    pass

def Tan(adarr):
    """
    ARGUMENTS:
     - adarr: adouble | adouble_array
    RETURNS: 
       adouble | adouble_array
    """
    pass

def ASin(adarr):
    """
    ARGUMENTS:
     - adarr: adouble | adouble_array
    RETURNS: 
       adouble | adouble_array
    """
    pass

def ACos(adarr):
    """
    ARGUMENTS:
     - adarr: adouble | adouble_array
    RETURNS: 
       adouble | adouble_array
    """
    pass

def ATan(adarr):
    """
    ARGUMENTS:
     - adarr: adouble | adouble_array
    RETURNS: 
       adouble | adouble_array
    """
    pass

def Log10(adarr):
    """
    ARGUMENTS:
     - adarr: adouble | adouble_array
    RETURNS: 
       adouble | adouble_array
    """
    pass

def Abs(adarr):
    """
    ARGUMENTS:
     - adarr: adouble | adouble_array
    RETURNS: 
       adouble | adouble_array
    """
    pass

def Ceil(adarr):
    """
    ARGUMENTS:
     - adarr: adouble | adouble_array
    RETURNS: 
       adouble | adouble_array
    """
    pass

def Floor(adarr):
    """
    ARGUMENTS:
     - adarr: adouble | adouble_array
    RETURNS: 
       adouble | adouble_array
    """
    pass

class daeCondition:
    """
    """
    def __init__(self, Name):
        """
        ARGUMENTS:
         - Name: string 
        """
        pass
    
    def __not__(self):
        """
        Logical operator not
        ARGUMENTS:
           None 
        RETURNS: 
           daeCondition
        """
        pass
    
    def __and__(self, Condition):
        """
        Logical operator & (AND)
        ARGUMENTS:
         - Condition: daeCondition 
        RETURNS: 
           daeCondition
        """
        pass
    
    def __or__(self, Condition):
        """
        Logical operator | (OR) 
        ARGUMENTS:
         - Condition: daeCondition 
        RETURNS: 
           daeCondition
        """
        pass

