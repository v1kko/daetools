#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            whats_the_time.py
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
What is the time? (AKA Hello world!) is a very simple model. The model consists of
a single variable (called 'time') and a single differential equation::

  d(time)/dt = 1

This way, the value of the variable 'time' is equal to the elapsed time in the simulation.

This tutorial presents the basic structure of daeModel and daeSimulation classes.
A typical DAETools simulation requires the following 8 tasks:

1. Importing DAE Tools pyDAE module(s)

2. Importing or declaration of units and variable types
   (unit and daeVariableType classes)
   
3. Developing a model by deriving a class from the base daeModel class and:

   - Declaring domains, parameters and variables in the daeModel.__init__ function
   - Declaring equations and their residual expressions in the
     daeModel.DeclareEquations function
     
4. Setting up a simulation by deriving a class from the base daeSimulation class and:

   - Specifying a model to be used in the simulation in the daeSimulation.__init__ function
   - Setting the values of parameters in the daeSimulation.SetUpParametersAndDomains function
   - Setting initial conditions in the daeSimulation.SetUpVariables function
   
5. Declaring auxiliary objects required for the simulation

   - DAE solver object
   - Data reporter object
   - Log object
   
6. Setting the run-time options for the simulation:

   - ReportingInterval
   - TimeHorizon

7. Connecting a data reporter

8. Initializing, running and finalizing the simulation

The 'time' variable plot:

.. image:: _static/whats_the_time-results.png
   :width: 500px
"""

# 1. Import the modules
import sys
from time import localtime, strftime
from daetools.pyDAE import *

# 2a. Import some unit definitions (if needed) from the pyUnits module
from pyUnits import m, kg, s, K, Pa, mol, J, W

"""
2b. Import or declare variable types
    Variable types are typically declared outside of model classes since
    they define common, reusable types.
    The daeVariable constructor takes 6 arguments:
     - Name: string
     - Units: unit object
     - LowerBound: float (not enforced at the moment)
     - UpperBound: float (not enforced at the moment)
     - InitialGuess: float
     - AbsoluteTolerance: float
     - ValueConstraint: daeeVariableValueConstraint (default is eNoConstraint - no constraint set)
    
    Standard variable types are defined in variable_types.py and can be imported with:

    >> from variable_types import length_t, volume_t, area_t ...
    
    Here, just as an example, a simple variable type with the time units is declared: time_type.
    Normally the time_t variable type should be used for time.
"""
time_type = daeVariableType("time_type", s, 0, 1E10,   0, 1e-5)

"""
3. Declare a model by deriving a class from the base daeModel class.
"""
class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        """
        3.1 Declare domains/parameters/variables/ports etc.
            Domains, parameters, variables, ports, etc has to be defined in the constructor: __init__
            First, the base class constructor has to be called.
            Then, all domains, parameters, variables, ports etc have to be declared as members of
            the new model class (except equations which are handled by the framework), since the base class
            keeps only their references. Therefore we write:
                     self.variable = daeVariable(...)
            and not:
                     variable = daeVariable(...)
            In this example we declare only one variable: tau.
            daeVariable constructor accepts 4 arguments:
             - Name: string
             - VariableType: daeVariableType
             - Parent: daeModel object (indicating the model where the variable will be added)
             - Description: string (optional argument - the variable description; the default value is an empty string)
            Variable names can consists of letters, numbers, underscores '_', brackets '(', ')',  commas ',' and
            standard html code names, such as Greek characters: &alpha; &beta; &gamma; etc. Spaces are not allowed.
            Examples of valid object names: Foo, bar_1, foo(1,2), &Alpha;_k1 
            Internally they will be used without '&' and ';' characters: alpha, beta, gamma, ...;
            but, in the equations exported to the MathML or Latex format they will be shown as native Greek letters.
            Also if you write the variable name as: Name_1 it will be transformed into Name with the subscript 1.
            In this example we use Greek character 'τ' to name the variable 'tau'.
        """
        daeModel.__init__(self, Name, Parent, Description)
        
        self.tau = daeVariable("&tau;", time_type, self, "Time elapsed in the process")

    def DeclareEquations(self):
        """
        3.2 Declare equations and state transition networks
            All models must implement DeclareEquations function and all equations must be specified here.

            Tipically the user-defined DeclareEquations function should first call the DeclareEquations
            function from the parent class (here daeModel). This is important when a model contains instances
            of other models to allow DeclareEquations calls from those (child-)models. If a model is simple
            (like in this example) there is no need for daeModel.DeclareEquations(self) call since it does nothing.
            However, it is a good idea to call it always (to avoid mistakes).
        """
        daeModel.DeclareEquations(self)

        """
        In this example we declare only one equation.
        CreateEquation function accepts two arguments: equation name and description (optional).
        All naming conventions apply here as well. Equations are written in an implicit form (as a residual):
                        'residual expression' = 0
        Residual expressions are defined by using the basic mathematical operations (+, -, *, /)
        and functions (sqrt, log, sin, cos, max, min, ...) on variables and parameters. Variables define several
        useful functions which return modified ADOL-C 'adouble' objects needed for construction of equation
        evaluation trees. adouble objects are used only for building the equation evaluation trees during the
        simulation initialization phase and cannot be used otherwise. They hold a variable value, a derivative
        (required for construction of a Jacobian matrix) and the tree evaluation information. The most
        otenly used functions are:
            - operator () which returns adouble object that holds the variable value
            - function dt() which returns adouble object that holds a time derivative of the variable
            - function d() and d2() which return adouble object that holds a partial derivative of the variable
            of the 1st and the 2nd order, respectively
        In this example we simply write that the variable time derivative is equal to 1:
                                d(tau) / dt = 1
        with the effect that the value of the 'tau' variable is equal to the time elapsed in the simulation
        (normally, the built-in function Time() should be used to get the current time in the simulation;
        however, this is just an example explaining the basic daetools concepts).
        Note that the variable objects should be declared as data members of the models they belong to and
        therefore accessed through the model objects.

        As of the version 1.2.0 all daetools objects have numerical values in terms of a unit of measurement (quantity)
        and unit-consistency is strictly enforced (although it can be turned off in daetools.cfg config file,
        or for particular equations by setting the equation's boolean 'CheckUnitsConsistency' property to false.
        The config files are searched for in the following directories:
          - %{HOME}/.daetools directory
          - .../site-packages/daetools directory
          - the directory where the executable is located
        All values and constants must be declared with the information about their units (unless dimensionless).
        Units of variables, parameters and domains are specified in their constructor while constants
        and arrays of constants are instantiated with the built-in Constant() and Array() functions.
        Obviously, the very strict unit-consistency requires an extra effort during the model development phase
        and makes models more verbose. However, it helps to eliminate some very hard to find errors.
        'quantity' objects have two properties: the 'value' and the 'units'.
        The pyDAE.pyUnits module offers the following predefined set of units:
            - All base SI units: m, kg, s, A, K, cd, mol
            - Some of the most commonly used derived SI units for time, volume, energy, electromagnetism etc
            (see units_pool.h file in trunk/Units folder)
            - Base SI units with the multiplies: deca-, hecto-, kilo-, mega-, giga-, tera-, peta-, exa-, zetta- and yotta-
            using the symbols: da, h, k, M, G, T, P, E, Z, Y
            - Base SI units with the fractions: deci-, centi-, milli-, micro-, nano-, pico-, femto-, atto-, zepto- and yocto-
            using the symbols: d, c, m, u, n, p, f, a, z, y

        ACHTUNG, ACHTUNG!!
        Never import all symbols from the pyUnits module (it will polute the namespace with thousands of unit symbols)!!

        Custom derived units can be constructed using the mathematical operations *, / and ** on unit objects.

        In the equation below we can write:
           self.tau.dt() - 1.0
        since the dtau/dt term is dimensionless. However, in a general case, we must use Constant(1.0 * units).
        """
        eq = self.CreateEquation("Time", "Differential equation to calculate the time elapsed in the process.")
        eq.Residual = dt(self.tau()) - 1.0

# 4. Declare a simulation by deriving a class from the base daeSimulation class
class simTutorial(daeSimulation):
    def __init__(self):
        """
        4.1 First, the base class constructor has to be called, and then the model for simulation instantiated.
            daeSimulation class has three properties used to store the model: 'Model', 'model' and 'm'.
            They are absolutely equivalent, and user can choose which one to use.
            For clarity, here the shortest one will be used: m.
        """
        daeSimulation.__init__(self)

        self.m = modTutorial("whats_the_time")
        self.m.Description = __doc__
        
    def SetUpParametersAndDomains(self):
        """
        4.2 Initialize domains and parameters
            Every simulation class must implement SetUpParametersAndDomains method, even if it is empty.
            It is used to set the values of the parameters, initialize domains etc.
            In this example nothing has to be done.
        """
        pass

    def SetUpVariables(self):
        """
        4.3 Set initial conditions, initial guesses, fix degreees of freedom, etc.
            Every simulation class must implement SetUpVariables method, even if it is empty.
            In this example the only thing needed to be done is to set the initial condition for the variable tau.
            That can be done using the SetInitialCondition function.
        """
        self.m.tau.SetInitialCondition(0.0)

# Use daeSimulator class (pyQt GUI)
def guiRun(qtApp):
    simulation = simTutorial()
    simulation.m.SetReportingOn(True)
    simulation.ReportTimeDerivatives = True
    simulation.ReportingInterval = 5
    simulation.TimeHorizon       = 100
    simulator  = daeSimulator(qtApp, simulation = simulation)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    """
    5. Create Log, Solver, DataReporter and Simulation object
       Every simulation requires the following three additional objects:
        - log is used to send the messages from the framework, set the simulation progress or to report errors
        - solver is DAE solver used to solve the underlying system of differential and algebraic equations
        - datareporter is used to send the data from the solver to daePlotter (or any other data receiver).
          Often, it is required to save the results in a specified file format; for more information see Tutorial 8.

       Various options used by daetools objects are located in the daetools/daetools.cfg config file (in JSON format).
       The config file can be obtained using the global function daeGetConfig:
           cfg  = daeGetConfig()
       The config file is first searched in the HOME directory, the application folder and finally in the default location.
       It also can be specified manually using the function daeSetConfigFile('path_to_daetools.cfg').
       However, this has to be done before the daetools objects are created.
       The options can also be programmatically changed using the Get/Set functions i.e. GetBoolean/SetBoolean:
           checkUnitsConsistency = cfg.GetBoolean("daetools.core.checkUnitsConsistency")
           cfg.SetBoolean("daetools.core.checkUnitsConsistency", True)
       Supported data types are: Bool, Integer, Float and String.
       The whole config file with all options can be printed using:
           print(cfg)
    """
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()

    """
    6. Additional settings
       - TimeHorizon: the duration of the simulation, in seconds (required)
       - ReportingInterval: the time interval after which the results are reported, in seconds (required)
       - Selection of the variables whose values will be reported.
         It can be set individually for each variable by using the property variable.ReportingOn = True/False,
         or by using the function daeModel.SetReportingOn() which enables/disables all variables in the model.
         The default is that no variable is reported.
         Here, we enable reporting of all variables in the model.
       - The time derivatives of all variables will be reported
    """
    simulation.TimeHorizon       = 100
    simulation.ReportingInterval = 5
    
    simulation.m.SetReportingOn(True)
    
    simulation.ReportTimeDerivatives = True
    # The same could be achieved using the following (if set before initialisation of the simulation):
    #cfg = daeGetConfig()
    #cfg.SetBoolean("daetools.activity.reportTimeDerivatives", True)

    """
    7. Connect the data reporter
       daeTCPIPDataReporter data reporter uses TCP/IP protocol to send the results to the data receiver (daePlotter).
       It contains the function Connect which accepts two arguments:
         - TCP/IP address and port as a string in the following form: '127.0.0.1:50000'.
           The default is an empty string which allows the data reporter to connect to the local
           (on this machine) daePlotter listening on the port 50000.
         - Process name; in this example we use the combination of the simulation name and the current date and time
    """
    simName = "whats_the_time" + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    """
    8. Run the simulation
     8.1 The simulation initialization
         The first task is to initialise the simulation by calling the function Initialize.
         As the fourth argument, it accepts an optional CalculateSensitivities (bool; default is False) which can
         enable calculation of sensitivities for given opt. variables (more details in the optimisation tutorials).
         After the successful initialisation the model reports can be generated.
         The function SaveModelReport exports the model report in the XML format.
         The function SaveRuntimeModelReport creates a runtime version of the model report
         (with all run-time information and all equations fully expanded)
    """
    simulation.Initialize(daesolver, datareporter, log)

    # Save the model report and the runtime model report
    simulation.m.SaveModelReport("whats_the_time.xml")
    simulation.m.SaveRuntimeModelReport("whats_the_time-rt.xml")

    """
    8.2 Solve at time = 0 (initialisation)
        The DAE system must be solved at the t = 0 where the specified initial conditions
        are used to calculate all other model variables.
        The function SolveInitial is used for that purpose.
    """
    simulation.SolveInitial()

    """
    8.3 Start the simulation using function daeSimulation.Run().
        The simulation will be run until the time reaches the specified TimeHorizon (in seconds).
        During the simulation the results will be reported after every ReportingInterval (in seconds).
    """
    simulation.Run()

    # 8.4 Finally, call the function Finalize to clean-up.
    simulation.Finalize()

# Simulations can also be executed in a simple way using the static daeActivity.simulate function.
# The function uses the default objects for all missing data (as in the consoleRun function above):
#   log          = daePythonStdOutLog()
#   daesolver    = daeIDAS()
#   datareporter = daeTCPIPDataReporter()
# Several additional arguments can be specified such as: 
#   time horizon, reporting interval, datareporter, log, daesolver, lasolver, relative tolerance,
#   callback functions to be called after initialisation, before and after simulation, etc.
# If the keyword arguments **kwargs contain 'guiRun=True' the simulation will be run using the Qt GUI.
def run(**kwargs):
    simulation = simTutorial()
    return daeActivity.simulate(simulation, reportingInterval     = 5, 
                                            timeHorizon           = 100,
                                            reportTimeDerivatives = True,
                                            **kwargs)

"""
This part of the code executes if the python script is executed from a shell
1) If called as: "python whats_the_time.py console" the simulation will be launched in the console
2) If called as: "python whats_the_time.py gui" the simulation will be launched with a GUI
   The default is "gui" and can be omitted.
"""
if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        consoleRun()
    else:
        qtApp = daeCreateQtApplication(sys.argv)
        guiRun(qtApp)
