#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                             whatsTheTime.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
********************************************************************************"""

"""
What is the time? (AKA Hello world!) is a very simple simulation.
It shows the basic structure of the model and the simulation classes.
A typical simulation imcludes 8 basic tasks:
    1. How to import the pyDAE module
    2. How to declare variable types
    3. How to define a model by deriving a class from the base daeModel, what are the methods
       that must be implemented by every model, and how to define an equation and its residual
    4. How to define a simulation by deriving a class from the base daeSimulation, what are the methods
       that must be implemented by every simulation, and how to set initial conditions
    5. How to create auxiliary objects required for the simulation (DAE solver, data reporter and logging objects)
    6. How to set simulation's additional settings
    7. How to connect a data reporter
    8. How to run a simulation
"""

# 1. Import the modules
import sys
from daetools.pyDAE import *
from time import localtime, strftime

# 2a. Import some unit definitions (if needed) from the pyUnits module
from pyUnits import m, kg, s, K, Pa, mol, J, W

"""
2b. Define variable types
   Variable types are typically declared outside of model classes since they define common, reusable types.
   The daeVariable constructor takes 6 arguments:
    - Name: string
    - Units: unit object
    - LowerBound: float (not enforced at the moment)
    - UpperBound: float (not enforced at the moment)
    - InitialGuess: float
    - AbsoluteTolerance: float
   Standard variable types are defined in daeVariableTypes.py
   Here, a very simple dimensionless variable type is declared:
"""
typeNone = daeVariableType("typeNone", unit(), 0, 1E10,   0, 1e-5)

"""
3. Define a model
   New models are derived from the base daeModel class.
"""
class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        """
        3.1 Declare domains/parameters/variables/ports
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
        self.tau = daeVariable("&tau;", typeNone, self, "Time elapsed in the process")

    def DeclareEquations(self):
        """
        3.2 Declare equations and state transition networks
            All models must implement DeclareEquations function and all equations must be specified here.
            Equations are created by the function CreateEquation in daeModel class. In this example we declare only
            one equation. CreateEquation accepts two arguments: equation name and description (optional). All naming
            conventions apply here as well. Equations are written in the form of a residual, which is accepted by DAE
            equation system solvers:
                                  'residual expression' = 0
            Residuals are defined by creating the above expression using the basic mathematical operations (+, -, *, /)
            and functions (sqrt, log, sin, cos, max, min, ...) on variables and parameters. Variables define several
            useful functions which return modified ADOL-C 'adouble' objects needed for construction of equation
            evaluation trees. adouble objects are used only for building the equation evaluation trees during the
            simulation initialization phase and cannot be used otherwise. They hold a variable value, a derivative
            (required for construction of a Jacobian matrix) and the tree evaluation information.
             - operator () which returns adouble object that calculates the variable value
             - function dt() which returns adouble object thatcalculates a time derivative of the variable
             - function d() and d2() which return adouble object that calculates a partial derivative of the variable
               of the 1st and the 2nd order, respectively
            In this example we simply write that the variable time derivative is equal to 1:
                                 d(tau) / dt = 1
            which calculates the time as the current time elapsed in the simulation (normally, the built-in function Time()
            should be used to get the current time in the simulation; this is just an example explaining the basic daetools
            concepts). Note that the variable objects should be declared as members of the models they belong to and
            therefore accessed through the model objects.

            As of the version 1.2.0 all daetools objects use quantities with physical dimensions and unit-consistency is
            strictly enforced (although it can be turned off in daetools.cfg config file, typically located in /etc/daetools
            folder). All values and constants must be declared with the information about their units. Units of variables,
            parameters and domains are specified in their constructor while constants and arrays of constants are instantiated
            with the built-in Constant() and Array() functions. Obviously, the very strict unit-consistency requires an extra
            effort during the model building phase and makes models more verbose. However, it helps eliminate some very hard
            to find errors and might save some NASA orbiters :-)
            'quantity' objects consist of the value and the units. The pyDAE.pyUnits module declares the following units:
              - All base SI units: m, kg, s, A, K, cd, mol
              - Some of the most commonly used derived SI units for time, volume, energy, electromagnetism etc
                (see units_pool.h file in trunk/Units folder)
              - Base SI units with the multiplies: deca-, hecto-, kilo-, mega-, giga-, tera-, peta-, exa-, zetta- and yotta-
                using the symbols: da, h, k, M, G, T, P, E, Z, Y)
              - Base SI units with the fractions: deci-, centi-, milli-, micro-, nano-, pico-, femto-, atto-, zepto- and yocto-
                using the symbols: d, c, m, u, n, p, f, a, z, y

            ACHTUNG, ACHTUNG!!
            Never import all symbols from the pyUnits module (it will polute the namespace with thousands of unit symbols)!!

            Custom derived units can be constructed using the mathematical operations *, / and ** on unit objects.
            In this example we declare a quantity with the value of 1.0 and units s^-1:
        """
        eq = self.CreateEquation("Time", "Differential equation to calculate the time elapsed in the process.")
        eq.Residual = self.tau.dt() - Constant(1.0 * 1/s)

# 4. Define a simulation
#    Simulations are derived from the base daeSimulation class
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
        self.m.Description = "What is the time? (AKA Hello world) is the simplest simulation. It shows the basic structure of the model and the simulation classes. " \
                             "The basic 8 steps are explained: \n" \
                             "1. How to import pyDAE module \n" \
                             "2. How to declare variable types \n" \
                             "3. How to define a model by deriving a class from the base daeModel, etc \n" \
                             "4. How to define a simulation by deriving a class from the base daeSimulation, etc \n" \
                             "5. How to create auxiliary objects required for the simulation (DAE solver, data reporter and logging objects) \n" \
                             "6. How to set simulation's additional settings \n" \
                             "7. How to connect the TCP/IP data reporter \n" \
                             "8. How to run a simulation \n"

    def SetUpParametersAndDomains(self):
        """
        4.2 Define the domains and parameters
            Every simulation class must implement SetUpParametersAndDomains method, even if it is empty.
            It is used to set the values of the parameters, initialize domains etc.
            In this example nothing has to be done.
        """
        pass

    def SetUpVariables(self):
        """
        4.3 Set initial conditions, initial guesses, fix degreees of freedom, etc.
            Every simulation class must implement SetUpVariables method, even if it is empty.
            In this example the only thing needed to be done is to set the initial condition for the variable tau to 0.
            That can be done by using SetInitialCondition function:
        """
        self.m.tau.SetInitialCondition(0)

# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 10
    sim.TimeHorizon       = 1000
    simulator  = daeSimulator(app, simulation=sim)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    """
    5.Create Log, Solver, DataReporter and Simulation object
      Every simulation requires the following four objects:
       - log is used to send the messages from other parts of the framework, informs us about the simulation progress or errors
       - solver is DAE solver used to solve the underlying system of differential and algebraic equations
       - datareporter is used to send the data from the solver to daePlotter
       - simulation object
    """
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()

    """
    6. Additional settings
    Here some additional information can be entered. The most common are:
     - TimeHorizon: the duration of the simulation
     - ReportingInterval: the interval to send the variable values
     - Selection of the variables which values will be reported. It can be set individually for each variable by using the property var.ReportingOn = True/False,
       or by the function SetReportingOn in daeModel class which enables reporting of all variables in that model
    """
    simulation.TimeHorizon = 500
    simulation.ReportingInterval = 10
    simulation.m.SetReportingOn(True)

    """
    7. Connect the data reporter
       daeTCPIPDataReporter data reporter uses TCP/IP protocol to send the results to the daePlotter.
       It contains the function Connect which accepts two arguments:
         - TCP/IP address and port as a string in the following form: '127.0.0.1:50000'.
           The default is an empty string which allows the data reporter to connect to the local (on this machine) daePlotter listening on the port 50000.
         - Process name; in this example we use the combination of the simulation name and the current date and time
    """
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    """
    8. Run the simulation
     8.1 The simulation initialization
         The first task is to initialize the simulation by calling the function Initialize. As the 4th argument, it accepts  an optional
         CalculateSensitivities (bool; default is False) which can enable calculation of sensitivities for given opt. variables.
         After the successful initialization the model report can be saved.
         The function SaveModelReport exports the model report in the XML format which can be opened in a web browser
         (like Mozilla Firefox, or others that support XHTML+MathMl standard).
         The function SaveRuntimeModelReport creates a runtime sort of the model report (with the equations fully expanded)
    """
    simulation.Initialize(daesolver, datareporter, log)

    # Save the model report and the runtime model report
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    """
    8.2 Solve at time = 0 (initialization)
        The DAE system must be first initialized. The function SolveInitial is used for that purpose.
    """
    simulation.SolveInitial()

    """
    8.3 Call the function Run from the daeSimulation class to start the simulation.
        It will last for TimeHorizon seconds and the results will be reported after every ReportingInterval number of seconds
    """
    simulation.Run()

    # 8.4 Finally, call the function Finalize to clean-up.
    simulation.Finalize()

"""
This part of the code executes if the python script is executed from a shell
1) If you use: "python whats_the_time.py console" the simulation will be launched from the console
2) If you use: "python whats_the_time.py gui" the simulation will be launched from a GUI
   The default is "gui" and you can omit it.
"""
if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        consoleRun()
    else:
        from PyQt4 import QtCore, QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
