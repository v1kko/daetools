#!/usr/bin/env python

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

# 2. Define variable types
#    Variable types are typically declared outside of model classes since they define common, reusable types.
#    The daeVariable constructor takes 6 arguments:
#     - Name: string
#     - Units: string
#     - LowerBound: float
#     - UpperBound: float
#     - InitialGuess: float
#     - AbsoluteTolerance: float 
#    Here a very simple variable type is declared 
typeNone = daeVariableType("None", "-", 0, 1E10,   0, 1e-5)

# 3. Define a model
#    New models are derived from the base daeModel class.
class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        # 3.1 Declare domains/parameters/variables/ports
        #     Domains, parameters, variables, ports, etc has to be defined in the constructor: __init__
        #     First, the base class constructor has to be called.
        #     Then, all domains, parameters, variables, ports etc has to be declared as members of 
        #     the new model class (except equations which are handled by the framework), since the base class 
        #     keeps only their references. Therefore we write:
        #                    self.variable = daeVariable(...) 
        #     and not:
        #                    variable = daeVariable(...)
        #
        #     In this example we declare only one variable: time.
        #     daeVariable constructor accepts 3 arguments:
        #      - Name: string
        #      - VariableType: daeVariableType
        #      - Parent: daeModel object (indicating the model where the variable will be added)
        #      - Description: string (optional argument - the variable description; the default value is an empty string)
        #     Variable names can consists of letters, numbers, underscore ('_') and no other special characters (' ', '.', ',', '/', '-', etc).
        #     However, there is an exception. You can use the standards html code names, such as Greek characters: &alpha; &beta; &gamma; etc.
        #     Internally they will be used without '&' and ';' characters: alpha, beta, gamma, ...;
        #     but, in the equations exported in the MathML or Latex format they will be shown as native Greek letters.
        #     Also if you write the variable name as: Name_1 it will be transformed into Name with the subscript 1.
        #     In this example we use Greek character 'tau' to name the variable 'time'.
        daeModel.__init__(self, Name, Parent, Description)
        self.time = daeVariable("&tau;", typeNone, self, "Time elapsed in the process, s")

    def DeclareEquations(self):
        # 3.2 Declare equations and state transition networks
        #     Here we declare only one equation. Equations are created by the function CreateEquation in daeModel class.
        #     It accepts two arguments: equation name and description (optional). All naming conventions apply here as well.
        #     Equations are written in the form of a residual, which is accepted by DAE equation system solvers:
        #                           'residual expression' = 0
        #     Residuals are defined by creating the above expression by using the basic mathematical operations (+, -, * and /) 
        #     and functions (sqrt, log, sin, cos, max, min, ...) on variables and parameters. Variables define several useful functions:
        #      - operator () which calculates the variable value
        #      - function dt() which calculates a time derivative of the variable
        #      - function d() and d2() which calculate a partial derivative of the variable of the 1st and the 2nd order, respectively
        #     In this example we simply write that the variable time derivative is equal to 1:
        #                          d(time) / dt = 1
        #     which calculates the time as the current time elapsed in the simulation (which is a common way to obtain the current time within the model).
        #     Note that the variables should be accessed through the model object, therefore we use self.time
        eq = self.CreateEquation("Time", "Differential equation to calculate the time elapsed in the process.")
        eq.Residual = self.time.dt() - 1.0
 
# 4. Define a simulation
#    Simulations are derived from the base daeSimulation class
class simTutorial(daeSimulation):
    def __init__(self):
        # 4.1 First, the base class constructor has to be called, and then the model for the simulation has to be instantiated.
        #     daeSimulation class has three properties used to store the model: 'Model', 'model' and 'm'.
        #     They are absolutely equivalent, and user can choose which one to use. For clarity, I prefer the shortest one: m.
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
        # 4.2 Define the domains and parameters
        #     Every simulation class must implement SetUpParametersAndDomains method, even if it is empty.
        #     It is used to set the values of the parameters, create domains etc. In this example nothing has to be done.
        pass
    
    def SetUpVariables(self):
        # 4.3 Set initial conditions, initial guesses, fix degreees of freedom, etc.
        #     Every simulation class must implement SetUpVariables method, even if it is empty.
        #     In this example the only thing needed to be done is to set the initial condition for the variable time to 0.
        #     That can be done by using SetInitialCondition function:
        self.m.time.SetInitialCondition(0)

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
    # 5.Create Log, Solver, DataReporter and Simulation object
    #   Every simulation requires the following four objects:
    #    - log is used to send the messages from other parts of the framework, informs us about the simulation progress or errors
    #    - solver is DAE solver used to solve the underlying system of differential and algebraic equations
    #    - datareporter is used to send the data from the solver to daePlotter
    #    - simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()

    # 6. Additional settings
    # Here some additional information can be entered. The most common are:
    #  - TimeHorizon: the duration of the simulation
    #  - ReportingInterval: the interval to send the variable values
    #  - Selection of the variables which values will be reported. It can be set individually for each variable by using the property var.ReportingOn = True/False,
    #    or by the function SetReportingOn in daeModel class which enables reporting of all variables in that model
    simulation.TimeHorizon = 500
    simulation.ReportingInterval = 10
    simulation.m.SetReportingOn(True)

    # 7. Connect the data reporter
    #    daeTCPIPDataReporter data reporter uses TCP/IP protocol to send the results to the daePlotter.
    #    It contains the function Connect which accepts two arguments:
    #      - TCP/IP address and port as a string in the following form: '127.0.0.1:50000'. 
    #        The default is an empty string which allows the data reporter to connect to the local (on this machine) daePlotter listening on the port 50000.
    #      - Process name; in this example we use the combination of the simulation name and the current date and time
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # 8. Run the simulation
    #  8.1 The simulation initialization
    #      The first task is to initialize the simulation by calling the function Initialize.
    #      After the successful initialization the model report can be saved.
    #      The function SaveModelReport exports the model report in the XML format which can be opened in a web browser 
    #      (like Mozilla Firefox, or others that support XHTML+MathMl standard).
    #      The function SaveRuntimeModelReport creates a runtime sort of the model report (with the equations fully expanded)
    simulation.Initialize(daesolver, datareporter, log)

    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    #  8.2 Solve at time = 0 (initialization)
    #      The DAE system must be first initialized. The function SolveInitial is used for that purpose.
    simulation.SolveInitial()

    # 8.3 Call the function Run from the daeSimulation class to start the simulation.
    #     It will last for TimeHorizon seconds and the results will be reported after every ReportingInterval number of seconds 
    simulation.Run()

    # 8.4 Finally, call the function Finalize to clean-up.
    simulation.Finalize()

# This part of the code executes if the python script is executed from a shell
# 1) If you use: "python whats_the_time.py console" the simulation will be launched from the console
# 2) If you use: "python whats_the_time.py gui" the simulation will be launched from a GUI
#    The default is "gui" and you can omit it.
if __name__ == "__main__":
    runInGUI = True
    if len(sys.argv) > 1:
        if(sys.argv[1] == 'console'):
            runInGUI = False
    if runInGUI:
        from PyQt4 import QtCore, QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
    else:
        consoleRun()
