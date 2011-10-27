/********************************************************************************
                 DAE Tools: cDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
*********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
*********************************************************************************/

/*
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
*/

/* 1. Include either 'variable_types.h' or 'dae_develop.h' header (if you do not need pre-defined variable types)*/
#include "variable_types.h"
#include "../dae_develop.h"

/* 2. Define variable types
   Variable types are typically declared outside of model classes since they define common, reusable types.
   Pre-defined variable types are declared in the 'variable_types.h' header.
   The daeVariable constructor takes 6 arguments:
    - Name: string
    - Units: string
    - LowerBound: float
    - UpperBound: float
    - InitialGuess: float
    - AbsoluteTolerance: float
   Here, a very simple variable type is declared: 
*/
daeVariableType typeNone("typeNone", s, 0, 1E10,   0, 1e-5);

/* 3. Define a model
    New models are derived from the base daeModel class. */
class modWhatsTheTime : public daeModel
{
daeDeclareDynamicClass(modWhatsTheTime)
public:
    daeVariable tau;
	
/*
    3.1 Declare domains/parameters/variables/ports
        All domains, parameters, variables, ports etc has to be declared as members of
        the new model class (except equations which are handled by the framework), since the base class
        keeps only their references. 
        Domains, parameters, variables, ports, etc has to be defined in the constructor
        First, the base class constructor has to be called.
        In this example we declare only one variable: tau.
        daeVariable constructor accepts 3 arguments:
         - Name: string
         - VariableType: daeVariableType
         - Parent: daeModel object (indicating the model where the variable will be added)
         - Description: string (optional argument - the variable description; the default value is an empty string)
        Variable names can consists of letters, numbers, underscore ('_') and no other special characters (' ', '.', ',', '/', '-', etc).
        However, there is an exception. You can use the standards html code names, such as Greek characters: &alpha; &beta; &gamma; etc.
        Internally they will be used without '&' and ';' characters: alpha, beta, gamma, ...;
        but, in the equations exported in the MathML or Latex format they will be shown as native Greek letters.
        Also if you write the variable name as: Name_1 it will be transformed into Name with the subscript 1.
        In this example we use Greek character 'τ' to name the variable 'tau'.
*/
    modWhatsTheTime(string strName, daeModel* pParent = NULL, string strDescription = "") 
      : daeModel(strName, pParent, strDescription),
        tau("&tau;", typeNone, this, "Time elapsed in the process")
    {
    }

    void DeclareEquations(void)
    {
/* 
    3.2 Declare equations and state transition networks
        Here we declare only one equation. Equations are created by the function CreateEquation in daeModel class.
        It accepts two arguments: equation name and description (optional). All naming conventions apply here as well.
        Equations are written in the form of a residual, which is accepted by DAE equation system solvers:
                                'residual expression' = 0
        Residuals are defined by creating the above expression by using the basic mathematical operations (+, -, * and /)
        and functions (sqrt, log, sin, cos, max, min, ...) on variables and parameters. Variables define several useful functions:
            - operator () which calculates the variable value
            - function dt() which calculates a time derivative of the variable
            - function d() and d2() which calculate a partial derivative of the variable of the 1st and the 2nd order, respectively
        In this example we simply write that the variable time derivative is equal to 1:
                                d(tau) / dt = 1
        which calculates the time as the current time elapsed in the simulation (which is a common way to obtain the current time within the model).
        Note that the variables should be accessed through the model object, therefore we use self.tau
*/
        daeEquation* eq;

        eq = CreateEquation("Time", "Differential equation to calculate the time elapsed in the process.");
        eq->SetResidual(tau.dt() - 1);
    }
};

/*
4. Define a simulation
   Simulations are derived from the base daeSimulation class
*/
class simWhatsTheTime : public daeSimulation
{
public:
    modWhatsTheTime m;
    
public:
/*
    4.1 First, the base class constructor has to be called, and then the model for the simulation has to be instantiated.
        daeSimulation class has three properties used to store the model: 'Model', 'model' and 'm'.
        They are absolutely equivalent, and user can choose which one to use. For clarity, I prefer the shortest one: m.
*/
    simWhatsTheTime(void) : m("cdaeWhatsTheTime")
    {
        SetModel(&m);
        m.SetDescription("What is the time? (AKA Hello world) is the simplest simulation. It shows the basic structure of the model and the simulation classes. "
                         "The basic 8 steps are explained: \n"
                         "1. How to import pyDAE module \n"
                         "2. How to declare variable types \n"
                         "3. How to define a model by deriving a class from the base daeModel, etc \n"
                         "4. How to define a simulation by deriving a class from the base daeSimulation, etc \n"
                         "5. How to create auxiliary objects required for the simulation (DAE solver, data reporter and logging objects) \n"
                         "6. How to set simulation's additional settings \n"
                         "7. How to connect the TCP/IP data reporter \n"
                         "8. How to run a simulation \n");
    }

public:
    void SetUpParametersAndDomains(void)
    {
    /*
    4.2 Define the domains and parameters
        Every simulation class must implement SetUpParametersAndDomains method, even if it is empty.
        It is used to set the values of the parameters, create domains etc. In this example nothing has to be done.
    */
    }

    void SetUpVariables(void)
    {
    /*
    4.3 Set initial conditions, initial guesses, fix degreees of freedom, etc.
        Every simulation class must implement SetUpVariables method, even if it is empty.
        In this example the only thing needed to be done is to set the initial condition for the variable tau to 0.
        That can be done by using SetInitialCondition function:
    */
        m.tau.SetInitialCondition(0);
    }
};

void runWhatsTheTime(void)
{ 
/*
5.Create Log, Solver, DataReporter and Simulation object
    Every simulation requires the following four objects:
    - log is used to send the messages from other parts of the framework, informs us about the simulation progress or errors
    - solver is DAE solver used to solve the underlying system of differential and algebraic equations
    - datareporter is used to send the data from the solver to daePlotter
    - simulation object
*/
    boost::scoped_ptr<daeSimulation_t>      pSimulation(new simWhatsTheTime);  
    boost::scoped_ptr<daeDataReporter_t>    pDataReporter(daeCreateTCPIPDataReporter());
    boost::scoped_ptr<daeIDASolver>         pDAESolver(new daeIDASolver());
    boost::scoped_ptr<daeLog_t>             pLog(daeCreateStdOutLog());
    
    if(!pSimulation)
        daeDeclareAndThrowException(exInvalidPointer); 
    if(!pDataReporter)
        daeDeclareAndThrowException(exInvalidPointer); 
    if(!pDAESolver)
        daeDeclareAndThrowException(exInvalidPointer); 
    if(!pLog)
        daeDeclareAndThrowException(exInvalidPointer); 

/*
6. Additional settings
Here some additional information can be entered. The most common are:
    - TimeHorizon: the duration of the simulation
    - ReportingInterval: the interval to send the variable values
    - Selection of the variables which values will be reported. It can be set individually for each variable by using the property var.ReportingOn = True/False,
      or by the function SetReportingOn in daeModel class which enables reporting of all variables in that model
*/
    pSimulation->SetTimeHorizon(500);
    pSimulation->SetReportingInterval(10);
    pSimulation->GetModel()->SetReportingOn(true);
    
/*
7. Connect the data reporter
    daeTCPIPDataReporter data reporter uses TCP/IP protocol to send the results to the daePlotter.
    It contains the function Connect which accepts two arguments:
        - TCP/IP address and port as a string in the following form: '127.0.0.1:50000'.
          The default is an empty string which allows the data reporter to connect to the local (on this machine) daePlotter listening on the port 50000.
        - Process name; in this example we use the combination of the simulation name and the current date and time
*/
    time_t rawtime;
    struct tm* timeinfo;
    char buffer[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);   
    strftime (buffer, 80, " [%d.%m.%Y %H:%M:%S]", timeinfo);
    string simName = pSimulation->GetModel()->GetName() + buffer;
    if(!pDataReporter->Connect(string(""), simName))
        daeDeclareAndThrowException(exInvalidCall); 

/*
8. Run the simulation
    8.1 The simulation initialization
        The first task is to initialize the simulation by calling the function Initialize. As the 4th argument, it accepts  an optional
        CalculateSensitivities (bool; default is False) which can enable calculation of sensitivities for given opt. variables.
        After the successful initialization the model report can be saved.
        The function SaveModelReport exports the model report in the XML format which can be opened in a web browser
        (like Mozilla Firefox, or others that support XHTML+MathMl standard).
        The function SaveRuntimeModelReport creates a runtime sort of the model report (with the equations fully expanded)
*/
    pSimulation->Initialize(pDAESolver.get(), pDataReporter.get(), pLog.get());
    
/*
    Save the model report and the runtime model report (optional)
*/
    pSimulation->GetModel()->SaveModelReport(pSimulation->GetModel()->GetName() + ".xml");
    pSimulation->GetModel()->SaveRuntimeModelReport(pSimulation->GetModel()->GetName() + "-rt.xml");
  
/*
    8.2 Solve at time = 0 (initialization)
        The DAE system must be first initialized. The function SolveInitial is used for that purpose.
*/
    pSimulation->SolveInitial();

/*
    8.3 Call the function Run from the daeSimulation class to start the simulation.
        It will last for TimeHorizon seconds and the results will be reported after every ReportingInterval number of seconds
*/
    pSimulation->Run();

/*
    8.4 Finally, call the function Finalize to clean-up.
*/
    pSimulation->Finalize();
} 
