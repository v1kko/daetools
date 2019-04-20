/********************************************************************************
                 DAE Tools: cDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic
*********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
*********************************************************************************/

/* What is the time? (AKA Hello world!) is a very simple model.
The model consists of a single variable ('time') and a single differential equation::

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
  8. Initializing, running and finalizing the simulation */

/* 1. Include daetools.h header (DAE Tools API is defined in several C++ namespaces). */
#include <daetools.h>
using namespace daetools::logging;
using namespace daetools::core;
using namespace daetools::solver;
using namespace daetools::datareporting;
using namespace daetools::activity;

/* Predefined base and derived units are defined in the units_pool namespace. */
using units_pool::s;

/* 2. Define variable types
 *    Variable types are typically declared outside of model classes since they define common, reusable types.
 *    Pre-defined variable types are located in the 'variable_types.h' header (namespace variable_types).
 *    The daeVariable constructor takes 6 arguments:
 *     - Name: string
 *     - Units: string
 *     - LowerBound: float
 *     - UpperBound: float
 *     - InitialGuess: float
 *     - AbsoluteTolerance: float
 *    Here, a very simple variable type is declared. */
namespace vt = daetools::core::variable_types;
daeVariableType typeTime("typeTime", s, 0, 1E10, 0, 1e-5);

/* 3. Define a model by deriving a new class from the base daeModel class. */
class modWhatsTheTime : public daeModel
{
public:
    daeVariable tau;

    /*  3.1 Declare domains/parameters/variables/ports
     *      All domains, parameters, variables, ports etc has to be declared as members of
     *      the new model class (except equations which are handled by the framework), since the base class
     *      keeps only their references.
     *      Domains, parameters, variables, ports, etc has to be defined in the constructor
     *      First, the base class constructor has to be called.
     *      In this example we declare only one variable: tau.
     *      daeVariable constructor accepts 3 arguments:
     *       - Name: std::string
     *       - VariableType: daeVariableType
     *       - Parent: daeModel object (indicating the model where the variable will be added)
     *       - Description: std::string (optional argument - the variable description; the default value is an empty std::string)
     *      Variable names can consists of letters, numbers, underscore ('_') and no other special characters (' ', '.', ',', '/', '-', etc).
     *      However, there is an exception. You can use the standards html code names, such as Greek characters: &alpha; &beta; &gamma; etc.
     *      Internally they will be used without '&' and ';' characters: alpha, beta, gamma, ...;
     *      but, in the equations exported in the MathML or Latex format they will be shown as native Greek letters.
     *      Also if you write the variable name as: Name_1 it will be transformed into Name with the subscript 1.
     *      In this example we use Greek character 'τ' to name the variable 'tau'. */
    modWhatsTheTime(std::string strName, daeModel* pParent = NULL, std::string strDescription = "")
      : daeModel(strName, pParent, strDescription),
        tau("&tau;", typeTime, this, "Time elapsed in the process")
    {
    }

    void DeclareEquations(void)
    {
        /*  3.2 Declare equations and state transition networks
         *      Here we declare only one equation. Equations are created by the function CreateEquation in daeModel class.
         *      It accepts two arguments: equation name and description (optional). All naming conventions apply here as well.
         *      Equations are written in the form of a residual, which is accepted by DAE equation system solvers:
         *                              'residual expression' = 0
         *      Residuals are defined by creating the above expression by using the basic mathematical operations (+, -, * and /)
         *      and functions (sqrt, log, sin, cos, max, min, ...) on variables and parameters. Variables define several useful functions:
         *          - operator () which calculates the variable value
         *          - function dt() which calculates a time derivative of the variable
         *          - function d() and d2() which calculate a partial derivative of the variable of the 1st and the 2nd order, respectively
         *      In this example we simply write that the variable time derivative is equal to 1:
         *                              d(tau) / dt = 1
         *      which calculates the time as the current time elapsed in the simulation (which is a common way to obtain the current time within the model).
         *      Note that the variables should be accessed through the model object, therefore we use self.tau */
        daeEquation* eq;

        eq = CreateEquation("Time", "Differential equation to calculate the time elapsed in the process.");
        eq->SetResidual(tau.dt() - Constant(1));
    }
};

/* 4. Define a simulation by deriving a new class from the base daeSimulation class. */
class simWhatsTheTime : public daeSimulation
{
public:
    modWhatsTheTime model;

public:
    /* 4.1 The model to be simulated has to be instantiated and set. */
    simWhatsTheTime(void) : model("cdaeWhatsTheTime")
    {
        SetModel(&model);
    }

public:
    void SetUpParametersAndDomains(void)
    {
        /* 4.2 Define the domains and parameters
         *     Every simulation class must implement SetUpParametersAndDomains method, even if it is empty.
         *     It is used to set the values of the parameters, create domains etc. In this example nothing has to be done. */
    }

    void SetUpVariables(void)
    {
        /* 4.3 Set initial conditions, initial guesses, fix degreees of freedom, etc.
         *     Every simulation class must implement SetUpVariables method, even if it is empty.
         *     In this example the only thing needed to be done is to set the initial condition for the variable tau to 0.
         *     That can be done by using SetInitialCondition function: */
        model.tau.SetInitialCondition(0 * s);
    }
};

int main(int argc, char *argv[])
{
    /* 5. Create Log, Solver, DataReporter and Simulation object
     *    Every simulation requires the following four objects:
     *     - log is used to send the messages from other parts of the framework, informs us about the simulation progress or errors
     *     - solver is DAE solver used to solve the underlying system of differential and algebraic equations
     *     - datareporter is used to send the data from the solver to daePlotter
     *     - simulation object */
    std::unique_ptr<daeSimulation_t>      pSimulation  (new simWhatsTheTime);
    std::unique_ptr<daeDataReporter_t>    pDataReporter(daeCreateTCPIPDataReporter());
    std::unique_ptr<daeDAESolver_t>       pDAESolver   (daeCreateIDASolver());
    std::unique_ptr<daeLog_t>             pLog         (daeCreateStdOutLog());

    /* 6. Additional settings
     *    Here some additional information can be entered. The most common are:
     *     - TimeHorizon: the duration of the simulation
     *     - ReportingInterval: the interval to send the variable values
     *     - Selection of the variables which values will be reported. It can be set individually for each variable by using the property var.ReportingOn = True/False,
     *       or by the function SetReportingOn in daeModel class which enables reporting of all variables in that model */
    pSimulation->SetTimeHorizon(500);
    pSimulation->SetReportingInterval(10);
    pSimulation->GetModel()->SetReportingOn(true);

    /* 7. Connect the data reporter
     *    daeTCPIPDataReporter data reporter uses TCP/IP protocol to send the results to the daePlotter.
     *    It contains the function Connect which accepts two arguments:
     *      - TCP/IP address and port as a std::string in the following form: '127.0.0.1:50000'.
     *        The default is an empty std::string which allows the data reporter to connect to the local (on this machine) daePlotter listening on the port 50000.
     *      - Process name; in this example we use the combination of the simulation name and the current date and time. */
    pDataReporter->Connect("", "cdae_whats_the_time-" + daetools::getFormattedDateTime());

    /* 8. Run the simulation
     *  8.1 The simulation initialization
     *      The first task is to initialize the simulation by calling the function Initialize. As the 4th argument, it accepts  an optional
     *      CalculateSensitivities (bool; default is False) which can enable calculation of sensitivities for given opt. variables.
     *      After the successful initialization the model report can be saved.
     *      The function SaveModelReport exports the model report in the XML format which can be opened in a web browser
     *      (like Mozilla Firefox, or others that support XHTML+MathMl standard).
     *      The function SaveRuntimeModelReport creates a runtime sort of the model report (with the equations fully expanded). */
    pSimulation->Initialize(pDAESolver.get(), pDataReporter.get(), pLog.get());

    /* Optional: save the model report and the runtime model report. */
    //pSimulation->GetModel()->SaveModelReport(pSimulation->GetModel()->GetName() + ".xml");
    //pSimulation->GetModel()->SaveRuntimeModelReport(pSimulation->GetModel()->GetName() + "-rt.xml");

    /*  8.2 Calculate corected initial conditions at time = 0. */
    pSimulation->SolveInitial();

    /*  8.3 Call the function Run from the daeSimulation class to start the simulation.
     *      It will last for TimeHorizon seconds and the results will be reported after every ReportingInterval number of seconds. */
    pSimulation->Run();

    /*  8.4 Finally, call the function Finalize to clean-up. */
    pSimulation->Finalize();
}
