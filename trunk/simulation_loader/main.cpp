#include "simulation_loader.h"

int main(int argc, char *argv[])
{
    try
    {
        dae::activity::simulation_loader::daeSimulationLoader loader;
        loader.LoadPythonSimulation("../daetools-package/daetools/examples/tutorial2.py", "simTutorial");
        loader.Simulate(true);
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
}

/*
#include <algorithm>
#include <functional>
#include <string>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/slice.hpp>
#include <boost/smart_ptr.hpp>
#include "../dae_develop.h"
#include "../DataReporting/datareporters.h"
#include "../Activity/simulation.h"
#include "../IDAS_DAESolver/ida_solver.h"

using std::string;
using std::vector;
using boost::python::object;
using boost::python::list;
using boost::python::dict;
using boost::python::tuple;
using boost::python::exec;
using boost::python::eval;
using boost::python::extract;
using boost::python::import;
using boost::python::len;

string simulation = ""
"log          = daePythonStdOutLog()\n"
"daesolver    = daeIDAS()\n"
"datareporter = daeTCPIPDataReporter()\n"
"simulation   = simTutorial()\n"
"simulation.TimeHorizon = 10\n"
"simulation.ReportingInterval = 1\n"
"simulation.m.SetReportingOn(True)\n"
"simName = simulation.m.Name + strftime(' [%d.%m.%Y %H:%M:%S]', localtime())\n"
"if(datareporter.Connect('', simName) == False):\n"
"    sys.exit()\n"
"simulation.Initialize(daesolver, datareporter, log)";


void CollectTopeLevelPorts(daeModel* pModel, std::map<string, daePort*>& mapPorts);
void CollectAllParameters(daeModel* pModel, std::map<string, daeParameter*>& mapParameters);
void CollectAllVariables(daeModel* pModel, std::map<string, daeVariable*>& mapVariables);

void CollectAllParameters(daeModel* pModel, std::map<string, daeParameter*>& mapParameters)
{
    // Insert objects from the model
    for(std::vector<daeParameter*>::const_iterator iter = pModel->Parameters().begin(); iter != pModel->Parameters().end(); iter++)
        mapParameters[(*iter)->GetCanonicalName()] = *iter;

    // Insert objects from the ports
    for(std::vector<daePort*>::const_iterator piter = pModel->Ports().begin(); piter != pModel->Ports().end(); piter++)
        for(std::vector<daeParameter*>::const_iterator citer = (*piter)->Parameters().begin(); citer != (*piter)->Parameters().end(); citer++)
            mapParameters[(*citer)->GetCanonicalName()] = *citer;

    // Insert objects from the child models (units)
    for(std::vector<daeModel*>::const_iterator miter = pModel->Models().begin(); miter != pModel->Models().end(); miter++)
        CollectAllParameters(*miter, mapParameters);
}

void CollectAllVariables(daeModel* pModel, std::map<string, daeVariable*>& mapVariables)
{
    // Insert objects from the model
    for(std::vector<daeVariable*>::const_iterator iter = pModel->Variables().begin(); iter != pModel->Variables().end(); iter++)
        mapVariables[(*iter)->GetCanonicalName()] = *iter;

    // Insert objects from the ports
    for(std::vector<daePort*>::const_iterator piter = pModel->Ports().begin(); piter != pModel->Ports().end(); piter++)
        for(std::vector<daeVariable*>::const_iterator citer = (*piter)->Variables().begin(); citer != (*piter)->Variables().end(); citer++)
            mapVariables[(*citer)->GetCanonicalName()] = *citer;

    // Insert objects from the child models (units)
    for(std::vector<daeModel*>::const_iterator miter = pModel->Models().begin(); miter != pModel->Models().end(); miter++)
        CollectAllVariables(*miter, mapVariables);
}

int main(int argc, char *argv[])
{
    try
    {
        string name;
        boost::python::ssize_t i, n;
        tuple t;
        object result, obj;

        Py_SetProgramName(argv[0]);
        Py_Initialize();

        boost::python::list l;
        l.append(1.0);
        l.append(2.0);
        l.append(3.0);

        object main_module = import("__main__");
        object main_namespace = main_module.attr("__dict__");

        result = exec("import os, sys, numpy", main_namespace);
        object np = main_namespace["numpy"];
        object arr = np.attr("array")(l);
        std::cout << extract<double>(arr[1]) << std::endl;

        Py_Finalize();
        return 0;

        //result = exec("import sys; print sys.path; result = str(sys.path)", main_namespace);
        //string sys_path = boost::python::extract<string>(main_namespace["result"]);

        result = exec("import os, sys, numpy", main_namespace);
        result = exec("sys.path.insert(0, '/home/ciroki/Data/daetools/trunk/daetools-package/daetools/examples'); print sys.path", main_namespace);
        //result = exec("from daetools.pyDAE import *", main_namespace);
        //result = exec("whats_the_time.consoleRun()", main_namespace);
        //result = exec("from PyQt4 import QtCore, QtGui", main_namespace);
        //result = exec("app = QtGui.QApplication(sys.argv)", main_namespace);

//        object whats_the_time_module = boost::python::import("whats_the_time");
//        object whats_the_time_module_namespace = whats_the_time_module.attr("__dict__");
//        object model = whats_the_time_module_namespace["modTutorial"];
//        result = model.attr("__init__")(model, "Uja");
        result = exec("import tutorial6", main_namespace);
        result = exec("__daetools_model__ = tutorial6.modTutorial('Uja')", main_namespace);
        daeModel* pModel = extract<daeModel*>(main_namespace["__daetools_model__"]);
        //std::cout << pModel->GetCanonicalName() << std::endl;

        std::map<string, daeVariable*> mapVariables;
        CollectAllVariables(pModel, mapVariables);
        for(std::map<string,daeVariable*>::const_iterator it = mapVariables.begin(); it != mapVariables.end(); it++)
            std::cout << it->first << std::endl;

//        result = main_namespace.attr("items")();
//        list items = extract<list>(result);
//        n = boost::python::len(items);

//        for(i = 0; i < n; i++)
//        {
//            t = extract<tuple>(items[i]);
//            name  = extract<string>(t[0]);
//            obj = extract<object>(t[1]);
//            string obj_str = extract<string>(obj.attr("__str__")());

//            std::cout << name << " = " << obj_str << std::endl;
//        }

//        result = exec("from whats_the_time import *", main_namespace);
//        // From this point I have daeSimulation pointer and can operate on it
//        result = exec(simulation.c_str(), main_namespace);
//        daeSimulation* pSimulation = boost::python::extract<daeSimulation*>(main_namespace["simulation"]);
//        pSimulation->SolveInitial();
//        pSimulation->Run();
//        pSimulation->Finalize();

        Py_Finalize();
        return 0;
    }
    catch(boost::python::error_already_set const &)
    {
        if (PyErr_ExceptionMatches(PyExc_ImportError))
        {
            PyErr_Print();
        }
        else
        {
            PyErr_Print();
        }
    }
}
*/
