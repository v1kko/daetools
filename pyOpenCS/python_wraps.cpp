#include "python_wraps.h"
using namespace std;
using namespace boost;
using namespace boost::python;

namespace daepython
{
csNumber_t true_divide1(csNumber_t &a, csNumber_t &b)
{
    return a/b;
}

csNumber_t true_divide2(const csNumber_t &a, const real_t v)
{
    return a/v;
}

csNumber_t true_divide3(const real_t v, const csNumber_t &a)
{
    return v/a;
}

csNumber_t floor_divide1(const csNumber_t &a, const csNumber_t &b)
{
    return a/b;
}

csNumber_t floor_divide2(const csNumber_t &a, const real_t v)
{
    return a/v;
}

csNumber_t floor_divide3(const real_t v, const csNumber_t &a)
{
    return v/a;
}

std::string csNumber_str(csNumber_t& self)
{
    if(self.node)
        return self.node->ToLatex();
    else
        return "csNumber_t()";
}

std::string csNumber_repr(csNumber_t& self)
{
    if(self.node)
        return self.node->ToLatex();
    else
        return "csNumber_t()";
}

const csNumber_t& GetTime(csModelBuilder_t& mb)
{
    return mb.GetTime();
}

boost::python::list GetDegreesOfFreedom(csModelBuilder_t& mb)
{
    return getListFromVector<csNumber_t>(mb.GetDegreesOfFreedom());
}

boost::python::list GetVariables(csModelBuilder_t& mb)
{
    return getListFromVector<csNumber_t>(mb.GetVariables());
}

boost::python::list GetTimeDerivatives(csModelBuilder_t& mb)
{
    return getListFromVector<csNumber_t>(mb.GetTimeDerivatives());
}

boost::python::list GetModelEquations(csModelBuilder_t& mb)
{
    return getListFromVector(mb.GetModelEquations());
}

boost::python::list GetVariableValues(csModelBuilder_t& mb)
{
    return getListFromVector(mb.GetVariableValues());
}

boost::python::list GetVariableTimeDerivatives(csModelBuilder_t& mb)
{
    return getListFromVector(mb.GetVariableTimeDerivatives());
}

boost::python::list GetDegreeOfFreedomValues(csModelBuilder_t& mb)
{
    return getListFromVector(mb.GetDegreeOfFreedomValues());
}

boost::python::list GetVariableNames(csModelBuilder_t& mb)
{
    return getListFromVector(mb.GetVariableNames());
}

boost::python::list GetVariableTypes(csModelBuilder_t& mb)
{
    return getListFromVector(mb.GetVariableTypes());
}

boost::python::list GetAbsoluteTolerances(csModelBuilder_t& mb)
{
    return getListFromVector(mb.GetAbsoluteTolerances());
}

// DAE Tools-specific initialisation function
void Initialise_DAETools_DAE_System(csModelBuilder_t& mb, boost::python::dict mb_data)
{
    uint32_t                  Ndofs;
    uint32_t                  Nvariables;
    std::vector<std::string>  variableNames;
    std::vector<real_t>       variableValues;
    std::vector<real_t>       variableDerivatives;
    std::vector<real_t>       absTolerances;
    std::vector<std::string>  dofNames;
    std::vector<real_t>       dofValues;
    std::vector< csNumber_t > equations;
    std::vector<cs::csNodePtr> nodes;

    Ndofs               = boost::python::extract< uint32_t >                   (mb_data["Ndofs"]);
    Nvariables          = boost::python::extract< uint32_t >                   (mb_data["Nvariables"]);
    variableNames       = boost::python::extract< std::vector<std::string> >   (mb_data["variableNames"]);
    variableValues      = boost::python::extract< std::vector<real_t> >        (mb_data["variableValues"]);
    variableDerivatives = boost::python::extract< std::vector<real_t> >        (mb_data["variableDerivatives"]);
    absTolerances       = boost::python::extract< std::vector<real_t> >        (mb_data["absTolerances"]);
    dofNames            = boost::python::extract< std::vector<std::string> >   (mb_data["dofNames"]);
    dofValues           = boost::python::extract< std::vector<real_t> >        (mb_data["dofValues"]);
    nodes               = boost::python::extract< std::vector<cs::csNodePtr> > (mb_data["equations"]);

    equations.resize(Nvariables);
    for(size_t i = 0; i < Nvariables; i++)
    {
        csNumber_t n;
        n.node = nodes[i];
        equations[i] = n;
    }

    mb.Initialize_DAE_System(Nvariables, Ndofs);

    mb.SetModelEquations          (equations);
    mb.SetVariableNames           (variableNames);
    mb.SetVariableValues          (variableValues);
    mb.SetVariableTimeDerivatives (variableDerivatives);
    mb.SetAbsoluteTolerances      (absTolerances);
    mb.SetDegreeOfFreedomValues   (dofValues);
    mb.SetDegreeOfFreedomNames    (dofNames);
}

void SetModelEquations(csModelBuilder_t& mb, boost::python::list objs)
{
    std::vector<csNumber_t> vec;
    getVectorFromList(objs, vec);
    mb.SetModelEquations(vec);
}

void SetVariableValues(csModelBuilder_t& mb, boost::python::list objs)
{
    std::vector<real_t> vec;
    getVectorFromList(objs, vec);
    mb.SetVariableValues(vec);
}

void SetVariableTimeDerivatives(csModelBuilder_t& mb, boost::python::list objs)
{
    std::vector<real_t> vec;
    getVectorFromList(objs, vec);
    mb.SetVariableTimeDerivatives(vec);
}

void SetDegreeOfFreedomValues(csModelBuilder_t& mb, boost::python::list objs)
{
    std::vector<real_t> vec;
    getVectorFromList(objs, vec);
    mb.SetDegreeOfFreedomValues(vec);
}

void SetVariableNames(csModelBuilder_t& mb, boost::python::list objs)
{
    std::vector<std::string> vec;
    getVectorFromList(objs, vec);
    mb.SetVariableNames(vec);
}

void SetVariableTypes(csModelBuilder_t& mb, boost::python::list objs)
{
    std::vector<int32_t> vec;
    getVectorFromList(objs, vec);
    mb.SetVariableTypes(vec);
}

void SetAbsoluteTolerances(csModelBuilder_t& mb, boost::python::list objs)
{
    std::vector<real_t> vec;
    getVectorFromList(objs, vec);
    mb.SetAbsoluteTolerances(vec);
}

static boost::python::object get_dict_from_json_string(const std::string& jsonOptions)
{
    boost::python::object main_module = import("__main__");
    boost::python::object main_namespace = main_module.attr("__dict__");
    exec("import json", main_namespace);
    boost::python::object json = main_namespace["json"];

    boost::python::dict kwargs;
    boost::python::object options_o(jsonOptions);
    boost::python::tuple args = boost::python::make_tuple(options_o);

    boost::python::object options_d = json.attr("loads")(*args, **kwargs);
    return options_d;
}

static std::string get_json_string_from_dict(boost::python::dict dictOptions)
{
    std::string jsonOptions;

    boost::python::object main_module = import("__main__");
    boost::python::object main_namespace = main_module.attr("__dict__");
    exec("import json", main_namespace);
    boost::python::object json = main_namespace["json"];

    boost::python::dict kwargs;
    kwargs["indent"] = 4;
    boost::python::tuple args = boost::python::make_tuple(dictOptions);
    boost::python::object options_o = json.attr("dumps")(*args, **kwargs);

    jsonOptions = boost::python::extract<std::string>(options_o);
    return jsonOptions;
}

boost::python::object GetDefaultSimulationOptions_DAE()
{
    return get_dict_from_json_string(csModelBuilder_t::GetDefaultSimulationOptions_DAE());
}

boost::python::object GetDefaultSimulationOptions_ODE()
{
    return get_dict_from_json_string(csModelBuilder_t::GetDefaultSimulationOptions_ODE());
}

boost::python::object GetSimulationOptions(csModelBuilder_t& mb)
{
    csSimulationOptionsPtr so = mb.GetSimulationOptions();
    std::string jsonOptions = so->ToString();
    return get_dict_from_json_string(jsonOptions);
}

void SetSimulationOptions(csModelBuilder_t& mb, boost::python::dict options_d)
{
    csSimulationOptionsPtr so = mb.GetSimulationOptions();
    std::string options_s = get_json_string_from_dict(options_d);
    so->LoadString(options_s);
}

boost::python::list PartitionSystem(csModelBuilder_t&     mb,
                                    uint32_t              Npe,
                                    csGraphPartitioner_t* graphPartitioner,
                                    boost::python::list   balancingConstraints_l,
                                    bool                  logPartitionResults,
                                    boost::python::dict   unaryOperationsFlops_d,
                                    boost::python::dict   binaryOperationsFlops_d)
{
    std::vector<std::string> balancingConstraints;
    std::map<csUnaryFunctions,uint32_t>  unaryOperationsFlops;
    std::map<csBinaryFunctions,uint32_t> binaryOperationsFlops;

    std::vector<csModelPtr> models = mb.PartitionSystem(Npe,
                                                        graphPartitioner,
                                                        balancingConstraints,
                                                        logPartitionResults,
                                                        unaryOperationsFlops,
                                                        binaryOperationsFlops);

    return getListFromVector(models);
}

void ExportModels(boost::python::list models_l,
                  const std::string&  outputDirectory,
                  boost::python::dict simulationOptions)
{
    std::vector<csModelPtr> models;
    getVectorFromList(models_l, models);

    boost::python::object main_module = import("__main__");
    boost::python::object main_namespace = main_module.attr("__dict__");
    exec("import json", main_namespace);
    boost::python::object json = main_namespace["json"];

    boost::python::dict kwargs;
    kwargs["indent"] = 4;
    boost::python::tuple args = boost::python::make_tuple(simulationOptions);
    boost::python::object options_o = json.attr("dumps")(*args, **kwargs);
    std::string options_s = boost::python::extract<std::string>(options_o);

    csModelBuilder_t::ExportModels(models, outputDirectory, options_s);
}

void csSimulate_dir(const std::string& inputDirectory)
{
    std::shared_ptr<csLog_t>          plog;
    std::shared_ptr<csDataReporter_t> pdatareporter;

    cs::csSimulate(inputDirectory, plog, pdatareporter, true);
}

void csSimulate_model(csModelPtr         csModel,
                      const std::string& jsonOptions,
                      const std::string& simulationDirectory)
{
    std::shared_ptr<csLog_t>          plog;
    std::shared_ptr<csDataReporter_t> pdatareporter;

    cs::csSimulate(csModel, jsonOptions, simulationDirectory, plog, pdatareporter, true);
}


}
