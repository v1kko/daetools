#ifndef DAE_PYTHON_WRAPS_H
#define DAE_PYTHON_WRAPS_H

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
#pragma warning(disable: 4250)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

#if defined(__MACH__) || defined(__APPLE__)
#include <python.h>
#endif
#include <string>
#include <boost/python.hpp>
#include <boost/python/slice.hpp>
#include <boost/python/dict.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/python/call_method.hpp>
#include <boost/python/reference_existing_object.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>

#include <OpenCS/models/cs_model_builder.h>
#include <OpenCS/models/partitioner_simple.h>
#include <OpenCS/models/partitioner_metis.h>
#include <OpenCS/models/partitioner_2d_npde.h>
#include <OpenCS/models/cs_partitioners.h>
#include <OpenCS/simulators/cs_logs.h>
#include <OpenCS/simulators/cs_data_reporters.h>
#include <OpenCS/evaluators/cs_evaluators.h>
#include <OpenCS/simulators/cs_simulators.h>
#include <OpenCS/models/cs_dae_model.h>
using namespace cs;

namespace daepython
{
template<typename ITEM>
boost::python::list getListFromVector(const std::vector<ITEM>& arrItems)
{
    boost::python::list l;

    for(size_t i = 0; i < arrItems.size(); i++)
        l.append(arrItems[i]);

    return l;
}

template<typename ITEM>
boost::python::list getListFromCArray(const ITEM* items, size_t n)
{
    boost::python::list l;

    for(size_t i = 0; i < n; i++)
        l.append(items[i]);

    return l;
}

template<typename ITEM>
boost::python::list getListFromVectorByRef(const std::vector<ITEM>& arrItems)
{
    boost::python::list l;

    for(size_t i = 0; i < arrItems.size(); i++)
        l.append(boost::ref(arrItems[i]));

    return l;
}

template<typename ITEM>
void getVectorFromList(boost::python::list& objects, std::vector<ITEM>& vec)
{
    boost::python::ssize_t n = boost::python::len(objects);
    vec.clear();
    vec.resize(n);
    for(boost::python::ssize_t i = 0; i < n; i++)
        vec[i] = boost::python::extract<ITEM>(objects[i]);
}

template<typename ITEM>
void getSetFromList(boost::python::list& objects, std::set<ITEM>& s)
{
    boost::python::ssize_t n = boost::python::len(objects);
    s.clear();
    for(boost::python::ssize_t i = 0; i < n; i++)
        s.insert( boost::python::extract<ITEM>(objects[i]) );
}

/*******************************************************
    csNumber_t
*******************************************************/
csNumber_t true_divide1(csNumber_t &a, csNumber_t &b);
csNumber_t true_divide2(const csNumber_t &a, const real_t v);
csNumber_t true_divide3(const real_t v, const csNumber_t &a);
csNumber_t floor_divide1(const csNumber_t &a, const csNumber_t &b);
csNumber_t floor_divide2(const csNumber_t &a, const real_t v);
csNumber_t floor_divide3(const real_t v, const csNumber_t &a);

std::string csNumber_str(csNumber_t& self);
std::string csNumber_repr(csNumber_t& self);

/*******************************************************
    csModelBuilder_t
*******************************************************/
const csNumber_t& GetTime(csModelBuilder_t& mb);
boost::python::list GetDegreesOfFreedom(csModelBuilder_t& mb);
boost::python::list GetVariables(csModelBuilder_t& mb);
boost::python::list GetTimeDerivatives(csModelBuilder_t& mb);

boost::python::list GetModelEquations(csModelBuilder_t& mb);
boost::python::list GetVariableValues(csModelBuilder_t& mb);
boost::python::list GetVariableTimeDerivatives(csModelBuilder_t& mb);
boost::python::list GetDegreeOfFreedomValues(csModelBuilder_t& mb);
boost::python::list GetVariableNames(csModelBuilder_t& mb);
boost::python::list GetVariableTypes(csModelBuilder_t& mb);
boost::python::list GetAbsoluteTolerances(csModelBuilder_t& mb);

// DAE Tools-specific initialisation function
void Initialise_DAETools_DAE_System(csModelBuilder_t& mb, boost::python::dict mb_data);

void SetModelEquations(csModelBuilder_t& mb, boost::python::list objs);
void SetVariableValues(csModelBuilder_t& mb, boost::python::list objs);
void SetVariableTimeDerivatives(csModelBuilder_t& mb, boost::python::list objs);
void SetDegreeOfFreedomValues(csModelBuilder_t& mb, boost::python::list objs);
void SetVariableNames(csModelBuilder_t& mb, boost::python::list objs);
void SetVariableTypes(csModelBuilder_t& mb, boost::python::list objs);
void SetAbsoluteTolerances(csModelBuilder_t& mb, boost::python::list objs);

boost::python::object GetSimulationOptions(csModelBuilder_t& mb);
void SetSimulationOptions(csModelBuilder_t& mb, boost::python::dict options);
boost::python::object GetDefaultSimulationOptions_DAE();
boost::python::object GetDefaultSimulationOptions_ODE();

boost::python::list PartitionSystem(csModelBuilder_t&     mb,
                                    uint32_t              Npe,
                                    csGraphPartitioner_t* graphPartitioner,
                                    boost::python::list   balancingConstraints  = boost::python::list(),
                                    bool                  logPartitionResults   = false,
                                    boost::python::dict   unaryOperationsFlops  = boost::python::dict(),
                                    boost::python::dict   binaryOperationsFlops = boost::python::dict());

void ExportModels(boost::python::list models,
                  const std::string&  outputDirectory,
                  boost::python::dict simulationOptions);

/*******************************************************
    Simulate functions
*******************************************************/
void csSimulate_dir(const std::string& inputDirectory);
void csSimulate_model(csModelPtr         csModel,
                      const std::string& jsonOptions,
                      const std::string& simulationDirectory);


/*******************************************************
    csDifferentialEquationModel wrapper
*******************************************************/
// Load functions require MPI rank (at the moment not available)
class csDifferentialEquationModel_Wrapper : public csDifferentialEquationModel,
                                            public boost::python::wrapper<csDifferentialEquationModel>
{
public:
    csDifferentialEquationModel_Wrapper(void)
    {
    }
};

/*******************************************************
    csGraphPartitioner_t wrapper
*******************************************************/
// Support for Python GIL
class pyGILState
{
public:
    pyGILState(bool doNothing_ = true) : doNothing(doNothing_)
    {
        if(doNothing)
            return;

        // Acquire the GIL
        gstate = PyGILState_Ensure();
    }

    virtual ~pyGILState()
    {
        if(doNothing)
            return;

        // Release the thread. No Python API allowed beyond this point
        PyGILState_Release(gstate);
    }

protected:
    const bool       doNothing;
    PyGILState_STATE gstate;
};


class csLog_Wrapper : public csLog_t,
                      public boost::python::wrapper<csLog_t>
{
public:
    csLog_Wrapper(void)
    {
    }

    std::string GetName() const
    {
        return this->get_override("GetName")();
    }

    bool Connect(int rank)
    {
        return this->get_override("Connect")();
    }

    void Disconnect()
    {
        this->get_override("Disconnect")();
    }

    bool IsConnected()
    {
        return this->get_override("IsConnected")();
    }

    void Message(const std::string& strMessage)
    {
        this->get_override("Message")(strMessage);
    }
};


class csDataReporter_Wrapper : public csDataReporter_t,
                               public boost::python::wrapper<csDataReporter_t>
{
public:
    csDataReporter_Wrapper(void)
    {
    }

    std::string GetName() const
    {
        return this->get_override("GetName")();
    }

    bool Connect(int rank)
    {
        return this->get_override("Connect")(rank);
    }

    bool IsConnected()
    {
        return this->get_override("IsConnected")();
    }

    bool Disconnect()
    {
        return this->get_override("Disconnect")();
    }

    bool RegisterVariables(const std::vector<std::string>& variableNames)
    {
        return this->get_override("RegisterVariables")();
    }

    bool StartNewResultSet(real_t time)
    {
        return this->get_override("StartNewResultSet")(time);
    }

    bool EndOfData()
    {
        return this->get_override("EndOfData")();
    }

    bool SendVariables(const real_t* values, const size_t n)
    {
        boost::python::list vals_l = getListFromCArray(values, n);
        return this->get_override("SendVariables")(vals_l);
    }

    bool SendDerivatives(const real_t* derivatives, const size_t n)
    {
        boost::python::list derivs_l = getListFromCArray(derivatives, n);
        return this->get_override("SendDerivatives")(derivs_l);
    }
};

class csGraphPartitioner_Wrapper : public csGraphPartitioner_t,
                                   public boost::python::wrapper<csGraphPartitioner_t>
{
public:
    csGraphPartitioner_Wrapper(void)
    {
    }

    std::string GetName()
    {
        return this->get_override("GetName")();
    }

    boost::python::list Partition_(int                  Npe,
                                   int                  Nvertices,
                                   int                  Nconstraints,
                                   boost::python::list  rowIndices_l,
                                   boost::python::list  colIndices_l,
                                   boost::python::list  vertexWeights_l)
    {
        throw std::runtime_error("The method Partition must be implemented in csGraphPartitioner_t-derived classes");
        return boost::python::list();
    }

    int Partition(int32_t                               Npe,
                  int32_t                               Nvertices,
                  int32_t                               Nconstraints,
                  std::vector<uint32_t>&                rowIndices,
                  std::vector<uint32_t>&                colIndices,
                  std::vector< std::vector<int32_t> >&  vertexWeights,
                  std::vector< std::set<int32_t> >&     partitions)
    {
        pyGILState GIL(false);

        if(boost::python::override fnPartition = this->get_override("Partition"))
        {
            boost::python::list rowIndices_l;
            boost::python::list colIndices_l;
            boost::python::list vertexWeights_l;
            boost::python::list partitions_l;

            rowIndices_l = getListFromVector(rowIndices);
            colIndices_l = getListFromVector(colIndices);
            for(int i = 0; i < vertexWeights.size(); i++)
            {
                const std::vector<int32_t>& vertexWeight_i = vertexWeights[i];
                boost::python::list vertexWeight = getListFromVector(vertexWeight_i);
                vertexWeights_l.append(vertexWeight);
            }

            boost::python::object res = fnPartition(Npe,
                                                    Nvertices,
                                                    Nconstraints,
                                                    rowIndices_l,
                                                    colIndices_l,
                                                    vertexWeights_l);

            partitions_l = boost::python::extract<boost::python::list>(res);
            boost::python::ssize_t n = boost::python::len(partitions_l);
            if(n > 0)
            {
                partitions.resize(n);
                for(boost::python::ssize_t i = 0; i < n; i++)
                {
                    boost::python::list p = boost::python::extract<boost::python::list>(partitions_l[i]);
                    getSetFromList(p, partitions[i]);
                }
            }
        }
        else
        {
            throw std::runtime_error("csGraphPartitioner_t::Partition is not implemented");
        }

        return 0;
    }
};

}

#endif
