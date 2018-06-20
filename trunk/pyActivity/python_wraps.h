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
//#include <boost/python/numeric.hpp>
#include <boost/python/slice.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include "../dae_develop.h"
#include "../DataReporting/datareporters.h"
#include "../Activity/simulation.h"
#include "../IDAS_DAESolver/ida_solver.h"
#include "../Core/base_logging.h"
//#include "../Core/tcpiplog.h"

namespace daepython
{
template<typename ITEM>
boost::python::list getListFromVector(const std::vector<ITEM>& arrItems)
{
    boost::python::list l;

    for(size_t i = 0; i < arrItems.size(); i++)
        l.append(boost::ref(arrItems[i]));

    return l;
}

template<typename ITEM>
boost::python::list getListFromVectorByValue(std::vector<ITEM>& arrItems)
{
    boost::python::list l;

    for(size_t i = 0; i < arrItems.size(); i++)
        l.append(arrItems[i]);

    return l;
}

template<typename ITEM>
boost::python::list getListFromVectorByValue(const ITEM* pItems, size_t n)
{
    boost::python::list l;

    for(size_t i = 0; i < n; i++)
        l.append(pItems[i]);

    return l;
}

template<typename KEY, typename VALUE>
boost::python::dict getDictFromMapByValue(std::map<KEY,VALUE>& mapItems)
{
    boost::python::dict res;
    typename std::map<KEY,VALUE>::iterator iter;

    for(iter = mapItems.begin(); iter != mapItems.end(); iter++)
    {
        KEY   key = iter->first;
        VALUE val = iter->second;
        res[key] = val;
    }

    return res;
}

template<typename TYPE1, typename TYPE2>
std::map<TYPE1,TYPE2> getMapFromDict(boost::python::dict d)
{
    std::map<TYPE1,TYPE2> m;

    // Get the list of pairs key:value
    boost::python::stl_input_iterator<boost::python::tuple> iter(d.attr("items")()), end;

    for(; iter != end; iter++)
    {
        boost::python::tuple t = *iter;
        TYPE1 key = boost::python::extract<TYPE1>(t[0]);
        TYPE2 val = boost::python::extract<TYPE1>(t[1]);
        m[key] = val;
    }

    return m;
}

template<typename TYPE>
std::vector<TYPE> getVectorFromList(boost::python::list l)
{
    std::vector<TYPE> v;

    // Get the list of pairs key:value
    size_t n = boost::python::len(l);
    v.resize(n);
    for(size_t i = 0; i < n; i++)
    {
        TYPE val = boost::python::extract<TYPE>(l[i]);
        v[i] = val;
    }

    return v;
}

boost::python::dict GetCallStats(daeSimulation& self);

class daeSimulationWrapper : public daeSimulation_t,
                             public boost::python::wrapper<daeSimulation_t>
{
public:
    daeModel_t* GetModel(void) const
    {
        return this->get_override("GetModel")();
    }

    void SetModel(daeModel_t* pModel)
    {
        this->get_override("SetModel")(pModel);
    }

    daeDataReporter_t* GetDataReporter(void) const
    {
        return this->get_override("GetDataReporter")();
    }

    daeLog_t* GetLog(void) const
    {
        return this->get_override("GetLog")();
    }

    void Run(void)
    {
        this->get_override("Run")();
    }

    void ReportData(void) const
    {
        this->get_override("ReportData")();
    }

    void SetTimeHorizon(real_t dTimeHorizon)
    {
        this->get_override("SetTimeHorizon")(dTimeHorizon);
    }

    real_t GetTimeHorizon(void) const
    {
        return this->get_override("GetTimeHorizon")();
    }

    void SetReportingInterval(real_t dReportingInterval)
    {
        this->get_override("SetReportingInterval")(dReportingInterval);
    }

    real_t GetReportingInterval(void) const
    {
        return this->get_override("GetReportingInterval")();
    }

    void Pause(void)
    {
        this->get_override("Pause")();
    }

    void Resume(void)
    {
        this->get_override("Resume")();
    }

    void Stop(void)
    {
        this->get_override("Stop")();
    }

    void Initialize(daeDAESolver_t* pDAESolver,
                    daeDataReporter_t* pDataReporter,
                    daeLog_t* pLog,
                    bool bCalculateSensitivities = false,
                    const std::string& strJSONRuntimeSettings = "")
    {
        this->get_override("Initialize")(pDAESolver, pDataReporter, pLog, bCalculateSensitivities, strJSONRuntimeSettings);
    }

    void Reinitialize(void)
    {
        this->get_override("Reinitialize")();
    }

    void SolveInitial(void)
    {
        this->get_override("SolveInitial")();
    }

    daeDAESolver_t* GetDAESolver(void) const
    {
        return this->get_override("GetDAESolver")();
    }

    void SetUpParametersAndDomains(void)
    {
        this->get_override("SetUpParametersAndDomains")();
    }

    void SetUpVariables(void)
    {
        this->get_override("SetUpVariables")();
    }

    real_t Integrate(daeeStopCriterion eStopCriterion, bool bReportDataAroundDiscontinuities = true)
    {
        return this->get_override("Integrate")(eStopCriterion, bReportDataAroundDiscontinuities);
    }

    real_t IntegrateForTimeInterval(real_t time_interval, daeeStopCriterion eStopCriterion, bool bReportDataAroundDiscontinuities = true)
    {
        return this->get_override("IntegrateForTimeInterval")(time_interval, eStopCriterion, bReportDataAroundDiscontinuities);
    }

    real_t IntegrateUntilTime(real_t time, daeeStopCriterion eStopCriterion, bool bReportDataAroundDiscontinuities = true)
    {
        return this->get_override("IntegrateUntilTime")(time, eStopCriterion, bReportDataAroundDiscontinuities);
    }

};


class daeDefaultSimulationWrapper : public daeSimulation,
                                    public boost::python::wrapper<daeSimulation>
{
public:
    daeDefaultSimulationWrapper(void)
    {
    }

    void SetModel_(boost::python::object Model)
    {
        model = Model;
        daeModel* pModel = boost::python::extract<daeModel*>(Model);
        this->m_pModel = pModel;
    }

    void Initialize(boost::python::object DAESolver,
                    boost::python::object DataReporter,
                    boost::python::object Log,
                    bool bCalculateSensitivities = false,
                    const std::string& strJSONRuntimeSettings = "")
    {
        daesolver    = DAESolver;
        datareporter = DataReporter;
        log          = Log;

        daeDAESolver_t*    pDAESolver    = boost::python::extract<daeDAESolver_t*>(daesolver);
        daeDataReporter_t* pDataReporter = boost::python::extract<daeDataReporter_t*>(datareporter);
        daeLog_t*          pLog          = boost::python::extract<daeLog_t*>(log);

        daeSimulation::Initialize(pDAESolver, pDataReporter, pLog, bCalculateSensitivities, strJSONRuntimeSettings);
    }

    boost::python::object GetModel_(void) const
    {
        return model;
    }

    boost::python::list GetDOFs(void) const
    {
        boost::python::list l;
        boost::shared_ptr<daeDataProxy_t> pDataProxy = m_pModel->GetDataProxy();
        const std::vector<real_t>& dofs = pDataProxy->GetAssignedVarsValues();
        for(size_t i = 0; i < dofs.size(); i++)
            l.append(dofs[i]);

        return l;
    }

    boost::python::list GetValues(void) const
    {
        boost::python::list l;
        boost::shared_ptr<daeDataProxy_t> pDataProxy = m_pModel->GetDataProxy();
        const std::vector<real_t*>& valRefs = pDataProxy->GetValuesReferences();
        for(size_t i = 0; i < valRefs.size(); i++)
            l.append(*valRefs[i]);

        return l;
    }

    boost::python::list GetTimeDerivatives(void) const
    {
        boost::python::list l;
        boost::shared_ptr<daeDataProxy_t> pDataProxy = m_pModel->GetDataProxy();
        const std::vector<real_t*>& dtRefs = pDataProxy->GetTimeDerivativesReferences();
        for(size_t i = 0; i < dtRefs.size(); i++)
        {
        // Assigned variables have no time derivatives mapped; thus, those items are left uninitialized (NULL)
            if(!dtRefs[i])
                l.append(0.0);
            else
                l.append(*dtRefs[i]);
        }
        return l;
    }

    boost::python::list GetVariableTypes(void) const
    {
        boost::shared_ptr<daeDataProxy_t> pDataProxy = m_pModel->GetDataProxy();
        size_t  nVars     = pDataProxy->GetTotalNumberOfVariables();
        real_t* pVarTypes = pDataProxy->GetVariableTypesPointer();

        boost::python::list l;
        for(size_t i = 0; i < nVars; i++)
            l.append(static_cast<int>(pVarTypes[i]));

        return l;
    }

    boost::python::list GetEqExecutionInfos(void) const
    {
        std::vector<daeEquationExecutionInfo*> ptrarrEquationExecutionInfos = GetEquationExecutionInfos();
        return getListFromVector(ptrarrEquationExecutionInfos);
    }

    real_t GetRelativeTolerance(void) const
    {
        if(!m_pDAESolver)
            daeDeclareAndThrowException(exInvalidPointer);
        return m_pDAESolver->GetRelativeTolerance();
    }

    boost::python::list GetAbsoluteTolerances(void) const
    {
        boost::shared_ptr<daeDataProxy_t> pDataProxy = m_pModel->GetDataProxy();
        size_t  nVars    = pDataProxy->GetTotalNumberOfVariables();
        real_t* pAbsTols = m_pModel->GetDataProxy()->GetAbsoluteTolerancesPointer();
        return getListFromVectorByValue(pAbsTols, nVars);
    }

    boost::python::dict GetIndexMappings(void) const
    {
    // Returns dictionary {overallIndex : blockIndex}
        if(!m_ptrBlock)
            daeDeclareAndThrowException(exInvalidPointer);

        boost::python::dict d;
        typedef std::map<size_t, size_t>::iterator c_iterator;

        std::map<size_t, size_t>& mapIndexes = dynamic_cast<daeBlock*>(m_ptrBlock)->m_mapVariableIndexes;

        for(c_iterator iter = mapIndexes.begin(); iter != mapIndexes.end(); iter++)
            d[iter->first] = iter->second;

        return d;
    }

    boost::python::object GetDataReporter_(void)
    {
        return datareporter;
    }

    boost::python::object GetLog_(void)
    {
        return log;
    }

    boost::python::object GetDAESolver_(void)
    {
        return daesolver;
    }

    void SetUpParametersAndDomains(void)
    {
        if(boost::python::override f = this->get_override("SetUpParametersAndDomains"))
            f();
        else
            this->daeSimulation::SetUpParametersAndDomains();
    }
    void def_SetUpParametersAndDomains(void)
    {
        this->daeSimulation::SetUpParametersAndDomains();
    }

    void SetUpVariables(void)
    {
        if(boost::python::override f = this->get_override("SetUpVariables"))
            f();
        else
            this->daeSimulation::SetUpVariables();
    }
    void def_SetUpVariables(void)
    {
        this->daeSimulation::SetUpVariables();
    }

    void DoDataPartitioning(daeEquationsIndexes& equationsOverallIndexes, std::map<size_t,size_t>& mapOverallBlockIndexes)
    {
        if(boost::python::override f = this->get_override("DoDataPartitioning"))
            f(boost::ref(equationsOverallIndexes), boost::ref(mapOverallBlockIndexes));
        else
            this->daeSimulation::DoDataPartitioning(equationsOverallIndexes, mapOverallBlockIndexes);
    }
    void def_DoDataPartitioning(daeEquationsIndexes& equationsOverallIndexes, std::map<size_t,size_t>& mapOverallBlockIndexes)
    {
        this->daeSimulation::DoDataPartitioning(equationsOverallIndexes, mapOverallBlockIndexes);
    }

    void DoPostProcessing()
    {
        if(boost::python::override f = this->get_override("DoPostProcessing"))
            f();
        else
            this->daeSimulation::DoPostProcessing();
    }
    void def_DoPostProcessing()
    {
        this->daeSimulation::DoPostProcessing();
    }

    void CleanUpSetupData(void)
    {
        if(boost::python::override f = this->get_override("CleanUpSetupData"))
            f();
        else
            this->daeSimulation::CleanUpSetupData();
    }
    void def_CleanUpSetupData(void)
    {
        this->daeSimulation::CleanUpSetupData();
    }

    void Run(void)
    {
        if(boost::python::override f = this->get_override("Run"))
            f();
        else
           return daeSimulation::Run();
    }
    void def_Run(void)
    {
        this->daeSimulation::Run();
    }

    void SetUpOptimization(void)
    {
        if(boost::python::override f = this->get_override("SetUpOptimization"))
            f();
        else
            this->daeSimulation::SetUpOptimization();
    }
    void def_SetUpOptimization(void)
    {
        this->daeSimulation::SetUpOptimization();
    }

    void SetUpParameterEstimation(void)
    {
        if(boost::python::override f = this->get_override("SetUpParameterEstimation"))
            f();
        else
            this->daeSimulation::SetUpParameterEstimation();
    }
    void def_SetUpParameterEstimation(void)
    {
        this->daeSimulation::SetUpParameterEstimation();
    }

    void SetUpSensitivityAnalysis(void)
    {
        if(boost::python::override f = this->get_override("SetUpSensitivityAnalysis"))
            f();
        else
            this->daeSimulation::SetUpSensitivityAnalysis();
    }
    void def_SetUpSensitivityAnalysis(void)
    {
        this->daeSimulation::SetUpSensitivityAnalysis();
    }

    daeOptimizationVariable* SetSensitivityParameter1(daeVariable& variable)
    {
        return this->daeSimulation::SetContinuousOptimizationVariable(variable, 0, 1, 1);
    }

    daeOptimizationVariable* SetSensitivityParameter2(adouble a)
    {
        return this->daeSimulation::SetContinuousOptimizationVariable(a, 0, 1, 1);
    }

    daeOptimizationVariable* SetContinuousOptimizationVariable11(daeVariable& variable, real_t LB, real_t UB, real_t defaultValue)
    {
        return this->daeSimulation::SetContinuousOptimizationVariable(variable, LB, UB, defaultValue);
    }

    daeOptimizationVariable* SetContinuousOptimizationVariable12(daeVariable& variable, quantity qLB, quantity qUB, quantity qdefaultValue)
    {
        return this->daeSimulation::SetContinuousOptimizationVariable(variable, qLB, qUB, qdefaultValue);
    }

    daeOptimizationVariable* SetBinaryOptimizationVariable1(daeVariable& variable, bool defaultValue)
    {
        return this->daeSimulation::SetBinaryOptimizationVariable(variable, defaultValue);
    }

    daeOptimizationVariable* SetIntegerOptimizationVariable1(daeVariable& variable, int LB, int UB, int defaultValue)
    {
        return this->daeSimulation::SetIntegerOptimizationVariable(variable, LB, UB, defaultValue);
    }

    daeOptimizationVariable* SetContinuousOptimizationVariable21(adouble a, real_t LB, real_t UB, real_t defaultValue)
    {
        return this->daeSimulation::SetContinuousOptimizationVariable(a, LB, UB, defaultValue);
    }

    daeOptimizationVariable* SetContinuousOptimizationVariable22(adouble a, real_t qLB, real_t qUB, real_t qdefaultValue)
    {
        return this->daeSimulation::SetContinuousOptimizationVariable(a, qLB, qUB, qdefaultValue);
    }

    daeOptimizationVariable* SetBinaryOptimizationVariable2(adouble a, bool defaultValue)
    {
        return this->daeSimulation::SetBinaryOptimizationVariable(a, defaultValue);
    }

    daeOptimizationVariable* SetIntegerOptimizationVariable2(adouble a, int LB, int UB, int defaultValue)
    {
        return this->daeSimulation::SetIntegerOptimizationVariable(a, LB, UB, defaultValue);
    }

    daeMeasuredVariable* SetMeasuredVariable1(daeVariable& variable)
    {
        return this->daeSimulation::SetMeasuredVariable(variable);
    }

    daeVariableWrapper* SetInputVariable1(daeVariable& variable)
    {
        return this->daeSimulation::SetInputVariable(variable);
    }

    daeOptimizationVariable* SetModelParameter1(daeVariable& variable, real_t LB, real_t UB, real_t defaultValue)
    {
        return this->daeSimulation::SetModelParameter(variable, LB, UB, defaultValue);
    }

    daeMeasuredVariable* SetMeasuredVariable2(adouble a)
    {
        return this->daeSimulation::SetMeasuredVariable(a);
    }

    daeVariableWrapper* SetInputVariable2(adouble a)
    {
        return this->daeSimulation::SetInputVariable(a);
    }

    daeOptimizationVariable* SetModelParameter2(adouble a, real_t LB, real_t UB, real_t defaultValue)
    {
        return this->daeSimulation::SetModelParameter(a, LB, UB, defaultValue);
    }

    boost::python::object GetObjectiveFunction_(void)
    {
        if(m_arrObjectiveFunctions.empty())
        {
            daeDeclareException(exInvalidCall);
            e << "No objective functions are defined (did you forget to enable calculateSensitivities flag in daeSimulation.Initialize() "
              << "function, implement the daeSimulation.SetUpSensitivityAnalysis function and set the number of objective functions "
              << " using the daeSimulation.SetNumberOfObjectiveFunctions() function?)";
            throw e;
        }

        daeObjectiveFunction* pFobj = m_arrObjectiveFunctions[0].get();
        return boost::python::object(boost::ref(pFobj));
    }

    boost::python::list GetOptimizationVariables(void)
    {
        daeOptimizationVariable* obj;
        boost::python::list l;

        for(size_t i = 0; i < m_arrOptimizationVariables.size(); i++)
        {
            obj = m_arrOptimizationVariables[i].get();
            l.append(boost::ref(obj));
        }
        return l;
    }

    boost::python::list GetConstraints(void)
    {
        daeOptimizationConstraint* obj;
        boost::python::list l;

        for(size_t i = 0; i < m_arrConstraints.size(); i++)
        {
            obj = m_arrConstraints[i].get();
            l.append(boost::ref(obj));
        }
        return l;
    }

    boost::python::list GetObjectiveFunctions(void)
    {
        daeObjectiveFunction* obj;
        boost::python::list l;

        for(size_t i = 0; i < m_arrObjectiveFunctions.size(); i++)
        {
            obj = m_arrObjectiveFunctions[i].get();
            l.append(boost::ref(obj));
        }
        return l;
    }

    boost::python::list GetInputVariables(void)
    {
        daeVariableWrapper* obj;
        boost::python::list l;

        for(size_t i = 0; i < m_arrInputVariables.size(); i++)
        {
            obj = m_arrInputVariables[i].get();
            l.append(boost::ref(obj));
        }
        return l;
    }

    boost::python::list GetMeasuredVariables(void)
    {
        daeMeasuredVariable* obj;
        boost::python::list l;

        for(size_t i = 0; i < m_arrMeasuredVariables.size(); i++)
        {
            obj = m_arrMeasuredVariables[i].get();
            l.append(boost::ref(obj));
        }
        return l;
    }

    boost::python::list GetModelParameters(void)
    {
        daeOptimizationVariable* obj;
        boost::python::list l;

        for(size_t i = 0; i < m_arrOptimizationVariables.size(); i++)
        {
            obj = m_arrOptimizationVariables[i].get();
            l.append(boost::ref(obj));
        }
        return l;
    }


    boost::python::list lGetActiveEquationSetMemory()
    {
        std::vector<size_t> sizes = daeSimulation::GetActiveEquationSetMemory();
        boost::python::list lSizes;

        for(size_t i = 0; i < sizes.size(); i++)
        {
            lSizes.append(sizes[i]);
        }
        return lSizes;
    }

    boost::python::dict dGetActiveEquationSetNodeCount()
    {
        std::map<std::string, size_t> mapCount = daeSimulation::GetActiveEquationSetNodeCount();
        boost::python::dict dCount;

        for(std::map<std::string, size_t>::iterator it = mapCount.begin(); it != mapCount.end(); it++)
        {
            dCount[it->first] = it->second;
        }
        return dCount;
    }

    boost::python::list GetReportingTimes(void) const
    {
        boost::python::list l;

        for(size_t i = 0; i < m_darrReportingTimes.size(); i++)
            l.append(m_darrReportingTimes[i]);
        return l;
    }

    void SetReportingTimes(boost::python::list l)
    {
        real_t value;
        std::vector<real_t> darrReportingTimes;
        boost::python::ssize_t n = boost::python::len(l);
        for(boost::python::ssize_t i = 0; i < n; i++)
        {
            value = boost::python::extract<real_t>(l[i]);
            darrReportingTimes.push_back(value);
        }

        daeSimulation::SetReportingTimes(darrReportingTimes);
    }

    void dExportComputeStackStructs(const std::string& filenameComputeStacks,
                                    const std::string& filenameJacobianIndexes,
                                    int startEquationIndex = 0,
                                    int endEquationIndex = -1,
                                    boost::python::dict dict_bi_to_bi_local = boost::python::dict())
   {
        std::map<int,int> bi_to_bi_local = getMapFromDict<int,int>(dict_bi_to_bi_local);
        ExportComputeStackStructs(filenameComputeStacks, filenameJacobianIndexes, startEquationIndex, endEquationIndex, bi_to_bi_local);
   }

    void dExportComputeStackStructs1(const std::string& filenameComputeStacks,
                                     const std::string& filenameJacobianIndexes,
                                     boost::python::list list_equationIndexes,
                                     boost::python::dict dict_bi_to_bi_local)
   {
        std::map<uint32_t,uint32_t> bi_to_bi_local = getMapFromDict<uint32_t,uint32_t>(dict_bi_to_bi_local);
        std::vector<uint32_t>  equationIndexes     = getVectorFromList<uint32_t>(list_equationIndexes);
        ExportComputeStackStructs(filenameComputeStacks, filenameJacobianIndexes, equationIndexes, bi_to_bi_local);
   }

public:
    boost::python::object model;
    boost::python::object daesolver;
    boost::python::object log;
    boost::python::object datareporter;
};



class daeOptimization_tWrapper : public daeOptimization_t,
                                 public boost::python::wrapper<daeOptimization_t>
{
public:
    void Run(void)
    {
        this->get_override("Run")();
    }

    void Finalize(void)
    {
        this->get_override("Finalize")();
    }

    void StartIterationRun(int iteration)
    {
        this->get_override("StartIterationRun")(iteration);
    }

    void EndIterationRun(int iteration)
    {
        this->get_override("EndIterationRun")(iteration);
    }
};

class daeOptimizationWrapper : public daeOptimization,
                               public boost::python::wrapper<daeOptimization>
{
public:
    void Initialize(boost::python::object Simulation,
                    boost::python::object NLPSolver,
                    boost::python::object DAESolver,
                    boost::python::object DataReporter,
                    boost::python::object Log,
                    const std::string&    initializationFile)
     {
        daeDefaultSimulationWrapper* pSimulation   = boost::python::extract<daeDefaultSimulationWrapper*>(Simulation);
        daeDAESolver_t*              pDAESolver    = boost::python::extract<daeDAESolver_t*>(DAESolver);
        daeDataReporter_t*           pDataReporter = boost::python::extract<daeDataReporter_t*>(DataReporter);
        daeLog_t*                    pLog          = boost::python::extract<daeLog_t*>(Log);
        daeNLPSolver_t*              pNLPSolver    = boost::python::extract<daeNLPSolver_t*>(NLPSolver);

        // Keep the python objects so they do not get destroyed if the go out of scope (in python code)
        if(!pSimulation)
            daeDeclareAndThrowException(exInvalidPointer);

        pSimulation->daesolver    = DAESolver;
        pSimulation->datareporter = DataReporter;
        pSimulation->log          = Log;

        simulation = Simulation;
        nlpsolver  = NLPSolver;

        daeOptimization::Initialize(pSimulation, pNLPSolver, pDAESolver, pDataReporter, pLog, initializationFile);
     }

    void StartIterationRun(int iteration)
    {
        if(boost::python::override f = this->get_override("StartIterationRun"))
            f(iteration);
        else
            this->daeOptimization::StartIterationRun(iteration);
    }
    void def_StartIterationRun(int iteration)
    {
        this->daeOptimization::StartIterationRun(iteration);
    }

    void EndIterationRun(int iteration)
    {
        if(boost::python::override f = this->get_override("EndIterationRun"))
            f(iteration);
        else
            this->daeOptimization::EndIterationRun(iteration);
    }
    void def_EndIterationRun(int iteration)
    {
        this->daeOptimization::EndIterationRun(iteration);
    }

    boost::python::object GetSimulation_(void) const
    {
        return simulation;
    }

public:
    boost::python::object simulation;
    boost::python::object nlpsolver;
};

}

#endif
