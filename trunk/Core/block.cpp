#include "stdafx.h"
#include "coreimpl.h"
#include "../IDAS_DAESolver/dae_array_matrix.h"
#include <omp.h>
#include "compute_stack_kernels_openmp.h"
using namespace dae::solver;

namespace dae
{
namespace core
{

daeBlock::daeBlock(void)
{
    m_bInitializeMode					= false;
    m_pDataProxy						= NULL;
    m_parrTimeDerivatives               = NULL;
    m_parrValues					    = NULL;
    m_parrResidual						= NULL;
    m_pmatJacobian						= NULL;
    m_dCurrentTime						= 0;
    m_dInverseTimeStep					= 0;
    m_nNumberOfEquations                = 0;
    m_nTotalNumberOfVariables           = 0;
    m_nCurrentVariableIndexForJacobianEvaluation = ULONG_MAX;

    m_nNuberOfResidualsCalls            = 0;
    m_nNuberOfJacobianCalls             = 0;
    m_nNuberOfSensitivityResidualsCalls = 0;
    m_dTotalTimeForResiduals            = 0;
    m_dTotalTimeForJacobian             = 0;
    m_dTotalTimeForSensitivityResiduals = 0;

    m_omp_num_threads = 0;
    m_computeStackEvaluator = NULL; //CreateComputeStackEvaluator();

/*
    daeConfig& cfg = daeGetConfig();
    m_omp_schedule           = cfg.GetString ("daetools.core.equations.schedule",         "default");
    m_omp_shedule_chunk_size = cfg.GetInteger("daetools.core.equations.scheduleChunkSize", 0);
    // If the schedule in the config file is 'default' then it is left to the implementation default
    if(m_omp_schedule == "static")
        omp_set_schedule(omp_sched_static, m_omp_shedule_chunk_size);
    else if(m_omp_schedule == "dynamic")
        omp_set_schedule(omp_sched_dynamic, m_omp_shedule_chunk_size);
    else if(m_omp_schedule == "guided")
        omp_set_schedule(omp_sched_guided, m_omp_shedule_chunk_size);
    else if(m_omp_schedule == "auto")
        omp_set_schedule(omp_sched_auto, m_omp_shedule_chunk_size);
*/

#if defined(DAE_MPI)
    m_nEquationIndexesStart = ULONG_MAX;
    m_nEquationIndexesEnd   = ULONG_MAX;
    m_nVariableIndexesStart = ULONG_MAX;
    m_nVariableIndexesEnd   = ULONG_MAX;
#endif
}

daeBlock::~daeBlock(void)
{
    // External library should delete the object.
    m_computeStackEvaluator = NULL;
}

void daeBlock::Open(io::xmlTag_t* pTag)
{
    io::daeSerializable::Open(pTag);
}

void daeBlock::Save(io::xmlTag_t* pTag) const
{
    io::daeSerializable::Save(pTag);
}

void daeBlock::SetComputeStackEvaluator(csComputeStackEvaluator_t* computeStackEvaluator)
{
    if(computeStackEvaluator)
    {
        m_pDataProxy->SetEvaluationMode(eComputeStack_External);
        m_computeStackEvaluator = computeStackEvaluator;
    }
}

csComputeStackEvaluator_t* daeBlock::GetComputeStackEvaluator()
{
    return m_computeStackEvaluator;
}

void daeBlock::CalculateConditions(real_t				dTime,
                                   daeArray<real_t>&	arrValues,
                                   daeArray<real_t>&	arrTimeDerivatives,
                                   daeArray<real_t>&	arrResults)
{
    map<size_t, daeExpressionInfo>::iterator iter;

    if(m_pDataProxy->PrintInfo())
        m_pDataProxy->LogMessage(string("Evaluate conditions at time ") + toStringFormatted(dTime, -1, 15) + string("..."), 0);

    SetTime(dTime);
    m_pDataProxy->SetCurrentTime(dTime);
    SetValuesArray(&arrValues);
    SetTimeDerivativesArray(&arrTimeDerivatives);

    daeExecutionContext EC;
    EC.m_pBlock						= this;
    EC.m_pDataProxy					= m_pDataProxy;
    EC.m_eEquationCalculationMode	= eCalculate;

    size_t nFnCounter = 0;
    for(iter = m_mapExpressionInfos.begin(); iter != m_mapExpressionInfos.end(); iter++)
    {
        arrResults.SetItem(nFnCounter, (*iter).second.m_pExpression->Evaluate(&EC).getValue());
        nFnCounter++;
    }
}

std::vector<size_t> daeBlock::GetActiveEquationSetMemory() const
{
    std::vector<size_t> sizes;
    // Initialise with zeroes
    sizes.resize(m_ptrarrEquationExecutionInfos_ActiveSet.size(), 0);
    for(int i = 0; i < m_ptrarrEquationExecutionInfos_ActiveSet.size(); i++)
    {
        daeEquationExecutionInfo* pEquationExecutionInfo = m_ptrarrEquationExecutionInfos_ActiveSet[i];
        if(pEquationExecutionInfo->m_EquationEvaluationNode)
            sizes[i] = pEquationExecutionInfo->m_EquationEvaluationNode->SizeOf();
    }
    return sizes;
}

std::map<std::string, size_t> daeBlock::GetActiveEquationSetNodeCount() const
{
    std::map<std::string, size_t> mapCount;
    for(int i = 0; i < m_ptrarrEquationExecutionInfos_ActiveSet.size(); i++)
    {
        daeEquationExecutionInfo* pEquationExecutionInfo = m_ptrarrEquationExecutionInfos_ActiveSet[i];
        if(pEquationExecutionInfo->m_EquationEvaluationNode)
            adNode::GetNodeCount(pEquationExecutionInfo->m_EquationEvaluationNode.get(), mapCount);
    }
    return mapCount;
}

void daeBlock::CalculateResiduals(real_t			dTime,
                                  daeArray<real_t>& arrValues,
                                  daeArray<real_t>& arrResiduals,
                                  daeArray<real_t>& arrTimeDerivatives)
{
    call_stats::TimerCounter tc(m_stats["Residuals"]);

    m_nNuberOfResidualsCalls++;
    double startTime = omp_get_wtime();

    if(m_pDataProxy->PrintInfo())
        m_pDataProxy->LogMessage(string("Calculate residuals at time ") + toStringFormatted(dTime, -1, 15) + string("..."), 0);

    SetTime(dTime);
    m_pDataProxy->SetCurrentTime(dTime);
    SetValuesArray(&arrValues);
    SetTimeDerivativesArray(&arrTimeDerivatives);
    SetResidualArray(&arrResiduals);
    SetInverseTimeStep(0.0);

    // Update equations if necessary (in general, applicable only to FE equations)
    daeModel* pTopLevelModel = dynamic_cast<daeModel*>(m_pDataProxy->GetTopLevelModel());
    pTopLevelModel->UpdateEquations();

    // Calls PyEval_InitThreads and PyEval_SaveThread in the constructor, and PyEval_RestoreThread in the destructor
    boost::shared_ptr<daeAllowThreads_t> _allowThreads_ = pTopLevelModel->CreateAllowThreads();

    if(m_pDataProxy->GetEvaluationMode() == eComputeStack_OpenMP ||
       m_pDataProxy->GetEvaluationMode() == eComputeStack_External)
    {
        const std::vector<real_t>& arrDOFs = m_pDataProxy->GetAssignedVarsValues();

        csComputeStackItem_t* computeStacks            = &m_arrAllComputeStacks[0];
        uint32_t*             activeEquationSetIndexes = &m_arrActiveEquationSetIndexes[0];

        real_t* dofs            = (arrDOFs.size() > 0 ? const_cast<real_t*>(&arrDOFs[0]) : NULL);
        real_t* values          = arrValues.Data();
        real_t* timeDerivatives = arrTimeDerivatives.Data();
        real_t* residuals       = arrResiduals.Data();

        cs::csEvaluationContext_t EC;
        EC.equationEvaluationMode      = cs::eEvaluateEquation;
        EC.sensitivityParameterIndex   = -1;
        EC.jacobianIndex               = -1;
        EC.numberOfVariables           = m_nNumberOfEquations;
        EC.numberOfEquations           = m_nNumberOfEquations; // Always total number (multi-device evaluators will adjust it if required)
        EC.numberOfDOFs                = arrDOFs.size();
        EC.numberOfComputeStackItems   = m_arrAllComputeStacks.size();
        EC.numberOfIncidenceMatrixItems= 0;
        EC.valuesStackSize             = 5;
        EC.lvaluesStackSize            = 20;
        EC.rvaluesStackSize            = 5;
        EC.currentTime                 = dTime;
        EC.inverseTimeStep             = 0; // Should not be needed here. Double check...
        EC.startEquationIndex          = 0; // Always 0 (multi-device evaluators will adjust it if required)
        EC.startJacobianIndex          = 0; // Always 0 (multi-device evaluators will adjust it if required)

        if(m_pDataProxy->GetEvaluationMode() == eComputeStack_OpenMP)
        {
            // Sequential run can be achieved by setting numThreads to 1.
            if(m_omp_num_threads > 0)
                omp_set_num_threads(m_omp_num_threads);

            #pragma omp parallel for firstprivate(EC)
            for(int ei = 0; ei < m_nNumberOfEquations; ei++)
            {
                openmp_evaluator::EvaluateEquations(computeStacks,
                                                    ei,
                                                    activeEquationSetIndexes,
                                                    EC,
                                                    dofs,
                                                    values,
                                                    timeDerivatives,
                                                    residuals);
            }
        }
        else if(m_pDataProxy->GetEvaluationMode() == eComputeStack_External)
        {
            if(!m_computeStackEvaluator)
                daeDeclareAndThrowException(exInvalidPointer);

            m_computeStackEvaluator->EvaluateEquations(EC,
                                                       dofs,
                                                       values,
                                                       timeDerivatives,
                                                       residuals);
        }
    }
    else if(m_pDataProxy->GetEvaluationMode() == eEvaluationTree_OpenMP)
    {
        daeExecutionContext EC;
        EC.m_pBlock						= this;
        EC.m_pDataProxy					= m_pDataProxy;
        EC.m_dInverseTimeStep			= GetInverseTimeStep();
        EC.m_pEquationExecutionInfo		= NULL;
        EC.m_eEquationCalculationMode	= eCalculate;

        if(m_ptrarrEquationExecutionInfos_ActiveSet.empty())
            daeDeclareAndThrowException(exInvalidCall);

        // m_ptrarrEquationExecutionInfos_ActiveSet should be previously updated with the currently active equation set.
        // Sequential run can be achieved by setting numThreads to 1.
        if(m_omp_num_threads > 0)
            omp_set_num_threads(m_omp_num_threads);

        #pragma omp parallel for firstprivate(EC)
        for(int i = 0; i < m_ptrarrEquationExecutionInfos_ActiveSet.size(); i++)
        {
            daeEquationExecutionInfo* pEquationExecutionInfo = m_ptrarrEquationExecutionInfos_ActiveSet[i];
            pEquationExecutionInfo->Residual(EC);
        }
    }
    else
    {
        daeDeclareAndThrowException(exNotImplemented);
    }

    double endTime = omp_get_wtime();
    m_dTotalTimeForResiduals += (endTime - startTime);
}

static void setJacobianMatrixItem(void* matrix, uint32_t row, uint32_t col, real_t value)
{
    daeMatrix<real_t>* mat = (daeMatrix<real_t>*)matrix;
    if(!mat)
        daeDeclareAndThrowException(exInvalidCall);
    mat->SetItem(row, col, value);
}

void daeBlock::CalculateJacobian(real_t				dTime,
                                 daeArray<real_t>&	arrValues,
                                 daeArray<real_t>&	arrResiduals,
                                 daeArray<real_t>&	arrTimeDerivatives,
                                 daeMatrix<real_t>&	matJacobian,
                                 real_t				dInverseTimeStep)
{
    call_stats::TimerCounter tc(m_stats["Jacobian"]);

    m_nNuberOfJacobianCalls++;
    double startTime = omp_get_wtime();

    if(m_pDataProxy->PrintInfo())
        m_pDataProxy->LogMessage(string("Calculate Jacobian at time ") + toStringFormatted(dTime, -1, 15) + string("..."), 0);

    SetTime(dTime);
    m_pDataProxy->SetCurrentTime(dTime);
    SetValuesArray(&arrValues);
    SetTimeDerivativesArray(&arrTimeDerivatives);
    SetResidualArray(&arrResiduals);
    SetJacobianMatrix(&matJacobian);
    SetInverseTimeStep(dInverseTimeStep);

    // Update equations if necessary (in general, applicable only to FE equations)
    daeModel* pTopLevelModel = dynamic_cast<daeModel*>(m_pDataProxy->GetTopLevelModel());
    pTopLevelModel->UpdateEquations();

    // Calls PyEval_InitThreads and PyEval_SaveThread in the constructor, and PyEval_RestoreThread in the destructor
    boost::shared_ptr<daeAllowThreads_t> _allowThreads_ = pTopLevelModel->CreateAllowThreads();

    if(m_pDataProxy->GetEvaluationMode() == eComputeStack_OpenMP ||
       m_pDataProxy->GetEvaluationMode() == eComputeStack_External)
    {
        const std::vector<real_t>& arrDOFs = m_pDataProxy->GetAssignedVarsValues();

        csComputeStackItem_t*    computeStacks             = &m_arrAllComputeStacks[0];
        uint32_t*                activeEquationSetIndexes  = &m_arrActiveEquationSetIndexes[0];
        csIncidenceMatrixItem_t* computeStackJacobianItems = &m_arrComputeStackJacobianItems[0];
        uint32_t                 noJacobianItems           = m_arrComputeStackJacobianItems.size();

        real_t* dofs            = (arrDOFs.size() > 0 ? const_cast<real_t*>(&arrDOFs[0]) : NULL);
        real_t* values          = arrValues.Data();
        real_t* timeDerivatives = arrTimeDerivatives.Data();

        cs::csEvaluationContext_t EC;
        EC.equationEvaluationMode      = cs::eEvaluateDerivative; // Jacobian
        EC.sensitivityParameterIndex   = -1;
        EC.jacobianIndex               = -1;
        EC.numberOfVariables           = m_nNumberOfEquations;
        EC.numberOfEquations           = m_nNumberOfEquations; // Always total number (multi-device evaluators will adjust it if required)
        EC.numberOfDOFs                = arrDOFs.size();
        EC.numberOfComputeStackItems   = m_arrAllComputeStacks.size();
        EC.numberOfIncidenceMatrixItems= noJacobianItems;
        EC.valuesStackSize             = 5;
        EC.lvaluesStackSize            = 20;
        EC.rvaluesStackSize            = 5;
        EC.currentTime                 = dTime;
        EC.inverseTimeStep             = dInverseTimeStep;
        EC.startEquationIndex          = 0; // Always 0 (multi-device evaluators will adjust it if required)
        EC.startJacobianIndex          = 0; // Always 0 (multi-device evaluators will adjust it if required)

        if(m_pDataProxy->GetEvaluationMode() == eComputeStack_OpenMP)
        {
            m_jacobian.resize(noJacobianItems);

            // Sequential run can be achieved by setting numThreads to 1.
            if(m_omp_num_threads > 0)
                omp_set_num_threads(m_omp_num_threads);

            #pragma omp parallel for firstprivate(EC)
            for(int ji = 0; ji < noJacobianItems; ji++)
            {
                openmp_evaluator::EvaluateDerivatives(computeStacks,
                                                      ji,
                                                      activeEquationSetIndexes,
                                                      computeStackJacobianItems,
                                                      EC,
                                                      dofs,
                                                      values,
                                                      timeDerivatives,
                                                      m_jacobian.data());
            }

            // Evaluated Jacobian values need to be copied to the Jacobian matrix.
            for(size_t ji = 0; ji < noJacobianItems; ji++)
            {
                const csIncidenceMatrixItem_t& jacobianItem = m_arrComputeStackJacobianItems[ji];
                matJacobian.SetItem(jacobianItem.equationIndex, jacobianItem.blockIndex, m_jacobian[ji]);
            }
        }
        else if(m_pDataProxy->GetEvaluationMode() == eComputeStack_External)
        {
            if(!m_computeStackEvaluator)
                daeDeclareAndThrowException(exInvalidPointer);

            m_jacobian.resize(noJacobianItems);

            m_computeStackEvaluator->EvaluateDerivatives(EC,
                                                         dofs,
                                                         values,
                                                         timeDerivatives,
                                                         m_jacobian.data());

            // Evaluated Jacobian values need to be copied to the Jacobian matrix.
            for(size_t ji = 0; ji < noJacobianItems; ji++)
            {
                const csIncidenceMatrixItem_t& jacobianItem = m_arrComputeStackJacobianItems[ji];
                matJacobian.SetItem(jacobianItem.equationIndex, jacobianItem.blockIndex, m_jacobian[ji]);
            }
        }
    }
    else if(m_pDataProxy->GetEvaluationMode() == eEvaluationTree_OpenMP)
    {
        daeExecutionContext EC;
        EC.m_pBlock						= this;
        EC.m_pDataProxy					= m_pDataProxy;
        EC.m_dInverseTimeStep			= GetInverseTimeStep();
        EC.m_pEquationExecutionInfo		= NULL;
        EC.m_eEquationCalculationMode	= eCalculateJacobian;

        if(m_ptrarrEquationExecutionInfos_ActiveSet.empty())
            daeDeclareAndThrowException(exInvalidCall);

        // m_ptrarrEquationExecutionInfos_ActiveSet should be previously updated with the currently active equation set.
        // Sequential run can be achieved by setting numThreads to 1.
        if(m_omp_num_threads > 0)
            omp_set_num_threads(m_omp_num_threads);

        #pragma omp parallel for firstprivate(EC)
        for(int i = 0; i < m_ptrarrEquationExecutionInfos_ActiveSet.size(); i++)
        {
            daeEquationExecutionInfo* pEquationExecutionInfo = m_ptrarrEquationExecutionInfos_ActiveSet[i];
            pEquationExecutionInfo->Jacobian(EC);
        }
    }
    else
    {
        daeDeclareAndThrowException(exNotImplemented);
    }

    double endTime = omp_get_wtime();
    m_dTotalTimeForJacobian += (endTime - startTime);
}

// For dynamic models
void daeBlock::CalculateSensitivityResiduals(real_t						dTime,
                                             const std::vector<size_t>& narrParameterIndexes,
                                             daeArray<real_t>&			arrValues,
                                             daeArray<real_t>&			arrTimeDerivatives,
                                             daeMatrix<real_t>&			matSValues,
                                             daeMatrix<real_t>&			matSTimeDerivatives,
                                             daeMatrix<real_t>&			matSResiduals)
{
    call_stats::TimerCounter tc(m_stats["SensitivityResiduals"]);

    m_nNuberOfSensitivityResidualsCalls++;
    double startTime = omp_get_wtime();

    if(m_pDataProxy->PrintInfo())
        m_pDataProxy->LogMessage(string("Calculate sensitivity residuals at time ") + toStringFormatted(dTime, -1, 15) + string("..."), 0);

    SetTime(dTime);
    m_pDataProxy->SetCurrentTime(dTime);
    SetValuesArray(&arrValues);
    SetTimeDerivativesArray(&arrTimeDerivatives);
    SetInverseTimeStep(0.0);
    m_pDataProxy->SetSensitivityMatrixes(&matSValues,
                                         &matSTimeDerivatives,
                                         &matSResiduals);

    // Update equations if necessary (in general, applicable only to FE equations)
    daeModel* pTopLevelModel = dynamic_cast<daeModel*>(m_pDataProxy->GetTopLevelModel());
    pTopLevelModel->UpdateEquations();

    // Calls PyEval_InitThreads and PyEval_SaveThread in the constructor, and PyEval_RestoreThread in the destructor
    boost::shared_ptr<daeAllowThreads_t> _allowThreads_ = pTopLevelModel->CreateAllowThreads();

    if(m_pDataProxy->GetEvaluationMode() == eComputeStack_OpenMP ||
       m_pDataProxy->GetEvaluationMode() == eComputeStack_External)
    {
        const std::vector<real_t>& arrDOFs = m_pDataProxy->GetAssignedVarsValues();

        csComputeStackItem_t* computeStacks            = &m_arrAllComputeStacks[0];
        uint32_t*             activeEquationSetIndexes = &m_arrActiveEquationSetIndexes[0];

        real_t* dofs            = (arrDOFs.size() > 0 ? const_cast<real_t*>(&arrDOFs[0]) : NULL);
        real_t* values          = arrValues.Data();
        real_t* timeDerivatives = arrTimeDerivatives.Data();

        daeDenseMatrix* sens_res_mat = dynamic_cast< daeDenseMatrix*>(&matSResiduals);
        if(!sens_res_mat)
            daeDeclareAndThrowException(exInvalidPointer);

        cs::csEvaluationContext_t EC;
        EC.equationEvaluationMode      = cs::eEvaluateSensitivityDerivative;
        EC.sensitivityParameterIndex   = -1;
        EC.jacobianIndex               = -1;
        EC.numberOfVariables           = m_nNumberOfEquations;
        EC.numberOfEquations           = m_nNumberOfEquations; // Always total number (multi-device evaluators will adjust it if required)
        EC.numberOfDOFs                = arrDOFs.size();
        EC.numberOfComputeStackItems   = m_arrAllComputeStacks.size();
        EC.numberOfIncidenceMatrixItems= 0;
        EC.valuesStackSize             = 5;
        EC.lvaluesStackSize            = 20;
        EC.rvaluesStackSize            = 5;
        EC.currentTime                 = dTime;
        EC.inverseTimeStep             = 0; // Should not be needed here. Double check...
        EC.startEquationIndex          = 0; // Always 0 (multi-device evaluators will adjust it if required)
        EC.startJacobianIndex          = 0; // Always 0 (multi-device evaluators will adjust it if required)

        for(size_t p = 0; p < narrParameterIndexes.size(); p++)
        {
            real_t* svalues    = matSValues.GetRow(p);
            real_t* sdvalues   = matSTimeDerivatives.GetRow(p);
            real_t* sresiduals = matSResiduals.GetRow(p);

            EC.sensitivityParameterIndex = narrParameterIndexes[p];

            if(m_pDataProxy->GetEvaluationMode() == eComputeStack_OpenMP)
            {
                // Sequential run can be achieved by setting numThreads to 1.
                if(m_omp_num_threads > 0)
                    omp_set_num_threads(m_omp_num_threads);

                #pragma omp parallel for firstprivate(EC)
                for(int ei = 0; ei < m_nNumberOfEquations; ei++)
                {
                    openmp_evaluator::EvaluateSensitivityDerivatives(computeStacks,
                                                                     ei,
                                                                     activeEquationSetIndexes,
                                                                     EC,
                                                                     dofs,
                                                                     values,
                                                                     timeDerivatives,
                                                                     svalues,
                                                                     sdvalues,
                                                                     sresiduals);
                }
            }
            else if(m_pDataProxy->GetEvaluationMode() == eComputeStack_External)
            {
                if(!m_computeStackEvaluator)
                    daeDeclareAndThrowException(exInvalidPointer);

                m_computeStackEvaluator->EvaluateSensitivityDerivatives(EC,
                                                                        dofs,
                                                                        values,
                                                                        timeDerivatives,
                                                                        svalues,
                                                                        sdvalues,
                                                                        sresiduals);
            }
        }
    }
    else if(m_pDataProxy->GetEvaluationMode() == eEvaluationTree_OpenMP)
    {
        daeExecutionContext EC;
        EC.m_pBlock						= this;
        EC.m_pDataProxy					= m_pDataProxy;
        EC.m_dInverseTimeStep			= 0; // Should not be needed here. Double check...
        EC.m_pEquationExecutionInfo		= NULL;
        EC.m_eEquationCalculationMode	= eCalculateSensitivityResiduals;

        // m_ptrarrEquationExecutionInfos_ActiveSet should be previously updated with the currently active equation set.
        // Sequential run can be achieved by setting numThreads to 1.
        if(m_omp_num_threads > 0)
            omp_set_num_threads(m_omp_num_threads);

        #pragma omp parallel for firstprivate(EC)
        for(int i = 0; i < m_ptrarrEquationExecutionInfos_ActiveSet.size(); i++)
        {

            daeEquationExecutionInfo* pEquationExecutionInfo = m_ptrarrEquationExecutionInfos_ActiveSet[i];
            pEquationExecutionInfo->SensitivityResiduals(EC, narrParameterIndexes);
        }
    }
    else
    {
        daeDeclareAndThrowException(exNotImplemented);
    }

    m_pDataProxy->ResetSensitivityMatrixes();

    double endTime = omp_get_wtime();
    m_dTotalTimeForSensitivityResiduals += (endTime - startTime);
}

// For steady-state models (not used anymore)
void daeBlock::CalculateSensitivityParametersGradients(const std::vector<size_t>& narrParameterIndexes,
                                                       daeArray<real_t>&		  arrValues,
                                                       daeArray<real_t>&		  arrTimeDerivatives,
                                                       daeMatrix<real_t>&		  matSResiduals)
{
    daeDeclareAndThrowException(exInvalidCall);
/*
    size_t i;
    daeSTN* pSTN;
    daeEquationExecutionInfo* pEquationExecutionInfo;

    if(m_pDataProxy->PrintInfo())
        m_pDataProxy->LogMessage(string("Calculate sensitivity gradients..."), 0);

    SetValuesArray(&arrValues);
    SetTimeDerivativesArray(&arrTimeDerivatives);
    m_pDataProxy->SetSensitivityMatrixes(NULL, NULL, &matSResiduals);

    daeExecutionContext EC;
    EC.m_pBlock						= this;
    EC.m_pDataProxy					= m_pDataProxy;
    EC.m_dInverseTimeStep			= GetInverseTimeStep();
    EC.m_pEquationExecutionInfo		= NULL;
    EC.m_eEquationCalculationMode	= eCalculateSensitivityParametersGradients;

    // Update equations if necessary (in general, applicable only to FE equations)
    dynamic_cast<daeModel*>(m_pDataProxy->GetTopLevelModel())->UpdateEquations(&EC);

    for(i = 0; i < m_ptrarrEquationExecutionInfos.size(); i++)
    {
        pEquationExecutionInfo = m_ptrarrEquationExecutionInfos[i];
        pEquationExecutionInfo->SensitivityParametersGradients(EC, narrParameterIndexes);
    }

// In general, neither objective function nor constraints can be within an STN
    for(i = 0; i < m_ptrarrSTNs.size(); i++)
    {
        pSTN = m_ptrarrSTNs[i];
        pSTN->CalculateSensitivityParametersGradients(EC, narrParameterIndexes);
    }

    m_pDataProxy->ResetSensitivityMatrixes();
*/
}

void daeBlock::CalcNonZeroElements(int& NNZ)
{
    size_t i;
    daeSTN* pSTN;
    daeEquationExecutionInfo* pEquationExecutionInfo;

// First find in normal equations (non-STN)
    for(i = 0; i < m_ptrarrEquationExecutionInfos.size(); i++)
    {
        pEquationExecutionInfo = m_ptrarrEquationExecutionInfos[i];
        NNZ += pEquationExecutionInfo->m_mapIndexes.size();
    }

// Then in STN equations
    for(i = 0; i < m_ptrarrSTNs.size(); i++)
    {
        pSTN = m_ptrarrSTNs[i];
        pSTN->CalcNonZeroElements(NNZ);
    }
}

void daeBlock::FillSparseMatrix(daeSparseMatrix<real_t>* pMatrix)
{
    size_t i;
    daeSTN* pSTN;
    daeEquationExecutionInfo* pEquationExecutionInfo;

    for(i = 0; i < m_ptrarrEquationExecutionInfos.size(); i++)
    {
        pEquationExecutionInfo = m_ptrarrEquationExecutionInfos[i];
        pMatrix->AddRow(pEquationExecutionInfo->m_mapIndexes);
    }

    for(i = 0; i < m_ptrarrSTNs.size(); i++)
    {
        pSTN = m_ptrarrSTNs[i];
        pSTN->FillSparseMatrix(pMatrix);
    }
}

void daeBlock::FillAbsoluteTolerancesInitialConditionsAndInitialGuesses(daeArray<real_t>& arrValues,
                                                                        daeArray<real_t>& arrTimeDerivatives,
                                                                        daeArray<real_t>& arrInitialConditionsTypes,
                                                                        daeArray<real_t>& arrAbsoluteTolerances,
                                                                        daeArray<real_t>& arrValueConstraints)
{
    map<size_t, size_t>::iterator iter;

    if(GetNumberOfEquations() != m_mapVariableIndexes.size())
    {
        daeDeclareException(exInvalidCall);
        e << "Number of equation is not equal to number of variables";
        throw e;
    }
    if(!m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    const real_t* pBlockValues            = m_pDataProxy->GetInitialValuesPointer();
    const real_t* pBlockInitialConditions = m_pDataProxy->GetInitialConditionsPointer();
    // Here I need information which variables are differential
    const real_t* pBlockIDs               = m_pDataProxy->GetVariableTypesGatheredPointer();
    const real_t* pBlockAbsoluteTolerance = m_pDataProxy->GetAbsoluteTolerancesPointer();
    const real_t* pBlockValuesConstraints = m_pDataProxy->GetVariableValuesConstraintsPointer();

    for(iter = m_mapVariableIndexes.begin(); iter != m_mapVariableIndexes.end(); iter++)
    {
        arrValues.SetItem                (iter->second, pBlockValues[iter->first]);
        arrTimeDerivatives.SetItem       (iter->second, pBlockInitialConditions[iter->first]);
        arrInitialConditionsTypes.SetItem(iter->second, pBlockIDs[iter->first]);
        arrAbsoluteTolerances.SetItem    (iter->second, pBlockAbsoluteTolerance[iter->first]);
        arrValueConstraints.SetItem      (iter->second, pBlockValuesConstraints[iter->first]);
    }
}

//void daeBlock::SetAllInitialConditions(real_t value)
//{
//	daeDeclareAndThrowException(exNotImplemented);

//	if(!m_pDataProxy)
//		daeDeclareAndThrowException(exInvalidPointer);

//	size_t n = m_pDataProxy->GetTotalNumberOfVariables();
//	for(size_t i = 0; i < n; i++)
//	{
//		if(m_pDataProxy->GetVariableTypeGathered(i) == cnDifferential)
//		{
//			m_pDataProxy->SetInitialCondition(i, value, m_pDataProxy->GetInitialConditionMode());
//			m_pDataProxy->SetVariableType(i, cnDifferential);
//		}
//	}
//}

size_t daeBlock::FindVariableBlockIndex(size_t nVariableOverallIndex) const
{
    map<size_t, size_t>::const_iterator iter = m_mapVariableIndexes.find(nVariableOverallIndex);
    if(iter != m_mapVariableIndexes.end()) // if found
        return iter->second;
    else
        return ULONG_MAX;
}

bool daeBlock::IsModelDynamic() const
{
    if(!m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    return m_pDataProxy->IsModelDynamic();
}
/*
real_t* daeBlock::GetValuesPointer()
{
#ifdef DAE_DEBUG
    if(!m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);
#endif
    return m_pDataProxy->GetValue(0);
}

real_t* daeBlock::GetTimeDerivativesPointer()
{
#ifdef DAE_DEBUG
    if(!m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);
#endif
    return m_pDataProxy->GetTimeDerivative(0);
}

real_t* daeBlock::GetAbsoluteTolerancesPointer()
{
#ifdef DAE_DEBUG
    if(!m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);
#endif
    return m_pDataProxy->GetAbsoluteTolerance(0);
}

real_t* daeBlock::GetVariableTypesPointer()
{
#ifdef DAE_DEBUG
    if(!m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);
#endif
    return m_pDataProxy->GetVariableTypes();
}
*/

void daeBlock::CleanUpSetupData()
{
    if(!m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);
    m_pDataProxy->CleanUpSetupData();
}

void daeBlock::CreateIndexMappings(real_t* pdValues, real_t* pdTimeDerivatives)
{
    if(!m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);
    m_pDataProxy->CreateIndexMappings(m_mapVariableIndexes, pdValues, pdTimeDerivatives);
}

void daeBlock::SetBlockData(daeArray<real_t>& arrValues, daeArray<real_t>& arrTimeDerivatives)
{
// Now we use block indexes to directly access the solvers arrays and therefore no need to actually copy anything
    m_parrValues          = &arrValues;
    m_parrTimeDerivatives = &arrTimeDerivatives;

/*
#ifdef DAE_DEBUG
    if(!m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);
#endif
// m_mapVariableIndexes<nOverallIndex, nBlockIndex>
    real_t* pBlockValues          = m_pDataProxy->GetValue(0);
    real_t* pBlockTimeDerivatives = m_pDataProxy->GetTimeDerivative(0);

    for(map<size_t, size_t>::iterator iter = m_mapVariableIndexes.begin(); iter != m_mapVariableIndexes.end(); iter++)
    {
        pBlockValues         [iter->first] = arrValues         [iter->second];
        pBlockTimeDerivatives[iter->first] = arrTimeDerivatives[iter->second];
    }
*/
}

/*
void daeBlock::CopyDataToSolver(daeArray<real_t>& arrValues, daeArray<real_t>& arrTimeDerivatives) const
{
#ifdef DAE_DEBUG
    if(!m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);
#endif
// m_mapVariableIndexes<nOverallIndex, nBlockIndex>
    const real_t* pBlockValues          = m_pDataProxy->GetValue(0);
    const real_t* pBlockTimeDerivatives = m_pDataProxy->GetTimeDerivative(0);

    for(map<size_t, size_t>::const_iterator iter = m_mapVariableIndexes.begin(); iter != m_mapVariableIndexes.end(); iter++)
    {
        arrValues         [iter->second] = pBlockValues         [iter->first];
        arrTimeDerivatives[iter->second] = pBlockTimeDerivatives[iter->first];
    }
}
*/

void daeBlock::Initialize(void)
{
    size_t i;
    pair<size_t, size_t> uintPair;
    map<size_t, size_t>::iterator iter;
    daeSTN* pSTN;

    if(!m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    m_pDataProxy->SetBlock(this);

    if(GetNumberOfEquations() != m_mapVariableIndexes.size())
    {
        daeDeclareException(exInvalidCall);
        e << "Number of equations [" << GetNumberOfEquations() << "] is not equal to number of variables used in equations [" << m_mapVariableIndexes.size() << "]";
        throw e;
    }

    // Load evaluation options
    daeConfig& cfg = daeGetConfig();
    m_omp_num_threads = 0;
    if(m_pDataProxy->GetEvaluationMode() == eEvaluationTree_OpenMP)
    {
        m_omp_num_threads = cfg.GetInteger("daetools.core.equations.evaluationTree_OpenMP.numThreads", 0);
    }
    else if(m_pDataProxy->GetEvaluationMode() == eComputeStack_OpenMP)
    {
        m_omp_num_threads = cfg.GetInteger("daetools.core.equations.computeStack_OpenMP.numThreads", 0);
    }

// First BuildExpressions in the top level model and its children
    daeModel* pModel = dynamic_cast<daeModel*>(m_pDataProxy->GetTopLevelModel());
    if(!pModel)
        daeDeclareException(exInvalidPointer);
    pModel->BuildExpressions(this);

// Then BuildExpressions in all STNs
    for(i = 0; i < m_ptrarrSTNs.size(); i++)
    {
        pSTN = m_ptrarrSTNs[i];
        if(!pSTN)
            daeDeclareAndThrowException(exInvalidPointer);

        pSTN->BuildExpressions(this);
    }

/* We do not have to check for discontinuities here - it will be done during the initialization
   And, btw, this does not affect anything, it just checks for discontinuities. Properly done,
   it should call ExecuteOnConditionActions() if CheckDiscontinuities() returns true.

    pModel->CheckDiscontinuities();

    for(i = 0; i < m_ptrarrSTNs.size(); i++)
    {
        pSTN = m_ptrarrSTNs[i];
        if(!pSTN)
            daeDeclareAndThrowException(exInvalidPointer);

        pSTN->CheckDiscontinuities();
    }
*/

    // RebuildActiveEquationSetAndRootExpressions will be called in the DAESolver::Initialize function.
    // Perhaps it should not be called here?
    //RebuildActiveEquationSetAndRootExpressions();
}

bool daeBlock::CheckForDiscontinuities(void)
{
    size_t i;
    daeSTN* pSTN;

    if(m_dCurrentTime > 0 && m_pDataProxy->PrintInfo())
        m_pDataProxy->LogMessage(string("Checking state transitions at time ") + toStringFormatted<real_t>(m_dCurrentTime, -1, 15) + string("..."), 0);

// Achtung, Achtung!!
// Moved to daeSimulation::Integrate_xxx() functions to get it reset before every call to simulation.Integrate/daesolver.Solve
//    m_pDataProxy->SetLastSatisfiedCondition(NULL);

// First check discontinuities in the top level model
    daeModel* pModel = dynamic_cast<daeModel*>(m_pDataProxy->GetTopLevelModel());
    if(pModel->CheckDiscontinuities())
        return true;

// Then check conditions from STNs
    for(i = 0; i < m_ptrarrSTNs.size(); i++)
    {
        pSTN = m_ptrarrSTNs[i];
        if(pSTN->CheckDiscontinuities())
            return true;
    }

    return false;
}

daeeDiscontinuityType daeBlock::ExecuteOnConditionActions(void)
{
    size_t i;
    daeSTN* pSTN;
    daeeDiscontinuityType eResult;

    m_pDataProxy->SetReinitializationFlag(false);
    m_pDataProxy->SetCopyDataFromBlock(false);

    daeModel* pModel = dynamic_cast<daeModel*>(m_pDataProxy->GetTopLevelModel());
    pModel->ExecuteOnConditionActions();

    for(i = 0; i < m_ptrarrSTNs.size(); i++)
    {
        pSTN = m_ptrarrSTNs[i];
        pSTN->ExecuteOnConditionActions();
    }

// If any of the actions changed the state it has to be indicated in those flags
    if(m_pDataProxy->GetReinitializationFlag() && m_pDataProxy->GetCopyDataFromBlock())
    {
        eResult = eModelDiscontinuityWithDataChange;
    }
    else if(m_pDataProxy->GetReinitializationFlag())
    {
        eResult = eModelDiscontinuity;
    }
    else
    {
        eResult = eNoDiscontinuity;
    }

    return eResult;
}

void daeBlock::RebuildActiveEquationSetAndRootExpressions(bool bCalculateSensitivities)
{
// 1. (Re-)build the root expressions.
    m_mapExpressionInfos.clear();

    // First rebuild for the top level model
    daeModel* pModel = dynamic_cast<daeModel*>(m_pDataProxy->GetTopLevelModel());
    pModel->AddExpressionsToBlock(this);

    // Then for all other STNs
    for(size_t i = 0; i < m_ptrarrSTNs.size(); i++)
    {
        daeSTN* pSTN = m_ptrarrSTNs[i];
        pSTN->AddExpressionsToBlock(this);
    }

// 2. (Re-)build the active equations set.
    // I could optimize this function by skipping addition of m_ptrarrEquationExecutionInfos every time!!
    // a) Add EEIs from models (excluding STNs).
    m_ptrarrEquationExecutionInfos_ActiveSet.clear();
    m_ptrarrEquationExecutionInfos_ActiveSet.reserve(m_nNumberOfEquations);

    m_ptrarrEquationExecutionInfos_ActiveSet.insert(m_ptrarrEquationExecutionInfos_ActiveSet.begin(),
                                                    m_ptrarrEquationExecutionInfos.begin(),
                                                    m_ptrarrEquationExecutionInfos.end());

    // b) Add EEIs from active states of all STNs/IFs.
    for(size_t i = 0; i < m_ptrarrSTNs.size(); i++)
    {
        daeSTN* pSTN = m_ptrarrSTNs[i];
        pSTN->CollectEquationExecutionInfos(m_ptrarrEquationExecutionInfos_ActiveSet);
    }

// 3. (Re-)build the jacobian items for the active equation set.
    if(m_pDataProxy->GetEvaluationMode() == eComputeStack_OpenMP ||
       m_pDataProxy->GetEvaluationMode() == eComputeStack_External)
    {
        BuildComputeStackStructs();

        if(m_pDataProxy->GetEvaluationMode() == eComputeStack_External)
        {
            if(!m_computeStackEvaluator)
                daeDeclareAndThrowException(exInvalidPointer);

            csComputeStackItem_t*   computeStacks             = &m_arrAllComputeStacks[0];
            uint32_t*               activeEquationSetIndexes  = &m_arrActiveEquationSetIndexes[0];
            csIncidenceMatrixItem_t* computeStackJacobianItems = &m_arrComputeStackJacobianItems[0];

            m_computeStackEvaluator->Initialize(bCalculateSensitivities,
                                                m_nNumberOfEquations,
                                                m_nNumberOfEquations,
                                                m_pDataProxy->GetAssignedVarsValues().size(),
                                                m_arrAllComputeStacks.size(),
                                                m_arrComputeStackJacobianItems.size(),
                                                m_arrComputeStackJacobianItems.size(),
                                                computeStacks,
                                                activeEquationSetIndexes,
                                                computeStackJacobianItems);

        }
        //printf("m_arrAllComputeStacks          = %d\n", m_arrAllComputeStacks.size());
        //printf("m_arrActiveEquationSetIndexes  = %d\n", m_arrActiveEquationSetIndexes.size());
        //printf("m_arrComputeStackJacobianItems = %d\n", m_arrComputeStackJacobianItems.size());
    }
}

std::map<std::string, call_stats::TimeAndCount> daeBlock::GetCallStats() const
{
    return m_stats;
}

void daeBlock::BuildComputeStackStructs()
{
    m_arrComputeStackJacobianItems.clear();
    m_arrActiveEquationSetIndexes.clear();

    size_t nnz = 0;
    for(size_t i = 0; i < m_ptrarrEquationExecutionInfos_ActiveSet.size(); i++)
    {
        daeEquationExecutionInfo* pEEI = m_ptrarrEquationExecutionInfos_ActiveSet[i];
        nnz += pEEI->m_mapIndexes.size();
    }
    m_arrComputeStackJacobianItems.reserve(nnz);
    m_arrActiveEquationSetIndexes.reserve(m_nNumberOfEquations);

    csIncidenceMatrixItem_t jd;
    for(size_t i = 0; i < m_ptrarrEquationExecutionInfos_ActiveSet.size(); i++)
    {
        daeEquationExecutionInfo* pEEI = m_ptrarrEquationExecutionInfos_ActiveSet[i];

        m_arrActiveEquationSetIndexes.push_back( pEEI->m_nComputeStackIndex );

        for(std::map<size_t, size_t>::iterator it = pEEI->m_mapIndexes.begin(); it != pEEI->m_mapIndexes.end(); it++)
        {
            jd.equationIndex = i;
            jd.overallIndex  = it->first;
            jd.blockIndex    = it->second;
            m_arrComputeStackJacobianItems.push_back(jd);
        }
    }
}

void daeBlock::CleanComputeStackStructs()
{
    m_arrComputeStackJacobianItems.clear();
    m_arrActiveEquationSetIndexes.clear();
}

void daeBlock::ExportComputeStackStructs(const std::string& filenameComputeStacks,
                                         const std::string& filenameJacobianIndexes,
                                         int startEquationIndex,
                                         int endEquationIndex,
                                         const std::map<int,int>& bi_to_bi_local)
{
    if(m_pDataProxy->GetEvaluationMode() != eComputeStack_OpenMP &&
       m_pDataProxy->GetEvaluationMode() != eComputeStack_External)
    {
        daeDeclareException(exInvalidCall);
        e << "ExportComputeStackStructs requires the equation evaluation mode eComputeStack_OpenMP or eComputeStack_External";
        throw e;
    }

    uint32_t*               activeEquationSetIndexes  = NULL;
    csComputeStackItem_t*   computeStacks             = NULL;
    csIncidenceMatrixItem_t* computeStackJacobianItems = NULL;
    uint32_t Ncs;
    uint32_t Nasi;
    uint32_t Nji;

    if(startEquationIndex == 0 && endEquationIndex == -1) // export all items
    {
        Ncs  = m_arrAllComputeStacks.size();
        Nasi = m_arrActiveEquationSetIndexes.size();
        Nji  = m_arrComputeStackJacobianItems.size();

        computeStacks             = &m_arrAllComputeStacks[0];
        activeEquationSetIndexes  = &m_arrActiveEquationSetIndexes[0];
        computeStackJacobianItems = &m_arrComputeStackJacobianItems[0];

        /* Write model equations. */
        std::ofstream f;
        f.open(filenameComputeStacks, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
        if(!f.is_open())
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot open " << filenameComputeStacks << " file.";
            throw e;
        }

        int32_t fileType      = cs::eInputFile_ModelEquations;
        int32_t opencsVersion = OPENCS_VERSION;
        f.write((char*)&fileType,       sizeof(int32_t));
        f.write((char*)&opencsVersion,  sizeof(int32_t));

        f.write((char*)&Nasi,                     sizeof(uint32_t));
        f.write((char*)&Ncs,                      sizeof(uint32_t));
        f.write((char*)activeEquationSetIndexes,  sizeof(uint32_t)               * Nasi);
        f.write((char*)computeStacks,             sizeof(csComputeStackItem_t)   * Ncs);
        f.close();

        /* Write incidence matrix. */
        f.open(filenameJacobianIndexes, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
        if(!f.is_open())
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot open " << filenameJacobianIndexes << " file.";
            throw e;
        }
        fileType      = cs::eInputFile_SparsityPattern;
        opencsVersion = OPENCS_VERSION;
        f.write((char*)&fileType,       sizeof(int32_t));
        f.write((char*)&opencsVersion,  sizeof(int32_t));

        uint32_t Neq = Nasi;
        std::vector<uint32_t> rowIndexes(Neq+1, 0);
        for(uint32_t ji = 0; ji < Nji; ji++)
        {
            csIncidenceMatrixItem_t& jd = computeStackJacobianItems[ji];
            rowIndexes[ jd.equationIndex + 1 ]++;
        }

        f.write((char*)&Neq,                      sizeof(uint32_t));
        f.write((char*)&Nji,                      sizeof(uint32_t));
        f.write((char*)&rowIndexes[0],            sizeof(uint32_t) * (Neq+1));
        f.write((char*)computeStackJacobianItems, sizeof(csIncidenceMatrixItem_t) * Nji);
        f.close();
    }
    else
    {
        uint32_t startComputeStackIndex    = -1;
        uint32_t startJacobianIndex        = -1;
        uint32_t numberOfComputeStacks     = 0;
        uint32_t numberOfJacobianItems     = 0;

        uint32_t numberOfEquations = endEquationIndex - startEquationIndex;

        activeEquationSetIndexes  = &m_arrActiveEquationSetIndexes[startEquationIndex];
        startComputeStackIndex    = activeEquationSetIndexes[0];
        computeStacks             = &m_arrAllComputeStacks[startComputeStackIndex];

        // Find the start index and the number of Jacobian matrix items.
        {
            std::vector<int> indexes;
            for(int i = 0; i < m_arrComputeStackJacobianItems.size(); i++)
            {
                csIncidenceMatrixItem_t jd = m_arrComputeStackJacobianItems[i];
                if(jd.equationIndex >= startEquationIndex && jd.equationIndex < endEquationIndex)
                    indexes.push_back(i);
            }
            startJacobianIndex    = indexes[0];
            numberOfJacobianItems = indexes.size();
            computeStackJacobianItems = &m_arrComputeStackJacobianItems[startJacobianIndex];
        }

        // Find the number of compute stack items.
        numberOfComputeStacks = 0;
        for(int ei = 0; ei < numberOfEquations; ei++)
        {
            uint32_t firstIndex = activeEquationSetIndexes[ei];
            csComputeStackItem_t* computeStack = &m_arrAllComputeStacks[firstIndex];
            csComputeStackItem_t item0 = computeStack[0];
            uint32_t computeStackSize  = item0.size;

            numberOfComputeStacks += computeStackSize;
        }

        // Before updating we need to copy all items so that the original data are not affected!
        // Note: it requires extra memory - is that a problem?
        std::vector<csComputeStackItem_t>    l_arrComputeStacks;
        std::vector<csIncidenceMatrixItem_t> l_arrComputeStackJacobianItems;
        std::vector<uint32_t>                l_arrActiveEquationSetIndexes;

        l_arrComputeStacks.resize(numberOfComputeStacks);
        for(int i = 0; i < numberOfComputeStacks; i++)
            l_arrComputeStacks[i] = computeStacks[i];

        l_arrComputeStackJacobianItems.resize(numberOfJacobianItems);
        for(int i = 0; i < numberOfJacobianItems; i++)
            l_arrComputeStackJacobianItems[i] = computeStackJacobianItems[i];

        l_arrActiveEquationSetIndexes.resize(numberOfEquations);
        for(int i = 0; i < numberOfEquations; i++)
            l_arrActiveEquationSetIndexes[i] = activeEquationSetIndexes[i];

        // Set the pointers to point to the new arrays.
        computeStacks             = &l_arrComputeStacks[0];
        computeStackJacobianItems = &l_arrComputeStackJacobianItems[0];
        activeEquationSetIndexes  = &l_arrActiveEquationSetIndexes[0];

        //printf("file: %s\n", strFilename.c_str());
        //printf("startComputeStackIndex = %d\n", (int)startComputeStackIndex);
        //for(int ei = 0; ei < numberOfEquations; ei++)
        //    printf("{%d, %d} ", (int)ei, (int)l_arrActiveEquationSetIndexes[ei]);
        //printf("\n");

        for(int ei = 0; ei < numberOfEquations; ei++)
        {
            // Update activeEquationSetIndexes to point to a zero-based array.
            uint32_t firstIndex     = l_arrActiveEquationSetIndexes[ei];
            uint32_t firstIndex_new = firstIndex - startComputeStackIndex;
            l_arrActiveEquationSetIndexes[ei] = firstIndex_new;

            // Update block indexes in the compute stack to mpi-node block indexes.
            csComputeStackItem_t* computeStack = &l_arrComputeStacks[firstIndex_new];
            csComputeStackItem_t item0 = computeStack[0];
            uint32_t computeStackSize  = item0.size;
            for(uint32_t i = 0; i < computeStackSize; i++)
            {
                csComputeStackItem_t& item = computeStack[i];

                if(item.opCode == eOP_Variable)
                {
                    uint32_t bi_local = bi_to_bi_local.at(item.data.indexes.blockIndex);
                    item.data.indexes.blockIndex = bi_local;
                }
                else if(item.opCode == eOP_DegreeOfFreedom)
                {
                    // Leave dofIndex as it is for we have all dofs in the model
                }
                else if(item.opCode == eOP_TimeDerivative)
                {
                    uint32_t bi_local = bi_to_bi_local.at(item.data.indexes.blockIndex);
                    item.data.indexes.blockIndex = bi_local;
                }
            }
        }

        // Update equationIndex in the computeStackJacobianItems to point to the mpi-node indexes.
        for(int ji = 0; ji < numberOfJacobianItems; ji++)
        {
            csIncidenceMatrixItem_t& jd = l_arrComputeStackJacobianItems[ji];
            jd.equationIndex -= startEquationIndex;
            jd.blockIndex     = bi_to_bi_local.at(jd.blockIndex);
        }

        Ncs  = numberOfComputeStacks;
        Nasi = numberOfEquations;
        Nji  = numberOfJacobianItems;

        /* Write model equations. */
        std::ofstream f;
        f.open(filenameComputeStacks, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
        if(!f.is_open())
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot open " << filenameComputeStacks << " file.";
            throw e;
        }

        int32_t fileType      = cs::eInputFile_ModelEquations;
        int32_t opencsVersion = OPENCS_VERSION;
        f.write((char*)&fileType,       sizeof(int32_t));
        f.write((char*)&opencsVersion,  sizeof(int32_t));

        f.write((char*)&Nasi,                     sizeof(uint32_t));
        f.write((char*)&Ncs,                      sizeof(uint32_t));
        f.write((char*)activeEquationSetIndexes,  sizeof(uint32_t)               * Nasi);
        f.write((char*)computeStacks,             sizeof(csComputeStackItem_t)   * Ncs);
        f.close();

        /* Write incidence matrix. */
        f.open(filenameJacobianIndexes, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
        if(!f.is_open())
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot open " << filenameJacobianIndexes << " file.";
            throw e;
        }

        fileType      = cs::eInputFile_SparsityPattern;
        opencsVersion = OPENCS_VERSION;
        f.write((char*)&fileType,       sizeof(int32_t));
        f.write((char*)&opencsVersion,  sizeof(int32_t));

        uint32_t Neq = Nasi;
        std::vector<uint32_t> rowIndexes(Neq+1, 0);
        for(uint32_t ji = 0; ji < Nji; ji++)
        {
            csIncidenceMatrixItem_t& jd = computeStackJacobianItems[ji];
            rowIndexes[ jd.equationIndex + 1 ]++;
        }

        f.write((char*)&Neq              ,        sizeof(uint32_t));
        f.write((char*)&Nji,                      sizeof(uint32_t));
        f.write((char*)&rowIndexes[0],            sizeof(uint32_t) * (Neq+1));
        f.write((char*)computeStackJacobianItems, sizeof(csIncidenceMatrixItem_t) * Nji);
        f.close();
    }
}

void daeBlock::ExportComputeStackStructs(const std::string& filenameComputeStacks,
                                         const std::string& filenameJacobianIndexes,
                                         const std::vector<uint32_t> &equationIndexes,
                                         const std::map<uint32_t, uint32_t> &bi_to_bi_local)
{
    if(m_pDataProxy->GetEvaluationMode() != eComputeStack_OpenMP &&
       m_pDataProxy->GetEvaluationMode() != eComputeStack_External)
    {
        daeDeclareException(exInvalidCall);
        e << "ExportComputeStackStructs requires the equation evaluation mode eComputeStack_OpenMP or eComputeStack_External";
        throw e;
    }

    std::map<uint32_t, uint32_t> ei_to_local_ei;
    for(uint32_t i = 0; i < equationIndexes.size(); i++)
    {
        uint32_t ei = equationIndexes[i];
        ei_to_local_ei[ei] = i;
    }

    uint32_t*               activeEquationSetIndexes  = NULL;
    csComputeStackItem_t*   computeStacks             = NULL;
    csIncidenceMatrixItem_t* computeStackJacobianItems = NULL;
    uint32_t Ncs;
    uint32_t Nasi;
    uint32_t Nji;

    uint32_t numberOfComputeStacks = 0;
    uint32_t numberOfJacobianItems = 0;
    uint32_t numberOfEquations     = equationIndexes.size();

    // Find the Jacobian matrix items.
    numberOfJacobianItems = 0;
    std::vector<int> jacobianItems;
    {
        std::map<uint32_t, uint32_t>::iterator eq_end = ei_to_local_ei.end();
        for(int i = 0; i < m_arrComputeStackJacobianItems.size(); i++)
        {
            csIncidenceMatrixItem_t jd = m_arrComputeStackJacobianItems[i];
            // If jd.equationIndex is in the map with requested equation indexes add it to the list.
            if( ei_to_local_ei.find(jd.equationIndex) != eq_end )
                jacobianItems.push_back(i);
        }
        numberOfJacobianItems = jacobianItems.size();

        //printf("numberOfJacobianItems = %d\n    ", (int)numberOfJacobianItems);
        //for(int ei = 0; ei < numberOfJacobianItems; ei++)
        //    printf("%d ", jacobianItems[ei]);
        //printf("\n");
    }

    // Find the number of compute stack items and collect pointers to first items.
    // Create a map <overallEquationIndex, localEquationIndex> to be used for updating jacobian items.
    numberOfComputeStacks = 0;
    std::vector<csComputeStackItem_t*> computeStackPointers;
    for(int i = 0; i < numberOfEquations; i++)
    {
        uint32_t ei = equationIndexes[i];
        uint32_t firstIndex = m_arrActiveEquationSetIndexes[ei];
        csComputeStackItem_t* item0 = &m_arrAllComputeStacks[firstIndex];
        uint32_t computeStackSize = item0->size;

        computeStackPointers.push_back(item0);

        numberOfComputeStacks += computeStackSize;
    }
    //printf("numberOfComputeStacks = %d\n", (int)numberOfComputeStacks);

    // Before updating we need to copy all items so that the original data are not affected!
    // Note: it requires extra memory - is that a problem?
    std::vector<csComputeStackItem_t>   l_arrComputeStacks;
    std::vector<csIncidenceMatrixItem_t> l_arrComputeStackJacobianItems;
    std::vector<uint32_t>               l_arrActiveEquationSetIndexes;

    l_arrComputeStacks.reserve(numberOfComputeStacks);
    l_arrActiveEquationSetIndexes.resize(numberOfEquations);
    uint32_t counter = 0;
    for(int i = 0; i < computeStackPointers.size(); i++)
    {
        csComputeStackItem_t* cs = computeStackPointers[i];
        uint32_t computeStackSize = cs->size;
        for(int j = 0; j < computeStackSize; j++)
            l_arrComputeStacks.push_back(cs[j]);

        // Update activeEquationSetIndexes to point to a zero-based array.
        l_arrActiveEquationSetIndexes[i] = counter;
        counter += computeStackSize;
    }

    l_arrComputeStackJacobianItems.resize(numberOfJacobianItems);
    for(int i = 0; i < numberOfJacobianItems; i++)
        l_arrComputeStackJacobianItems[i] = m_arrComputeStackJacobianItems[ jacobianItems[i] ];

    // Set the pointers to point to the new arrays.
    computeStacks             = &l_arrComputeStacks[0];
    computeStackJacobianItems = &l_arrComputeStackJacobianItems[0];
    activeEquationSetIndexes  = &l_arrActiveEquationSetIndexes[0];

    //printf("%s numberOfJacobianItems = %d\n", filenameJacobianIndexes.c_str(), (int)numberOfJacobianItems);
    //printf("file: %s\n", strFilename.c_str());
    //printf("startComputeStackIndex = %d\n", (int)startComputeStackIndex);
    //for(int ei = 0; ei < numberOfEquations; ei++)
    //    printf("{%d, %d} ", (int)ei, (int)l_arrActiveEquationSetIndexes[ei]);
    //printf("\n");

    std::map<uint32_t, uint32_t>::const_iterator bi_end = bi_to_bi_local.end();
    std::map<uint32_t, uint32_t>::const_iterator bi_cit;
    for(int ei = 0; ei < numberOfEquations; ei++)
    {
        uint32_t firstIndex = l_arrActiveEquationSetIndexes[ei];

        // Update block indexes in the compute stack to mpi-node block indexes.
        csComputeStackItem_t* computeStack = &l_arrComputeStacks[firstIndex];
        uint32_t computeStackSize = computeStack->size;
        for(uint32_t i = 0; i < computeStackSize; i++)
        {
            csComputeStackItem_t& item = computeStack[i];

            if(item.opCode == eOP_Variable)
            {
                bi_cit = bi_to_bi_local.find(item.data.indexes.blockIndex);
                if(bi_cit == bi_end)
                {
                    daeDeclareException(exInvalidCall);
                    e << "Block index " << item.data.indexes.blockIndex << " cannot be found in bi_to_bi_local map";
                    throw e;
                }
                item.data.indexes.blockIndex = bi_cit->second;
            }
            else if(item.opCode == eOP_DegreeOfFreedom)
            {
                // Leave dofIndex as it is for we have all dofs in the model
            }
            else if(item.opCode == eOP_TimeDerivative)
            {
                bi_cit = bi_to_bi_local.find(item.data.indexes.blockIndex);
                if(bi_cit == bi_end)
                {
                    daeDeclareException(exInvalidCall);
                    e << "Block index " << item.data.indexes.blockIndex << " cannot be found in bi_to_bi_local map";
                    throw e;
                }
                item.data.indexes.blockIndex = bi_cit->second;
            }
        }
    }

    // Update equationIndex in the computeStackJacobianItems to point to the mpi-node indexes.
    for(int ji = 0; ji < numberOfJacobianItems; ji++)
    {
        csIncidenceMatrixItem_t& jd = l_arrComputeStackJacobianItems[ji];

        jd.equationIndex = ei_to_local_ei.at(jd.equationIndex);
        jd.blockIndex    = bi_to_bi_local.at(jd.blockIndex);
    }

    Ncs  = numberOfComputeStacks;
    Nasi = numberOfEquations;
    Nji  = numberOfJacobianItems;

    /* Write model equations. */
    std::ofstream f;
    f.open(filenameComputeStacks, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    if(!f.is_open())
    {
        daeDeclareException(exInvalidCall);
        e << "Cannot open " << filenameComputeStacks << " file.";
        throw e;
    }

    int32_t fileType      = cs::eInputFile_ModelEquations;
    int32_t opencsVersion = OPENCS_VERSION;
    f.write((char*)&fileType,       sizeof(int32_t));
    f.write((char*)&opencsVersion,  sizeof(int32_t));

    f.write((char*)&Nasi,                     sizeof(uint32_t));
    f.write((char*)&Ncs,                      sizeof(uint32_t));
    f.write((char*)activeEquationSetIndexes,  sizeof(uint32_t)               * Nasi);
    f.write((char*)computeStacks,             sizeof(csComputeStackItem_t)   * Ncs);
    f.close();

    /* Write incidence matrix. */
    f.open(filenameJacobianIndexes, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    if(!f.is_open())
    {
        daeDeclareException(exInvalidCall);
        e << "Cannot open " << filenameJacobianIndexes << " file.";
        throw e;
    }

    fileType      = cs::eInputFile_SparsityPattern;
    opencsVersion = OPENCS_VERSION;
    f.write((char*)&fileType,       sizeof(int32_t));
    f.write((char*)&opencsVersion,  sizeof(int32_t));

    uint32_t Neq = Nasi;
    std::vector<uint32_t> rowIndexes(Neq+1, 0);
    for(uint32_t ji = 0; ji < Nji; ji++)
    {
        csIncidenceMatrixItem_t& jd = computeStackJacobianItems[ji];
        rowIndexes[ jd.equationIndex + 1 ]++;
    }

    f.write((char*)&Neq,                      sizeof(uint32_t));
    f.write((char*)&Nji,                      sizeof(uint32_t));
    f.write((char*)&rowIndexes[0],            sizeof(uint32_t) * (Neq+1));
    f.write((char*)computeStackJacobianItems, sizeof(csIncidenceMatrixItem_t) * Nji);
    f.close();
}

/*
void daeBlock::ExportComputeStackStructs(const std::string& filenameComputeStacks,
                                         const std::string& filenameJacobianIndexes,
                                         const std::vector<int>& equationIndexes,
                                         const std::map<int,int>& bi_to_bi_local)
{
    if(m_pDataProxy->GetEvaluationMode() != eComputeStack_OpenMP &&
       m_pDataProxy->GetEvaluationMode() != eComputeStack_External)
    {
        daeDeclareException(exInvalidCall);
        e << "ExportComputeStackStructs requires the equation evaluation mode eComputeStack_OpenMP or eComputeStack_External";
        throw e;
    }

    uint32_t*               activeEquationSetIndexes  = NULL;
    csComputeStackItem_t*   computeStacks             = NULL;
    csIncidenceMatrixItem_t* computeStackJacobianItems = NULL;
    uint32_t Ncs;
    uint32_t Nasi;
    uint32_t Nji;

    uint32_t numberOfComputeStacks = 0;
    uint32_t numberOfJacobianItems = 0;
    uint32_t numberOfEquations     = equationIndexes.size();

    // Find the Jacobian matrix items.
    numberOfJacobianItems = 0;
    std::vector<int> jacobianItems;
    {
        std::vector<int>::const_iterator eq_begin = equationIndexes.begin();
        std::vector<int>::const_iterator eq_end   = equationIndexes.end();
        for(int i = 0; i < m_arrComputeStackJacobianItems.size(); i++)
        {
            csIncidenceMatrixItem_t jd = m_arrComputeStackJacobianItems[i];
            if( std::find(eq_begin, eq_end, (int)jd.equationIndex) != eq_end )
                jacobianItems.push_back(i);
        }
        numberOfJacobianItems = jacobianItems.size();

        //printf("numberOfJacobianItems = %d\n    ", (int)numberOfJacobianItems);
        //for(int ei = 0; ei < numberOfJacobianItems; ei++)
        //    printf("%d ", jacobianItems[ei]);
        //printf("\n");
    }

    // Find the number of compute stack items and collect pointers to first items.
    // Create a map <overallEquationIndex, localEquationIndex> to be used for updating jacobian items.
    numberOfComputeStacks = 0;
    std::map<int,int> ei_to_local_ei;
    std::vector<csComputeStackItem_t*> computeStackPointers;
    for(int i = 0; i < numberOfEquations; i++)
    {
        int ei = equationIndexes[i];
        uint32_t firstIndex = m_arrActiveEquationSetIndexes[ei];
        csComputeStackItem_t* item0 = &m_arrAllComputeStacks[firstIndex];
        uint32_t computeStackSize = item0->size;

        ei_to_local_ei[ei] = i;
        computeStackPointers.push_back(item0);

        numberOfComputeStacks += computeStackSize;
    }
    //printf("numberOfComputeStacks = %d\n", (int)numberOfComputeStacks);

    // Before updating we need to copy all items so that the original data are not affected!
    // Note: it requires extra memory - is that a problem?
    std::vector<csComputeStackItem_t>   l_arrComputeStacks;
    std::vector<csIncidenceMatrixItem_t> l_arrComputeStackJacobianItems;
    std::vector<uint32_t>               l_arrActiveEquationSetIndexes;

    l_arrComputeStacks.reserve(numberOfComputeStacks);
    l_arrActiveEquationSetIndexes.resize(numberOfEquations);
    uint32_t counter = 0;
    for(int i = 0; i < computeStackPointers.size(); i++)
    {
        csComputeStackItem_t* cs = computeStackPointers[i];
        uint32_t computeStackSize = cs->size;
        for(int j = 0; j < computeStackSize; j++)
            l_arrComputeStacks.push_back(cs[j]);

        // Update activeEquationSetIndexes to point to a zero-based array.
        l_arrActiveEquationSetIndexes[i] = counter;
        counter += computeStackSize;
    }

    l_arrComputeStackJacobianItems.resize(numberOfJacobianItems);
    for(int i = 0; i < numberOfJacobianItems; i++)
        l_arrComputeStackJacobianItems[i] = m_arrComputeStackJacobianItems[ jacobianItems[i] ];

    // Set the pointers to point to the new arrays.
    computeStacks             = &l_arrComputeStacks[0];
    computeStackJacobianItems = &l_arrComputeStackJacobianItems[0];
    activeEquationSetIndexes  = &l_arrActiveEquationSetIndexes[0];

    //printf("file: %s\n", strFilename.c_str());
    //printf("startComputeStackIndex = %d\n", (int)startComputeStackIndex);
    //for(int ei = 0; ei < numberOfEquations; ei++)
    //    printf("{%d, %d} ", (int)ei, (int)l_arrActiveEquationSetIndexes[ei]);
    //printf("\n");

    for(int ei = 0; ei < numberOfEquations; ei++)
    {
        uint32_t firstIndex = l_arrActiveEquationSetIndexes[ei];

        // Update block indexes in the compute stack to mpi-node block indexes.
        csComputeStackItem_t* computeStack = &l_arrComputeStacks[firstIndex];
        uint32_t computeStackSize = computeStack->size;
        for(uint32_t i = 0; i < computeStackSize; i++)
        {
            csComputeStackItem_t& item = computeStack[i];

            if(item.opCode == eOP_Variable)
            {
                uint32_t bi_local = bi_to_bi_local.at(item.data.indexes.blockIndex);
                item.data.indexes.blockIndex = bi_local;
            }
            else if(item.opCode == eOP_DegreeOfFreedom)
            {
                // Leave dofIndex as it is for we have all dofs in the model
            }
            else if(item.opCode == eOP_TimeDerivative)
            {
                uint32_t bi_local = bi_to_bi_local.at(item.data.indexes.blockIndex);
                item.data.indexes.blockIndex = bi_local;
            }
        }
    }

    // Update equationIndex in the computeStackJacobianItems to point to the mpi-node indexes.
    for(int ji = 0; ji < numberOfJacobianItems; ji++)
    {
        csIncidenceMatrixItem_t& jd = l_arrComputeStackJacobianItems[ji];

        int ei = static_cast<int>(jd.equationIndex);
        int bi = static_cast<int>(jd.blockIndex);
        jd.equationIndex = static_cast<uint32_t>(ei_to_local_ei.at(ei));
        jd.blockIndex    = static_cast<uint32_t>(bi_to_bi_local.at(bi));
    }

    Ncs  = numberOfComputeStacks;
    Nasi = numberOfEquations;
    Nji  = numberOfJacobianItems;

    std::ofstream f;
    f.open(filenameComputeStacks, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    if(!f.is_open())
    {
        daeDeclareException(exInvalidCall);
        e << "Cannot open " << filenameComputeStacks << " file.";
        throw e;
    }
    f.write((char*)&Nasi,                     sizeof(uint32_t));
    f.write((char*)&Ncs,                      sizeof(uint32_t));
    f.write((char*)activeEquationSetIndexes,  sizeof(uint32_t)               * Nasi);
    f.write((char*)computeStacks,             sizeof(csComputeStackItem_t)   * Ncs);
    f.close();

    f.open(filenameJacobianIndexes, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    if(!f.is_open())
    {
        daeDeclareException(exInvalidCall);
        e << "Cannot open " << filenameJacobianIndexes << " file.";
        throw e;
    }
    f.write((char*)&Nji,                      sizeof(uint32_t));
    f.write((char*)computeStackJacobianItems, sizeof(csIncidenceMatrixItem_t) * Nji);
    f.close();
}
*/

bool daeBlock::CheckOverlappingAndAddVariables(const vector<size_t>& narrVariablesInEquation)
{
    size_t i, k;
    pair<size_t, size_t> uintPair;
    map<size_t, size_t>::iterator iter;

    for(i = 0; i < narrVariablesInEquation.size(); i++)
    {
        iter = m_mapVariableIndexes.find(narrVariablesInEquation[i]);
        if(iter != m_mapVariableIndexes.end()) //if found
        {
            for(k = 0; k < narrVariablesInEquation.size(); k++)
            {
                uintPair.first  = narrVariablesInEquation[k];  // overall block
                uintPair.second = m_mapVariableIndexes.size(); // index in block
                m_mapVariableIndexes.insert(uintPair);
            }
            return true;
        }
    }

    return false;
}

void daeBlock::AddVariables(const map<size_t, size_t>& mapIndexes)
{
    pair<size_t, size_t> uintPair;
    map<size_t, size_t>::const_iterator iter;

    for(iter = mapIndexes.begin(); iter != mapIndexes.end(); iter++)
    {
        uintPair.first  = (*iter).first;				// overall index
        uintPair.second = m_mapVariableIndexes.size();	// index in block
        m_mapVariableIndexes.insert(uintPair);
    }
}

string daeBlock::GetCanonicalName(void) const
{
    return m_strName;
}

string daeBlock::GetName(void) const
{
    return m_strName;
}

void daeBlock::SetName(const string& strName)
{
    m_strName = strName;
}

size_t daeBlock::GetNumberOfRoots() const
{
    size_t nNoRoots = 0;

    return (nNoRoots + m_mapExpressionInfos.size());
}

daeDataProxy_t* daeBlock::GetDataProxy(void) const
{
    return m_pDataProxy;
}

void daeBlock::SetDataProxy(daeDataProxy_t* pDataProxy)
{
    if(!pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);
    m_pDataProxy = pDataProxy;
}

map<size_t, size_t>& daeBlock::GetVariableIndexesMap()
{
    return m_mapVariableIndexes;
}

void daeBlock::AddEquationExecutionInfo(daeEquationExecutionInfo* pEquationExecutionInfo)
{
    if(!pEquationExecutionInfo)
        daeDeclareAndThrowException(exInvalidPointer);

    m_ptrarrEquationExecutionInfos.push_back(pEquationExecutionInfo);
}

vector<daeEquationExecutionInfo*>& daeBlock::GetEquationExecutionInfos_ActiveSet()
{
    return m_ptrarrEquationExecutionInfos_ActiveSet;
}

size_t daeBlock::GetNumberOfEquations() const
{
    return m_nNumberOfEquations;
}

real_t daeBlock::GetValue(size_t nBlockIndex) const
{
#ifdef DAE_DEBUG
    if(!m_parrValues)
        daeDeclareAndThrowException(exInvalidPointer);
#endif
    return (*m_parrValues).GetItem(nBlockIndex);
}

void daeBlock::SetValue(size_t nBlockIndex, real_t dValue)
{
#ifdef DAE_DEBUG
    if(!m_parrValues)
        daeDeclareAndThrowException(exInvalidPointer);
#endif
    (*m_parrValues).SetItem(nBlockIndex, dValue);
}

real_t daeBlock::GetTimeDerivative(size_t nBlockIndex) const
{
#ifdef DAE_DEBUG
    if(!m_parrTimeDerivatives)
        daeDeclareAndThrowException(exInvalidPointer);
    if(nBlockIndex >= m_nNumberOfEquations)
        daeDeclareAndThrowException(exOutOfBounds);
#endif
    return (*m_parrTimeDerivatives).GetItem(nBlockIndex);
}

void daeBlock::SetTimeDerivative(size_t nBlockIndex, real_t dTimeDerivative)
{
#ifdef DAE_DEBUG
    if(!m_parrTimeDerivatives)
        daeDeclareAndThrowException(exInvalidPointer);
#endif
    (*m_parrTimeDerivatives).SetItem(nBlockIndex, dTimeDerivative);
}

real_t daeBlock::GetResidual(size_t nEquationIndex) const
{
#ifdef DAE_DEBUG
    if(!m_parrResidual)
        daeDeclareAndThrowException(exInvalidPointer);
#endif
    return (*m_parrResidual).GetItem(nEquationIndex);
}

void daeBlock::SetResidual(size_t nEquationIndex, real_t dResidual)
{
#ifdef DAE_DEBUG
    if(!m_parrResidual)
        daeDeclareAndThrowException(exInvalidPointer);
#endif
    (*m_parrResidual).SetItem(nEquationIndex, dResidual);
}

real_t daeBlock::GetJacobian(size_t nEquationIndex, size_t nVariableindexInBlock) const
{
#ifdef DAE_DEBUG
    if(!m_pmatJacobian)
        daeDeclareAndThrowException(exInvalidPointer);
#endif
    return m_pmatJacobian->GetItem(nEquationIndex, nVariableindexInBlock);
}

void daeBlock::SetJacobian(size_t nEquationIndex, size_t nVariableindexInBlock, real_t dJacobianItem)
{
#ifdef DAE_DEBUG
    if(!m_pmatJacobian)
        daeDeclareAndThrowException(exInvalidPointer);
#endif
    m_pmatJacobian->SetItem(nEquationIndex, nVariableindexInBlock, dJacobianItem);
}

void daeBlock::SetValuesArray(daeArray<real_t>* pValues)
{
#ifdef DAE_DEBUG
    if(!pValues)
        daeDeclareAndThrowException(exInvalidPointer);
#endif
    m_parrValues = pValues;
}

daeArray<real_t>* daeBlock::GetValuesArray(void) const
{
    return m_parrValues;
}

void daeBlock::SetTimeDerivativesArray(daeArray<real_t>* pTimeDerivatives)
{
#ifdef DAE_DEBUG
    if(!pTimeDerivatives)
        daeDeclareAndThrowException(exInvalidPointer);
#endif
    m_parrTimeDerivatives = pTimeDerivatives;
}

daeArray<real_t>* daeBlock::GetTimeDerivativesArray(void) const
{
    return m_parrTimeDerivatives;
}

daeMatrix<real_t>* daeBlock::GetJacobianMatrix() const
{
    return m_pmatJacobian;
}

void daeBlock::SetJacobianMatrix(daeMatrix<real_t>* pJacobian)
{
#ifdef DAE_DEBUG
    if(!pJacobian)
        daeDeclareAndThrowException(exInvalidPointer);
#endif
    m_pmatJacobian = pJacobian;
}

daeArray<real_t>* daeBlock::GetResidualArray() const
{
    return m_parrResidual;
}

void daeBlock::SetResidualArray(daeArray<real_t>* pResidual)
{
#ifdef DAE_DEBUG
    if(!pResidual)
        daeDeclareAndThrowException(exInvalidPointer);
#endif
    m_parrResidual = pResidual;
}

real_t daeBlock::GetTime() const
{
    return m_dCurrentTime;
}

void daeBlock::SetTime(real_t dTime)
{
    m_dCurrentTime = dTime;
}

real_t daeBlock::GetInverseTimeStep() const
{
    return m_dInverseTimeStep;
}

void daeBlock::SetInverseTimeStep(real_t dInverseTimeStep)
{
    m_dInverseTimeStep = dInverseTimeStep;
}

bool daeBlock::GetInitializeMode() const
{
    return m_bInitializeMode;
}

void daeBlock::SetInitializeMode(bool bMode)
{
    m_bInitializeMode = bMode;
}

bool daeBlock::CheckObject(vector<string>& strarrErrors) const
{
    dae_capacity_check(m_ptrarrEquationExecutionInfos);
    dae_capacity_check(m_ptrarrSTNs);

    return true;
}


}
}

