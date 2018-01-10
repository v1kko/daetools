#ifndef IDA_DAESOLVER_H
#define IDA_DAESOLVER_H

#include "../Core/helpers.h"
#include "../Core/activity.h"
#include "solver_class_factory.h"
#include "dae_array_matrix.h"
#include "ida_la_solver_interface.h"
#include "../config.h"

#if defined(DAE_MPI)
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>
#endif

namespace dae
{
namespace solver
{
/*********************************************************************************************
    daeIDASolver
**********************************************************************************************/
class daeIDASolverData;
class DAE_SOLVER_API daeIDASolver : public daeDAESolver_t
{
public:
    daeIDASolver(void);
    virtual ~daeIDASolver(void);

public:
    virtual void						Initialize(daeBlock_t* pBlock,
                                                   daeLog_t* pLog,
                                                   daeSimulation_t* pSimulation,
                                                   daeeInitialConditionMode eMode,
                                                   bool bCalculateSensitivities,
                                                   const std::vector<size_t>& narrParametersIndexes);
    virtual void						SolveInitial(void);
    virtual real_t						Solve(real_t dTime, daeeStopCriterion eCriterion, bool bReportDataAroundDiscontinuities = true);
    virtual void						SetRelativeTolerance(real_t relTol);
    virtual size_t						GetNumberOfVariables(void) const;
    virtual real_t						GetRelativeTolerance(void) const;
    virtual daeeInitialConditionMode	GetInitialConditionMode(void) const;
    virtual void						SetInitialConditionMode(daeeInitialConditionMode eMode);
    virtual daeBlock_t*					GetBlock(void) const;
    virtual daeLog_t*					GetLog(void) const;
    virtual void						RefreshEquationSetAndRootFunctions(void);
    virtual void						Reinitialize(bool bCopyDataFromBlock, bool bResetSensitivities = false);
    virtual void						Reset(void);
    virtual daeMatrix<real_t>&			GetSensitivities(void);
    virtual std::string					GetName(void) const;
    virtual void                        SetTimeHorizon(real_t timeHorizon);

    virtual void OnCalculateResiduals();
    virtual void OnCalculateConditions() ;
    virtual void OnCalculateJacobian();
    virtual void OnCalculateSensitivityResiduals();

    std::vector<real_t>           GetEstLocalErrors();
    std::vector<real_t>           GetErrWeights();
    std::map<std::string, real_t> GetIntegratorStats();

    daeIDALASolver_t* GetLASolver() const;
    void SetLASolver(daeeIDALASolverType eLASolverType);
    void SetLASolver(daeLASolver_t* pLASolver);

    void SetLastIDASError(const string& strLastError);
    string CreateIDAErrorMessage(int flag);

protected:
    virtual void CreateArrays(void);
    virtual void SetInitialOptions(void);
    virtual void CreateIDA(void);
    virtual void CreateLinearSolver(void);
    virtual void SetupSensitivityCalculation(void);

    void CalculateGradients(void);

    bool CheckFlag(int flag);

    void ResetIDASolver(bool bCopyDataFromBlock, real_t t0, bool bResetSensitivities);

public:
    void SaveMatrixAsXPM(const std::string& strFilename);

public:
    daeeInitialConditionMode			m_eInitialConditionMode;
    void*								m_pIDA;
    daeLog_t*							m_pLog;
    daeBlock_t*							m_pBlock;
    daeSimulation_t*					m_pSimulation;
    daeeIDALASolverType					m_eLASolver;
    daeIDALASolver_t*					m_pLASolver;
    size_t								m_nNumberOfEquations;
    real_t								m_dRelTolerance;
    real_t								m_dSensRelTolerance;
    real_t								m_dSensAbsTolerance;
    real_t								m_timeStart;
    real_t								m_dCurrentTime;
    real_t								m_dTargetTime;
    real_t								m_dNextTimeAfterReinitialization;

    daeRawDataArray<real_t>				m_arrValues;
    daeRawDataArray<real_t>				m_arrTimeDerivatives;
    daeRawDataArray<real_t>				m_arrResiduals;
    daeRawDataArray<real_t>				m_arrRoots;
    daeDenseMatrix						m_matJacobian;

    daeDenseMatrix						m_matSValues;
    daeDenseMatrix						m_matSTimeDerivatives;
    daeDenseMatrix						m_matSResiduals;

    boost::shared_ptr<daeIDASolverData>	m_pIDASolverData;
    bool								m_bIsModelDynamic;
    bool								m_bCalculateSensitivities;
    std::vector<size_t>					m_narrParametersIndexes;
    int									m_eSensitivityMethod;
    std::string							m_strSensitivityMethod;
    bool								m_bErrorControl;
    bool								m_bPrintInfo;
    bool								m_bResetLAMatrixAfterDiscontinuity;
    int                                 m_iNumberOfSTNRebuildsDuringInitialization;
    std::string                         m_strLastIDASError;
};


}
}

#endif
