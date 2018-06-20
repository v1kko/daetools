#include "stdafx.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include "trilinos_amesos_la_solver.h"

using namespace std;

#ifdef DAE_SINGLE_PRECISION
#error Trilinos Amesos LA Solver does not support single precision floating point values
#endif

namespace dae
{
namespace solver
{
daeLASolver_t* daeCreateTrilinosSolver(const std::string& strSolverName, const std::string& strPreconditionerName)
{
    return new daeTrilinosSolver(strSolverName, strPreconditionerName);
}

std::vector<string> daeTrilinosSupportedSolvers()
{
    std::vector<string> strarrSolvers;

    Amesos Factory;
    if(Factory.Query("Amesos_Klu"))
        strarrSolvers.push_back("Amesos_Klu");
    if(Factory.Query("Amesos_Lapack"))
        strarrSolvers.push_back("Amesos_Lapack");
    if(Factory.Query("Amesos_Scalapack"))
        strarrSolvers.push_back("Amesos_Scalapack");
    if(Factory.Query("Amesos_Umfpack"))
        strarrSolvers.push_back("Amesos_Umfpack");
    if(Factory.Query("Amesos_Pardiso"))
        strarrSolvers.push_back("Amesos_Pardiso");
    if(Factory.Query("Amesos_Taucs"))
        strarrSolvers.push_back("Amesos_Taucs");
    if(Factory.Query("Amesos_Superlu"))
        strarrSolvers.push_back("Amesos_Superlu");
    if(Factory.Query("Amesos_Superludist"))
        strarrSolvers.push_back("Amesos_Superludist");
    if(Factory.Query("Amesos_Dscpack"))
        strarrSolvers.push_back("Amesos_Dscpack");
    if(Factory.Query("Amesos_Mumps"))
        strarrSolvers.push_back("Amesos_Mumps");

    strarrSolvers.push_back("AztecOO");
    strarrSolvers.push_back("AztecOO_Ifpack");
    strarrSolvers.push_back("AztecOO_ML");

    return strarrSolvers;
}

std::vector<string> daeIfpackSupportedPreconditioners()
{
    std::vector<string> strarrPreconditioners;

    Ifpack factory;
    for(int i = 0; i < factory.numPrecTypes; i++)
        strarrPreconditioners.push_back(factory.precTypeNames[i]);

    return strarrPreconditioners;
}

daeTrilinosSolver::daeTrilinosSolver(const std::string& strSolverName, const std::string& strPreconditionerName)
#ifdef HAVE_MPI
    : m_Comm(MPI_COMM_WORLD)
#endif
{
    m_pBlock				= NULL;
    m_strSolverName			= strSolverName;
    m_strPreconditionerName = strPreconditionerName;
    m_nNoEquations			= 0;
    m_eTrilinosSolver		= eAmesos;

    m_nNumIters						= 200;
    m_dTolerance					= 1e-3;
    m_bIsPreconditionerConstructed	= false;
    m_bMatrixStructureChanged		= false;

    if(m_strSolverName == "AztecOO")
    {
        m_eTrilinosSolver = eAztecOO;
    }
    else if(m_strSolverName == "AztecOO_Ifpack")
    {
        m_eTrilinosSolver = eAztecOO_Ifpack;

        bool preconditionerSupported = false;
        std::vector<std::string> strarrPreconditioners = daeIfpackSupportedPreconditioners();
        for(size_t i = 0; i < strarrPreconditioners.size(); i++)
        {
            if(m_strPreconditionerName == strarrPreconditioners[i])
            {
                preconditionerSupported = true;
                break;
            }
        }
        if(!preconditionerSupported)
        {
            daeDeclareException(exRuntimeCheck);
            e << "Error: Ifpack preconditioner: [" << m_strPreconditionerName << "] is not supported!";
            throw e;
        }
    }
    else if(m_strSolverName == "AztecOO_ML")
    {
        m_eTrilinosSolver = eAztecOO_ML;

        if(m_strPreconditionerName != "SA"       &&
           m_strPreconditionerName != "DD"       &&
           m_strPreconditionerName != "DD-ML"    &&
           m_strPreconditionerName != "DD-ML-LU" &&
           m_strPreconditionerName != "maxwell"  &&
           m_strPreconditionerName != "NSSA")
        {
            daeDeclareException(exRuntimeCheck);
            e << "Error: ML preconditioner: [" << m_strPreconditionerName << "] is not supported!";
            throw e;
        }

        ML_Epetra::SetDefaults(m_strPreconditionerName, m_parameterList/*m_parameterListML*/);
        m_parameterList.set("reuse: enable", true);
        m_parameterList.set<bool>("ML validate parameter list", false);
    }
    else
    {
        Amesos Factory;
        if(!Factory.Query(m_strSolverName.c_str()))
        {
            daeDeclareException(exRuntimeCheck);
            e << "Error: the solver: [" << m_strSolverName << "] is not supported!" << "\n"
              << "Supported Amesos solvers: " << "\n"
              << "   * Amesos_Klu: "         << (Factory.Query("Amesos_Klu")         ? "true" : "false") << "\n"
              << "   * Amesos_Lapack: "      << (Factory.Query("Amesos_Lapack")      ? "true" : "false") << "\n"
              << "   * Amesos_Scalapack: "   << (Factory.Query("Amesos_Scalapack")   ? "true" : "false") << "\n"
              << "   * Amesos_Umfpack: "     << (Factory.Query("Amesos_Umfpack")     ? "true" : "false") << "\n"
              << "   * Amesos_Pardiso: "     << (Factory.Query("Amesos_Pardiso")     ? "true" : "false") << "\n"
              << "   * Amesos_Taucs: "       << (Factory.Query("Amesos_Taucs")       ? "true" : "false") << "\n"
              << "   * Amesos_Superlu: "     << (Factory.Query("Amesos_Superlu")     ? "true" : "false") << "\n"
              << "   * Amesos_Superludist: " << (Factory.Query("Amesos_Superludist") ? "true" : "false") << "\n"
              << "   * Amesos_Dscpack: "     << (Factory.Query("Amesos_Dscpack")     ? "true" : "false") << "\n"
              << "   * Amesos_Mumps: "       << (Factory.Query("Amesos_Mumps")       ? "true" : "false") << "\n"
              << "Supported AztecOO solvers: " << "\n"
              << "   * AztecOO (with built-in preconditioners) \n"
              << "   * AztecOO_Ifpack \n"
              << "   * AztecOO_ML \n";
            throw e;
        }
        m_eTrilinosSolver = eAmesos;
    }
}

daeTrilinosSolver::~daeTrilinosSolver(void)
{
    FreeMemory();
}

int daeTrilinosSolver::Create(size_t n,
                              size_t nnz,
                              daeBlockOfEquations_t* block)
{
    call_stats::TimerCounter tc(m_stats["Create"]);

    m_pBlock = block;
    m_nNoEquations = n;

    m_map.reset(new Epetra_Map(m_nNoEquations, 0, m_Comm));
    m_vecB.reset(new Epetra_Vector(*m_map));
    m_vecX.reset(new Epetra_Vector(*m_map));

    return Reinitialize(nnz);
}

int daeTrilinosSolver::Reinitialize(size_t nnz)
{
    call_stats::TimerCounter tc(m_stats["Reinitialize"]);

    if(!m_pBlock)
        return -1;

// ACHTUNG, ACHTUNG!!!
// If the matrix structure changes we cannot modify Epetra_Crs_Matrix - only to re-build it!
// Therefore, here we create a new Epetra_CrsMatrix and use it to re-init the Jacobian matrix
    m_matEPETRA.reset(new Epetra_CrsMatrix(Copy, *m_map, 0));

    m_matJacobian.InitMatrix(m_nNoEquations, m_matEPETRA.get());
    m_matJacobian.ResetCounters();
    m_pBlock->FillSparseMatrix(&m_matJacobian);
    m_matJacobian.Sort();
    //m_matJacobian.Print();
    m_bMatrixStructureChanged = true;

    return 0;
}

int daeTrilinosSolver::SaveAsXPM(const std::string& strFileName)
{
    m_matJacobian.SaveMatrixAsXPM(strFileName);
    return 0;
}

int daeTrilinosSolver::SaveAsMatrixMarketFile(const std::string& strFileName, const std::string& strMatrixName, const std::string& strMatrixDescription)
{
    m_matJacobian.SaveAsMatrixMarketFile(strFileName, strMatrixName, strMatrixDescription);
    return 0;
}

std::string daeTrilinosSolver::GetName(void) const
{
    return m_strSolverName + (m_strPreconditionerName.empty() ? "" : " (" + m_strPreconditionerName + ")");
}

std::string daeTrilinosSolver::GetPreconditionerName(void) const
{
    return m_strPreconditionerName;
}

void daeTrilinosSolver::FreeMemory(void)
{
    m_map.reset();
    m_matEPETRA.reset();
    m_vecB.reset();
    m_vecX.reset();
    m_Problem.reset();
    m_pAmesosSolver.reset();
    m_pAztecOOSolver.reset();
    m_pPreconditionerML.reset();
    m_pPreconditionerIfpack.reset();
}

int daeTrilinosSolver::Init()
{
    return 0;
}

int daeTrilinosSolver::Free()
{
    return 0;
}

bool daeTrilinosSolver::SetupLinearProblem(void)
{
    int nResult;

    // At this point we should have the structure of A and the numerical values of A, x and b computed

    m_Problem.reset(new Epetra_LinearProblem(m_matEPETRA.get(), m_vecX.get(), m_vecB.get()));

    if(m_eTrilinosSolver == eAmesos)
    {
        Amesos Factory;
        m_pAmesosSolver.reset(Factory.Create(m_strSolverName.c_str(), *m_Problem));

        m_pAmesosSolver->SetParameters(m_parameterList);

        nResult = m_pAmesosSolver->SymbolicFactorization();
        if(nResult != 0)
        {
            std::cout << "Symbolic factorization failed: " << nResult << std::endl;
            return false;
        }
    }
    else if(m_eTrilinosSolver == eAztecOO)
    {
        m_pAztecOOSolver.reset(new AztecOO(*m_Problem.get()));

        m_pAztecOOSolver->SetParameters(m_parameterList);
    }
    else if(m_eTrilinosSolver == eAztecOO_Ifpack)
    {
        m_pAztecOOSolver.reset(new AztecOO(*m_Problem.get()));

        m_pAztecOOSolver->SetParameters(m_parameterList);

        //Ifpack_Analyze(*m_matEPETRA);

        Ifpack factory;
        Ifpack_Preconditioner* preconditioner = factory.Create(m_strPreconditionerName, m_matEPETRA.get());
        if(preconditioner == NULL)
        {
            std::cout << "Failed to create Ifpack preconditioner " << m_strPreconditionerName << std::endl;
            return false;
        }
        m_pPreconditionerIfpack.reset(preconditioner);
        printf("  Instantiated %s preconditioner (requested: %s)\n", preconditioner->Label(), m_strPreconditionerName.c_str());

        m_pPreconditionerIfpack->SetParameters(m_parameterList /*m_parameterListIfpack*/);
        m_bIsPreconditionerConstructed = false;

        nResult = m_pPreconditionerIfpack->Initialize();
        if(nResult != 0)
        {
            std::cout << "Ifpack preconditioner initialize failed" << std::endl;
            return false;
        }

        nResult = m_pPreconditionerIfpack->Compute();
        if(nResult != 0)
        {
            std::cout << "Ifpack preconditioner compute failed: " << std::endl;
            return -1;
        }

        nResult = m_pAztecOOSolver->SetPrecOperator(m_pPreconditionerIfpack.get());
        if(nResult != 0)
        {
            std::cout << "Failed to set the Ifpack preconditioner" << std::endl;
            return false;
        }

        //m_parameterListIfpack.print(std::cout, 0, true, true);
    }
    else if(m_eTrilinosSolver == eAztecOO_ML)
    {
        m_pAztecOOSolver.reset(new AztecOO(*m_Problem.get()));

        m_pAztecOOSolver->SetParameters(m_parameterList);

        ML_Epetra::MultiLevelPreconditioner* preconditioner = new ML_Epetra::MultiLevelPreconditioner(*m_matEPETRA.get(), m_parameterList/*m_parameterListML*/, true);
        if(preconditioner == NULL)
        {
            std::cout << "Failed to create ML preconditioner " << m_strPreconditionerName << std::endl;
            return false;
        }
        m_pPreconditionerML.reset(preconditioner);
        m_bIsPreconditionerConstructed = false;

        nResult = m_pAztecOOSolver->SetPrecOperator(m_pPreconditionerML.get());
        if(nResult != 0)
        {
            std::cout << "Failed to set ML preconditioner" << std::endl;
            return false;
        }

        //m_parameterListML.print(std::cout, 0, true, true);
    }
    else
    {
        daeDeclareAndThrowException(exNotImplemented);
    }

    //m_parameterList.print(std::cout, 0, true, true);

    m_bMatrixStructureChanged = false;

    return true;
}

void daeTrilinosSolver::PrintPreconditionerInfo(void)
{
    if(m_eTrilinosSolver == eAztecOO)
    {
    }
    else if(m_eTrilinosSolver == eAztecOO_ML)
    {
    }
    else if(m_eTrilinosSolver == eAztecOO_Ifpack)
    {
        std::cout << *m_pPreconditionerIfpack.get() << std::endl;
        //Ifpack_Analyze(*m_matEPETRA.get(), false, 1);
    }
}

int daeTrilinosSolver::Setup(real_t  time,
                             real_t  inverseTimeStep,
                             real_t* pdValues,
                             real_t* pdTimeDerivatives,
                             real_t* pdResiduals)
{
    call_stats::TimerCounter tc(m_stats["Setup"]);

    int nResult;

    size_t Neq = m_nNoEquations;

    {
        call_stats::TimerCounter tcje(m_stats["Jacobian"]);

        m_arrValues.InitArray(Neq, pdValues);
        m_arrTimeDerivatives.InitArray(Neq, pdTimeDerivatives);
        m_arrResiduals.InitArray(Neq, pdResiduals);

        m_matJacobian.ClearMatrix();
        m_pBlock->CalculateJacobian(time,
                                    inverseTimeStep,
                                    m_arrValues,
                                    m_arrResiduals,
                                    m_arrTimeDerivatives,
                                    m_matJacobian);
        //m_matJacobian.Print();
    }

    if(m_bMatrixStructureChanged)
    {
        if(!SetupLinearProblem())
        {
            std::cout << "SetupLinearProblem failed" << std::endl;
            return -1;
        }
    }

    if(m_eTrilinosSolver == eAmesos)
    {
        if(!m_pAmesosSolver)
            daeDeclareAndThrowException(exInvalidCall);

        nResult = m_pAmesosSolver->NumericFactorization();
        if(nResult != 0)
        {
            std::cout << "Amesos: numeric factorization failed: " << nResult << std::endl;
            return -1;
        }
    }
    else if(m_eTrilinosSolver == eAztecOO)
    {
        double condest;
        if(!m_pAztecOOSolver)
            daeDeclareAndThrowException(exInvalidCall);

        m_pAztecOOSolver->DestroyPreconditioner();

        nResult = m_pAztecOOSolver->ConstructPreconditioner(condest);
        printf("    t = %.15f compute preconditioner (condest = %.2e)\n", time, condest);
        if(nResult < 0)
        {
            std::cout << "AztecOO: construct preconditioner failed: " << nResult << std::endl;
            return -1;
        }
    }
    else if(m_eTrilinosSolver == eAztecOO_Ifpack)
    {
        if(!m_pAztecOOSolver || !m_pPreconditionerIfpack)
            daeDeclareAndThrowException(exInvalidCall);

        nResult = m_pPreconditionerIfpack->Compute();
        double condest = m_pPreconditionerIfpack->Condest();
        printf("    t = %.15f compute preconditioner (condest = %.2e)\n", time, condest);
        if(nResult != 0)
        {
            std::cout << "Ifpack: compute preconditioner failed: " << nResult << std::endl;
            return -1;
        }
        //std::cout << *m_pPreconditionerIfpack.get() << std::endl;
        //Ifpack_Analyze(*m_matEPETRA.get(), false, 1);
    }
    else if(m_eTrilinosSolver == eAztecOO_ML)
    {
        int nResult;

        if(!m_pAztecOOSolver || !m_pPreconditionerML)
            daeDeclareAndThrowException(exInvalidCall);

        // Some memory corruption occurs during the tests.
        //m_pPreconditionerML->TestSmoothers();

        if(m_pPreconditionerML->IsPreconditionerComputed())
            m_pPreconditionerML->DestroyPreconditioner();

        nResult = m_pPreconditionerML->ComputePreconditioner();
        if(nResult != 0)
        {
            std::cout << "ML: compute preconditioner failed: " << nResult << std::endl;
            return -1;
        }
    }

    return 0;
}

int daeTrilinosSolver::Solve(real_t  time,
                             real_t  inverseTimeStep,
                             real_t  cjratio,
                             real_t* pdB,
                             real_t* weight,
                             real_t* pdValues,
                             real_t* pdTimeDerivatives,
                             real_t* pdResiduals)
{
    call_stats::TimerCounter tc(m_stats["Solve"]);

    int nResult;
    size_t Neq = m_nNoEquations;

    double *b, *x;
    m_vecB->ExtractView(&b);
    m_vecX->ExtractView(&x);

    ::memcpy(b, pdB, Neq*sizeof(double));

    if(m_eTrilinosSolver == eAmesos)
    {
        nResult = m_pAmesosSolver->Solve();
        if(nResult != 0)
        {
            std::cout << "Amesos: solve failed: " << nResult << std::endl;
            return -1;
        }
    }
    else if(m_eTrilinosSolver == eAztecOO       ||
            m_eTrilinosSolver == eAztecOO_ML    ||
            m_eTrilinosSolver == eAztecOO_Ifpack)
    {
        try
        {
            nResult = m_pAztecOOSolver->Iterate(m_nNumIters, m_dTolerance);
            //printf("    AztecOO niters = %d (%.5f s)\n", m_pAztecOOSolver->NumIters(), m_pAztecOOSolver->SolveTime());

            if(nResult == 1)
            {
                std::cout << "AztecOO solve failed: Max. iterations reached" << std::endl;
                return -1;
            }
            else if(nResult < 0)
            {
                string strError;

                if(nResult == -1)
                    strError = "Aztec status AZ_param: option not implemented";
                else if(nResult == -2)
                    strError = "Aztec status AZ_breakdown: numerical breakdown";
                else if(nResult == -3)
                    strError = "Aztec status AZ_loss: loss of precision";
                else if(nResult == -4)
                    strError = "Aztec status AZ_ill_cond: GMRES hessenberg ill-conditioned";

                std::cout << "AztecOO solve failed: " << strError << std::endl;
                return -1;
            }
        }
        catch(std::exception& e)
        {
            std::cout << "AztecOO solve failed (exception thrown): " << e.what() << std::endl;
            return -1;
        }
        catch(...)
        {
            std::cout << "AztecOO solve failed (unknown exception thrown)" << std::endl;
            return -1;
        }
    }

    ::memcpy(pdB, x, Neq*sizeof(double));
    if(cjratio != 1.0)
    {
        for(size_t i = 0; i < Neq; i++)
            pdB[i] *= 2.0 / (1.0 + cjratio);
    }

    return 0;
}

std::map<std::string, call_stats::TimeAndCount> daeTrilinosSolver::GetCallStats() const
{
    return m_stats;
}

void daeTrilinosSolver::SetOption_string(const std::string& optionName, const std::string& value)
{
    m_parameterList.set<std::string>(optionName, value);
}

void daeTrilinosSolver::SetOption_float(const std::string& optionName, double value)
{
    m_parameterList.set<double>(optionName, value);
}

void daeTrilinosSolver::SetOption_int(const std::string& optionName, int value)
{
    m_parameterList.set<int>(optionName, value);
}

void daeTrilinosSolver::SetOption_bool(const std::string& optionName, bool value)
{
    m_parameterList.set<bool>(optionName, value);
}

std::string daeTrilinosSolver::GetOption_string(const std::string& optionName)
{
    return m_parameterList.get<std::string>(optionName);
}

double daeTrilinosSolver::GetOption_float(const std::string& optionName)
{
    return m_parameterList.get<double>(optionName);
}

int daeTrilinosSolver::GetOption_int(const std::string& optionName)
{
    return m_parameterList.get<int>(optionName);
}

bool daeTrilinosSolver::GetOption_bool(const std::string& optionName)
{
    return m_parameterList.get<bool>(optionName);
}

Teuchos::ParameterList& daeTrilinosSolver::GetParameterList(void)
{
    return m_parameterList;
}

/*
void daeTrilinosSolver::SetOptions(Teuchos::ParameterList& paramList)
{
    m_parameterList = paramList;

    if(m_eTrilinosSolver == eAmesos)
    {
        m_pAmesosSolver->SetParameters(m_parameterList);
    }
    else if(m_eTrilinosSolver == eAztecOO)
    {
        m_pAztecOOSolver->SetParameters(m_parameterList);
    }
    else if(m_eTrilinosSolver == eAztecOO_ML)
    {
        m_pAztecOOSolver->SetParameters(m_parameterList);
        m_pPreconditionerML->SetParameters(m_parameterList);
    }
    else if(m_eTrilinosSolver == eAztecOO_Ifpack)
    {
        m_pAztecOOSolver->SetParameters(m_parameterList);
        m_pPreconditionerIfpack->SetParameters(m_parameterList);
    }
    m_parameterList.print(cout, 0, true, true);
}
*/



//Teuchos::ParameterList& daeTrilinosSolver::GetAmesosOptions(void)
//{
//    return m_parameterList/*m_parameterListAmesos*/;
//}

//Teuchos::ParameterList& daeTrilinosSolver::GetAztecOOOptions(void)
//{
//    return m_parameterList/*m_parameterListAztec*/;
//}

//Teuchos::ParameterList& daeTrilinosSolver::GetIfpackOptions(void)
//{
//    return m_parameterList/*m_parameterListIfpack*/;
//}

//Teuchos::ParameterList& daeTrilinosSolver::GetMLOptions(void)
//{
//    return m_parameterList/*m_parameterListML*/;
//}

//void daeTrilinosSolver::SetAmesosOptions(Teuchos::ParameterList& paramList)
//{
//    //m_parameterListAmesos = paramList;
//    m_parameterList = paramList;
//}

//void daeTrilinosSolver::SetAztecOOOptions(Teuchos::ParameterList& paramList)
//{
//    //m_parameterListAztec = paramList;
//    m_parameterList = paramList;
//    if(m_pAztecOOSolver)
//    {
//        //m_parameterListAztec.print(cout, 0, true, true);
//        m_parameterList.print(cout, 0, true, true);
//        m_pAztecOOSolver->SetParameters(m_parameterList/*m_parameterListAztec*/, true);
//    }
//}

//void daeTrilinosSolver::SetIfpackOptions(Teuchos::ParameterList& paramList)
//{
//    //m_parameterListIfpack = paramList;
//    m_parameterList = paramList;
//    if(m_pPreconditionerIfpack)
//    {
//        //m_parameterListIfpack.print(cout, 0, true, true);
//        m_parameterList.print(cout, 0, true, true);
//        m_pPreconditionerIfpack->SetParameters(m_parameterList/*m_parameterListIfpack*/);
//    }
//}

//void daeTrilinosSolver::SetMLOptions(Teuchos::ParameterList& paramList)
//{
//    //m_parameterListML = paramList;
//    m_parameterList = paramList;
//    if(m_pPreconditionerML)
//    {
//        //m_parameterListML.print(cout, 0, true, true);
//        m_parameterList.print(cout, 0, true, true);
//        m_pPreconditionerML->SetParameterList(m_parameterList/*m_parameterListML*/);
//    }
//}

}
}



