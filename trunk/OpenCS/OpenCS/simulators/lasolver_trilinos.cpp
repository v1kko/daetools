/***********************************************************************************
                 OpenCS Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#include "daesimulator.h"
#include "lasolver_trilinos.h"

#include <string>
#include <iostream>
#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/lexical_cast/try_lexical_convert.hpp>
using namespace cs_dae_simulator;

namespace cs_dae_simulator
{
static daeLASolver_t* daeCreateTrilinosSolver(const std::string& strSolverName, const std::string& strPreconditionerName)
{
    return new daeTrilinosSolver(strSolverName, strPreconditionerName);
}

static std::vector<std::string> daeTrilinosSupportedSolvers()
{
    std::vector<std::string> strarrSolvers;

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

static std::vector<std::string> daeIfpackSupportedPreconditioners()
{
    std::vector<std::string> strarrPreconditioners;

    Ifpack factory;
    for(int i = 0; i < factory.numPrecTypes; i++)
        strarrPreconditioners.push_back(factory.precTypeNames[i]);

    return strarrPreconditioners;
}

class daeLASolverData : public daeBlockOfEquations_t,
                        public cs::csMatrixAccess_t
{
public:
    daeLASolverData(size_t numVars, cs::csDifferentialEquationModel_t* mod, std::shared_ptr<daeLASolver_t> lasolver) :
        numberOfVariables(numVars), model(mod)
    {
        pmatJacobian = NULL;
        m_pLASolver  = lasolver;
        model->GetSparsityPattern(N, NNZ, IA, JA);
    }

    /* daeBlockOfEquations_t interface */
    virtual int CalcNonZeroElements()
    {
        return NNZ;
    }

    virtual void FillSparseMatrix(daeSparseMatrix<real_t>* pmatrix)
    {
        if(numberOfVariables != N)
            throw std::runtime_error("");
        if(numberOfVariables+1 != IA.size())
            throw std::runtime_error("");
        if(JA.size() != NNZ)
            throw std::runtime_error("");

        std::map<size_t, size_t> mapIndexes;
        for(int row = 0; row < numberOfVariables; row++)
        {
            int colStart = IA[row];
            int colEnd   = IA[row+1];
            mapIndexes.clear();
            for(int col = colStart; col < colEnd; col++)
            {
                size_t bi = JA[col];
                mapIndexes[bi] = bi;
            }
            pmatrix->AddRow(mapIndexes);
            /*
            printf("row %d\n", row);
            for(std::map<size_t, size_t>::iterator it = mapIndexes.begin(); it != mapIndexes.end(); it++)
                printf("{%d,%d} ", it->first, it->second);
            printf("\n");
            */
        }
    }

    virtual void CalculateJacobian(real_t               time,
                                   real_t               inverseTimeStep,
                                   daeArray<real_t>&	arrValues,
                                   daeArray<real_t>&	arrTimeDerivatives,
                                   daeMatrix<real_t>&	matJacobian)
    {
        real_t* values          = arrValues.Data();
        real_t* timeDerivatives = arrTimeDerivatives.Data();

        // Calling SetAndSynchroniseData is not required here since it has previously been called by residuals function.
        pmatJacobian = &matJacobian;

        model->EvaluateJacobian(time, inverseTimeStep, this);

        pmatJacobian = NULL;
    }

    /* csMatrixAccess_t interface */
    void SetItem(size_t row, size_t col, real_t value)
    {
        if(!pmatJacobian)
            throw std::runtime_error("daeLASolverData: matrix pointer not set");
        pmatJacobian->SetItem(row, col, value);
    }

public:
    std::shared_ptr<daeLASolver_t>      m_pLASolver; // not used, only stored here
    daeMatrix<real_t>*                  pmatJacobian;
    int                                 N;
    int                                 NNZ;
    std::vector<int>                    IA;
    std::vector<int>                    JA;
    size_t                              numberOfVariables;
    cs::csDifferentialEquationModel_t*  model;
};

static void SetParameters(daeLASolver_t* lasolver, boost::property_tree::ptree& pt)
{
    BOOST_FOREACH(boost::property_tree::ptree::value_type& pt_child, pt)
    {
        if(pt_child.second.size() == 0) // It is a leaf
        {
            std::cout << "  Set parameter: " << pt_child.first << " = " << pt_child.second.data() << " (";

            bool bValue;
            int iValue;
            double dValue;
            std::string sValue;

            std::string data = pt_child.second.data();
            if(data == "true" || data == "True")
            {
                std::cout << "bool)" << std::endl;
                lasolver->SetOption_bool(pt_child.first, true);
            }
            else if(data == "false" || data == "False")
            {
                std::cout << "bool)" << std::endl;
                lasolver->SetOption_bool(pt_child.first, true);
            }
            else if(boost::conversion::try_lexical_convert<int>(data, iValue))
            {
                std::cout << "integer)" << std::endl;
                lasolver->SetOption_int(pt_child.first, iValue);
            }
            else if(boost::conversion::try_lexical_convert<double>(data, dValue))
            {
                std::cout << "float)" << std::endl;
                lasolver->SetOption_float(pt_child.first, dValue);
            }
            else if(boost::conversion::try_lexical_convert<bool>(data, bValue))
            {
                std::cout << "bool)" << std::endl;
                lasolver->SetOption_bool(pt_child.first, bValue);
            }
            else
            {
                std::cout << "string)" << std::endl;
                sValue = data;
                lasolver->SetOption_string(pt_child.first, sValue);
            }
        }
    }
}


/* daeLinearSolver_Trilinos class */
daeLinearSolver_Trilinos::daeLinearSolver_Trilinos()
{
}

daeLinearSolver_Trilinos::~daeLinearSolver_Trilinos()
{
}

int daeLinearSolver_Trilinos::Initialize(cs::csDifferentialEquationModel_t* model, size_t numberOfVariables, bool isODESystem_)
{
    daeLASolver_t* lasolver = NULL;
    std::shared_ptr<daeLASolver_t> pLASolver;

    daeSimulationOptions& cfg = daeSimulationOptions::GetConfig();
    std::string solverName = cfg.GetString("LinearSolver.Name", "Not specified");
    boost::property_tree::ptree& pt = *((boost::property_tree::ptree*)cfg.ptree);

    if(solverName == "Amesos_Superlu" || solverName == "Amesos_Umfpack" || solverName == "Amesos_Klu")
    {
        lasolver = daeCreateTrilinosSolver(solverName, "");
        pLASolver.reset(lasolver);

        printf("Processing linear solver parameters from '%s' ...\n", cfg.configFile.c_str());
        SetParameters(lasolver, pt.get_child("LinearSolver.Parameters"));
    }
    else if(solverName == "AztecOO")
    {
        std::string preconditionerLibrary = cfg.GetString("LinearSolver.Preconditioner.Library", "Not specified");
        std::string preconditionerName    = cfg.GetString("LinearSolver.Preconditioner.Name", "Not specified");

        if(preconditionerLibrary == "AztecOO") // Native AztecOO preconditioners
        {
            lasolver = daeCreateTrilinosSolver("AztecOO", preconditionerName);
            pLASolver.reset(lasolver);
        }
        else if(preconditionerLibrary == "Ifpack")
        {
            lasolver = daeCreateTrilinosSolver("AztecOO_Ifpack", preconditionerName);
            pLASolver.reset(lasolver);
        }
        else if(preconditionerLibrary == "ML")
        {
            lasolver = daeCreateTrilinosSolver("AztecOO_ML", preconditionerName);
            pLASolver.reset(lasolver);
        }
        else
        {
            throw std::runtime_error("Unsupported preconditioner specified: " + preconditionerName);
        }

        printf("Processing linear solver parameters from '%s' ...\n", cfg.configFile.c_str());
        SetParameters(lasolver, pt.get_child("LinearSolver.Parameters"));

        printf("Processing preconditioner parameters from '%s' ...\n", cfg.configFile.c_str());
        SetParameters(lasolver, pt.get_child("LinearSolver.Preconditioner.Parameters"));
    }
    else
    {
        throw std::runtime_error("Unsupported linear solver specified: " + solverName);
    }

    isODESystem = isODESystem_;

    daeLASolverData* la_data = new daeLASolverData(numberOfVariables, model, pLASolver);
    data = la_data;

    pLASolver->Create(la_data->N, la_data->NNZ, la_data);
    pLASolver->Init(isODESystem);

    return 0;
}

int daeLinearSolver_Trilinos::Setup(real_t  time,
                                    real_t  inverseTimeStep,
                                    real_t  jacobianScaleFactor,
                                    real_t* values,
                                    real_t* timeDerivatives)
{
    daeLASolverData* la_data = (daeLASolverData*)this->data;
    if(!la_data)
        return -1;
    if(!la_data->m_pLASolver)
        return -1;

    int ret = la_data->m_pLASolver->Setup(time,
                                          inverseTimeStep,
                                          jacobianScaleFactor,
                                          values,
                                          timeDerivatives);
    return ret;
}

int daeLinearSolver_Trilinos::Solve(real_t  time,
                                    real_t  inverseTimeStep,
                                    real_t  cjratio,
                                    real_t* b,
                                    real_t* weight,
                                    real_t* values,
                                    real_t* timeDerivatives)
{
    daeLASolverData* la_data = (daeLASolverData*)this->data;
    if(!la_data)
        return -1;
    if(!la_data->m_pLASolver)
        return -1;

    int ret = la_data->m_pLASolver->Solve(time,
                                          inverseTimeStep,
                                          cjratio,
                                          b,
                                          weight,
                                          values,
                                          timeDerivatives);
    return ret;
}

int daeLinearSolver_Trilinos::Free()
{
    daeLASolverData* la_data = (daeLASolverData*)this->data;
    if(!la_data)
        return -1;
    if(!la_data->m_pLASolver)
        return -1;

    int ret = la_data->m_pLASolver->Free();
    return ret;
}


/* DAE Tools implementation (slightly modified) wrapped by daeLinearSolver_Trilinos. */
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

    isODESystem = false;

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
            csThrowException("Error: Ifpack preconditioner: [" + m_strPreconditionerName + "] is not supported!");
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
            csThrowException("Error: ML preconditioner: [" + m_strPreconditionerName + "] is not supported!");
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
            std::string msg = "Error: the solver: [" + m_strSolverName + "] is not supported!" + "\n"
                              + "Supported Amesos solvers: " + "\n"
                              + "   * Amesos_Klu: "         + (Factory.Query("Amesos_Klu")         ? "true" : "false") + "\n"
                              + "   * Amesos_Lapack: "      + (Factory.Query("Amesos_Lapack")      ? "true" : "false") + "\n"
                              + "   * Amesos_Scalapack: "   + (Factory.Query("Amesos_Scalapack")   ? "true" : "false") + "\n"
                              + "   * Amesos_Umfpack: "     + (Factory.Query("Amesos_Umfpack")     ? "true" : "false") + "\n"
                              + "   * Amesos_Pardiso: "     + (Factory.Query("Amesos_Pardiso")     ? "true" : "false") + "\n"
                              + "   * Amesos_Taucs: "       + (Factory.Query("Amesos_Taucs")       ? "true" : "false") + "\n"
                              + "   * Amesos_Superlu: "     + (Factory.Query("Amesos_Superlu")     ? "true" : "false") + "\n"
                              + "   * Amesos_Superludist: " + (Factory.Query("Amesos_Superludist") ? "true" : "false") + "\n"
                              + "   * Amesos_Dscpack: "     + (Factory.Query("Amesos_Dscpack")     ? "true" : "false") + "\n"
                              + "   * Amesos_Mumps: "       + (Factory.Query("Amesos_Mumps")       ? "true" : "false") + "\n"
                              + "Supported AztecOO solvers: " + "\n"
                              + "   * AztecOO (with built-in preconditioners) \n"
                              + "   * AztecOO_Ifpack \n"
                              + "   * AztecOO_ML \n";
            csThrowException(msg);
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
    m_pBlock = block;
    m_nNoEquations = n;

    m_map.reset(new Epetra_Map(m_nNoEquations, 0, m_Comm));
    m_vecB.reset(new Epetra_Vector(*m_map));
    m_vecX.reset(new Epetra_Vector(*m_map));

    return Reinitialize(nnz);
}

int daeTrilinosSolver::Reinitialize(size_t nnz)
{
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

int daeTrilinosSolver::Init(bool isODE)
{
    isODESystem = isODE;
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
        csThrowException("NotImplemented");
    }

    m_parameterList.print(std::cout, 0, true, true);

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
                             real_t  jacobianScaleFactor,
                             real_t* pdValues,
                             real_t* pdTimeDerivatives)
{
    int nResult;

    size_t Neq = m_nNoEquations;

    {
        m_arrValues.InitArray(Neq, pdValues);
        m_arrTimeDerivatives.InitArray(Neq, pdTimeDerivatives);

        m_matJacobian.ClearMatrix();
        m_pBlock->CalculateJacobian(time,
                                    inverseTimeStep,
                                    m_arrValues,
                                    m_arrTimeDerivatives,
                                    m_matJacobian);
        //m_matJacobian.Print();

        // The Jacobian matrix we calculated is the Jacobian for the system of equations.
        // For ODE systems it is the RHS Jacobian (Jrhs) not the full system Jacobian.
        // The Jacobian matrix for ODE systems is in the form: [J] = [I] - gamma * [Jrhs],
        //   where gamma is a scaling factor sent by the ODE solver.
        // Here, the full Jacobian must be calculated.
        if(isODESystem)
        {
            // (1) Scale the Jrhs with the -gamma factor:
            m_matEPETRA->Scale(-jacobianScaleFactor);

            // (2) Extract the diagonal items and replace them with diag[i] = 1 + diag[i]:
            int ret = m_matEPETRA->ExtractDiagonalCopy(*m_diagonal);
            if(ret != 0)
                csThrowException("Failed to extract the diagonal copy of the Jacobian matrix.");

            for(uint32_t i = 0; i < Neq; i++)
                (*m_diagonal)[i] = 1.0 + (*m_diagonal)[i];

            // (3) Put the modified diagonal back to the Epetra matrix.
            ret = m_matEPETRA->ReplaceDiagonalValues(*m_diagonal);
            if(ret != 0)
            {
                printf("ret = %d\n", ret);
                csThrowException("Failed to replace the diagonal items in the Jacobian matrix.");
            }
        }
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
            csThrowException("InvalidCall");

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
            csThrowException("InvalidCall");

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
            csThrowException("InvalidCall");

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
            csThrowException("InvalidCall");

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
                             real_t* pdTimeDerivatives)
{
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
                std::string strError;

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
}
