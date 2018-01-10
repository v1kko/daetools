#ifndef TRILINOS_AMESOS_LA_SOLVER_H
#define TRILINOS_AMESOS_LA_SOLVER_H

#include "../IDAS_DAESolver/ida_la_solver_interface.h"
#include "../IDAS_DAESolver/solver_class_factory.h"
#include "../IDAS_DAESolver/dae_array_matrix.h"
#include <idas/idas.h>
#include <idas/idas_impl.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>
#include <Epetra_SerialComm.h>

#include <Amesos.h>
#include <AztecOO.h>

#include <Epetra_RowMatrix.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Vector.h>
#include <Epetra_LinearProblem.h>
#include <Epetra_Map.h>
#include "EpetraExt_RowMatrixOut.h"
#include <boost/smart_ptr.hpp>
#ifdef HAVE_MPI
#include "mpi.h"
#include <Epetra_MpiComm.h>
#else
#include <Epetra_SerialComm.h>
#endif

#include <Ifpack_ConfigDefs.h>
#include <Ifpack.h>
#include <Ifpack_AdditiveSchwarz.h>
#include <Ifpack_Amesos.h>
#include <Ifpack_ILU.h>
#include <Ifpack_ILUT.h>
#include <Ifpack_IC.h>
#include <Ifpack_ICT.h>
#include <Ifpack_PointRelaxation.h>
#include <Ifpack_BlockRelaxation.h>
#include <ml_include.h>
#include <ml_MultiLevelPreconditioner.h>
#include "base_solvers.h"

namespace dae
{
namespace solver
{
class daeEpetraCSRMatrix : public daeSparseMatrix<double>
{
public:
    daeEpetraCSRMatrix(void)
    {
        indexing   = CSR_C_STYLE;
        rowCounter = 0;
    }

    ~daeEpetraCSRMatrix(void)
    {
    }

public:
    void InitMatrix(size_t n, Epetra_CrsMatrix* m)
    {
        N          = n;
        matrix     = m;
        rowCounter = 0;
    }

    bool GetIndexing(void)
    {
        return indexing;
    }

    void SetIndexing(bool ind)
    {
        indexing = ind;
    }

    double GetItem(size_t i, size_t j) const
    {
        daeDeclareException(exNotImplemented);
        return 0.0;
    }

    void SetItem(size_t i, size_t j, double val)
    {
        if(!matrix)
            daeDeclareException(exInvalidPointer);
        if(i >= N || j >= N)
            daeDeclareException(exOutOfBounds);

        int indices = j;
        double values = val;
        int res = matrix->ReplaceGlobalValues((int)i, 1, &values, &indices);
        if(res != 0)
            daeDeclareException(exInvalidCall);
    }

    void ClearMatrix(void)
    {
        if(!matrix)
            daeDeclareException(exInvalidPointer);
        matrix->PutScalar(0.0);
    }

    size_t GetNrows(void) const
    {
        return N;
    }

    size_t GetNcols(void) const
    {
        return N;
    }

    real_t* GetRow(size_t row)
    {
        daeDeclareAndThrowException(exNotImplemented);
        return NULL;
    }

    real_t* GetColumn(size_t col)
    {
        daeDeclareAndThrowException(exNotImplemented);
        return NULL;
    }

    void AddRow(const std::map<size_t, size_t>& mapIndexes)
    {
        double* values;
        int i, n, *indexes;
        std::map<size_t, size_t>::const_iterator iter;

        if(!matrix)
            daeDeclareException(exInvalidPointer);

        n = mapIndexes.size();
        values  = new double[n];
        indexes = new int[n];

        for(i = 0, iter = mapIndexes.begin(); iter != mapIndexes.end(); i++, iter++)
        {
            values[i]  = 0.0;
            indexes[i] = iter->second;
        }

        int res = matrix->InsertGlobalValues(rowCounter, n, values, indexes);
        if(res != 0)
            daeDeclareException(exInvalidCall);

        delete[] values;
        delete[] indexes;
        rowCounter++;
    }

    void ResetCounters(void)
    {
    // Reset the row counter
        rowCounter = 0;
    }

    void Sort(void)
    {
        if(!matrix)
            daeDeclareException(exInvalidPointer);
        matrix->FillComplete(true);
    }

    void Print(bool bStructureOnly = false) const
    {
        if(!matrix)
            daeDeclareException(exInvalidPointer);
        std::cout << "Epetra CRS Matrix:" << std::endl;
        matrix->Print(std::cout);
    }

    void SaveMatrixAsXPM(const std::string& strFilename)
    {
        size_t i, j;
        std::ofstream of(strFilename.c_str(), std::ios_base::out);
        if(!of.is_open())
            return;

        char* row  = new char[N+1];
        row[N]  = '\0';

        of << "/* XPM */" << std::endl;
        of << "static char *dummy[]={" << std::endl;
        of << "\"" << N << " " << N << " " << 2 << " " << 1 << "\"," << std::endl;
        of << "\"- c #ffffff\"," << std::endl;
        of << "\"X c #000000\"," << std::endl;

        int res, NumEntries;
        double* Values;
        int* Indices;

        for(i = 0; i < N; i++)
        {
            memset(row, '-', N);

            res = matrix->ExtractMyRowView(i, NumEntries, Values, Indices);
            if(res == 0)
            {
                for(j = 0; j < NumEntries; j++)
                    row[ Indices[j] ] = 'X';
            }

            of << "\"" << row << (i == N-1 ? "\"" : "\",") << std::endl;
        }
        of << "};" << std::endl;
        of.close();
        delete[] row;
    }

    void SaveAsMatrixMarketFile(const std::string& strFileName, const std::string& strMatrixName, const std::string& strMatrixDescription)
    {
        EpetraExt::RowMatrixToMatrixMarketFile(strFileName.c_str(), *matrix, strMatrixName.c_str(), strMatrixDescription.c_str());
    }

protected:
    int					N;
    size_t				rowCounter;
    bool				indexing;
    Epetra_CrsMatrix*	matrix;
};

class DAE_SOLVER_API daeTrilinosSolver : public dae::solver::daeIDALASolver_t
{
public:
    daeTrilinosSolver(const std::string& strSolverName, const std::string& strPreconditionerName);
    ~daeTrilinosSolver(void);

    int Create(void* ida, size_t n, daeDAESolver_t* pDAESolver);
    int Reinitialize(void* ida);
    int SaveAsXPM(const std::string& strFileName);
    int SaveAsMatrixMarketFile(const std::string& strFileName, const std::string& strMatrixName, const std::string& strMatrixDescription);
    string GetName(void) const;
    string GetPreconditionerName(void) const;

    int Init(void* ida);
    int Setup(void* ida,
              N_Vector	vectorVariables,
              N_Vector	vectorTimeDerivatives,
              N_Vector	vectorResiduals,
              N_Vector	vectorTemp1,
              N_Vector	vectorTemp2,
              N_Vector	vectorTemp3);
    int Solve(void* ida,
              N_Vector	b,
              N_Vector	weight,
              N_Vector	vectorVariables,
              N_Vector	vectorTimeDerivatives,
              N_Vector	vectorResiduals);
    int Free(void* ida);

    bool SetupLinearProblem(void);
    void PrintPreconditionerInfo(void);

    void SetAmesosOptions(Teuchos::ParameterList& paramList);
    void SetAztecOOOptions(Teuchos::ParameterList& paramList);
    void SetIfpackOptions(Teuchos::ParameterList& paramList);
    void SetMLOptions(Teuchos::ParameterList& paramList);

    Teuchos::ParameterList& GetAmesosOptions(void);
    Teuchos::ParameterList& GetAztecOOOptions(void);
    Teuchos::ParameterList& GetIfpackOptions(void);
    Teuchos::ParameterList& GetMLOptions(void);

    void SetOpenBLASNoThreads(int n);

protected:
    bool CheckData() const;
    void AllocateMemory(void);
    void FreeMemory(void);

public:
    daeBlock_t*	m_pBlock;
    std::string m_strSolverName;

// General Trilinos Solver Data
    boost::shared_ptr<Epetra_LinearProblem>	m_Problem;
    boost::shared_ptr<Epetra_Vector>		m_vecB;
    boost::shared_ptr<Epetra_Vector>		m_vecX;
    boost::shared_ptr<Epetra_Map>			m_map;
    boost::shared_ptr<Epetra_CrsMatrix>		m_matEPETRA;

#ifdef HAVE_MPI
    Epetra_MpiComm		m_Comm;
#else
    Epetra_SerialComm	m_Comm;
#endif

    enum daeeTrilinosSolverType
    {
        eAmesos,
        eAztecOO,
        eAztecOO_Ifpack,
        eAztecOO_ML
    };

    daeeTrilinosSolverType	m_eTrilinosSolver;
    int						m_nNoEquations;
    daeDAESolver_t*			m_pDAESolver;
    daeRawDataArray<real_t>	m_arrValues;
    daeRawDataArray<real_t> m_arrTimeDerivatives;
    daeRawDataArray<real_t> m_arrResiduals;
    daeEpetraCSRMatrix		m_matJacobian;
    size_t					m_nJacobianEvaluations;
    bool					m_bMatrixStructureChanged;

/* AMESOS */
    boost::shared_ptr<Amesos_BaseSolver>	m_pAmesosSolver;
    Teuchos::ParameterList					m_parameterListAmesos;

/* AZTECOO */
    std::string												m_strPreconditionerName;
    boost::shared_ptr<AztecOO>								m_pAztecOOSolver;
    boost::shared_ptr<Ifpack_Preconditioner>				m_pPreconditionerIfpack;
    boost::shared_ptr<ML_Epetra::MultiLevelPreconditioner>	m_pPreconditionerML;
    Teuchos::ParameterList									m_parameterListAztec;
    Teuchos::ParameterList									m_parameterListIfpack;
    Teuchos::ParameterList									m_parameterListML;
    int														m_nNumIters;
    double													m_dTolerance;
    bool													m_bIsPreconditionerConstructed;
};

}
}

#endif

