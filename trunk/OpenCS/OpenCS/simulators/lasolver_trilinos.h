#ifndef TRILINOS_AMESOS_LA_SOLVER_H
#define TRILINOS_AMESOS_LA_SOLVER_H

#include "lasolver_interfaces.h"
#include "daesimulator.h"
#include "auxiliary.h"
#include <Epetra_SerialComm.h>

#include <Amesos.h>
#include <AztecOO.h>

#include <Epetra_RowMatrix.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Vector.h>
#include <Epetra_LinearProblem.h>
#include <Epetra_Map.h>
#include "EpetraExt_RowMatrixOut.h"
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
#include <ml_include.h>
#include <ml_MultiLevelPreconditioner.h>

namespace cs_dae_simulator
{
// Trilinos Epetra_CrsMatrix wrapper.
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
        throw std::runtime_error("NotImplemented");
        return 0.0;
    }

    void SetItem(size_t i, size_t j, double val)
    {
        if(!matrix)
            throw std::runtime_error("InvalidPointer");
        if(i >= N || j >= N)
            throw std::runtime_error("OutOfBounds");

        int indices = j;
        double values = val;
        int res = matrix->ReplaceGlobalValues((int)i, 1, &values, &indices);
        if(res != 0)
            throw std::runtime_error("InvalidCall");
    }

    void ClearMatrix(void)
    {
        if(!matrix)
            throw std::runtime_error("InvalidPointer");
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
        throw std::runtime_error("NotImplemented");
        return NULL;
    }

    real_t* GetColumn(size_t col)
    {
        throw std::runtime_error("NotImplemented");
        return NULL;
    }

    void AddRow(const std::map<size_t, size_t>& mapIndexes)
    {
        double* values;
        int i, n, *indexes;
        std::map<size_t, size_t>::const_iterator iter;

        if(!matrix)
            throw std::runtime_error("InvalidPointer");

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
            throw std::runtime_error("InvalidCall");

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
            throw std::runtime_error("InvalidPointer");
        matrix->FillComplete(true);
    }

    void Print(bool bStructureOnly = false) const
    {
        if(!matrix)
            throw std::runtime_error("InvalidPointer");
        std::cout << "Epetra CRS Matrix:" << std::endl;
        matrix->Print(std::cout);
    }

    void SaveMatrixAsXPM(const std::string& strFilename)
    {
    }

    void SaveAsMatrixMarketFile(const std::string& strFileName, const std::string& strMatrixName, const std::string& strMatrixDescription)
    {
    }

protected:
    int					N;
    size_t				rowCounter;
    bool				indexing;
    Epetra_CrsMatrix*	matrix;
};

// DAE Tools implementation (slightly modified) wrapped by daeLinearSolver_Trilinos.
// Important:
//   Must be merged with daeLinearSolver_Trilinos at some point!!
class OPENCS_SIMULATORS_API daeTrilinosSolver : public daeLASolver_t
{
public:
    daeTrilinosSolver(const std::string& strSolverName, const std::string& strPreconditionerName);
    ~daeTrilinosSolver(void);

    virtual int Create(size_t n,
                       size_t nnz,
                       daeBlockOfEquations_t* block);
    virtual int Reinitialize(size_t nnz);
    virtual int Init(bool isODE);
    virtual int Setup(real_t  time,
                      real_t  inverseTimeStep,
                      real_t  jacobianScaleFactor,
                      real_t* pdValues,
                      real_t* pdTimeDerivatives);
    virtual int Solve(real_t  time,
                      real_t  inverseTimeStep,
                      real_t  cjratio,
                      real_t* b,
                      real_t* weight,
                      real_t* pdValues,
                      real_t* pdTimeDerivatives);
    virtual int Free();
    virtual int SaveAsXPM(const std::string& strFileName);
    virtual int SaveAsMatrixMarketFile(const std::string& strFileName, const std::string& strMatrixName, const std::string& strMatrixDescription);

    virtual void        SetOption_string(const std::string& strName, const std::string& Value);
    virtual void        SetOption_float(const std::string& strName, double Value);
    virtual void        SetOption_int(const std::string& strName, int Value);
    virtual void        SetOption_bool(const std::string& strName, bool Value);

    virtual std::string GetOption_string(const std::string& strName);
    virtual double      GetOption_float(const std::string& strName);
    virtual int         GetOption_int(const std::string& strName);
    virtual bool        GetOption_bool(const std::string& strName);

    virtual std::string GetName(void) const;
    std::string GetPreconditionerName(void) const;

    bool SetupLinearProblem(void);
    void PrintPreconditionerInfo(void);

    Teuchos::ParameterList& GetParameterList(void);

protected:
    void FreeMemory(void);

public:
    daeBlockOfEquations_t*	m_pBlock;
    std::string             m_strSolverName;

// General Trilinos Solver Data
    std::shared_ptr<Epetra_LinearProblem>	m_Problem;
    std::shared_ptr<Epetra_Vector>          m_vecB;
    std::shared_ptr<Epetra_Vector>          m_vecX;
    std::shared_ptr<Epetra_Map>             m_map;
    std::shared_ptr<Epetra_CrsMatrix>		m_matEPETRA;
    std::shared_ptr<Epetra_Vector>          m_diagonal;
    bool                                    isODESystem;

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
    daeRawDataArray<real_t>	m_arrValues;
    daeRawDataArray<real_t> m_arrTimeDerivatives;
    daeEpetraCSRMatrix		m_matJacobian;
    bool					m_bMatrixStructureChanged;

    Teuchos::ParameterList	m_parameterList;

    /* AMESOS */
    std::shared_ptr<Amesos_BaseSolver>                      m_pAmesosSolver;

    /* AZTECOO */
    std::string												m_strPreconditionerName;
    std::shared_ptr<AztecOO>								m_pAztecOOSolver;
    std::shared_ptr<Ifpack_Preconditioner>			        m_pPreconditionerIfpack;
    std::shared_ptr<ML_Epetra::MultiLevelPreconditioner>	m_pPreconditionerML;
    int														m_nNumIters;
    double													m_dTolerance;
    bool													m_bIsPreconditionerConstructed;
};

}

#endif

