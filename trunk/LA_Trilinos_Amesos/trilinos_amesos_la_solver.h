#ifndef TRILINOS_AMESOS_LA_SOLVER_H
#define TRILINOS_AMESOS_LA_SOLVER_H

#include "../Solver/ida_la_solver_interface.h"
#include "../Solver/solver_class_factory.h"
#include "../Solver/dae_array_matrix.h"
#include <idas/idas.h>
#include <idas/idas_impl.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>
#include <Epetra_SerialComm.h>
#include <Amesos.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Vector.h>
#include <Epetra_LinearProblem.h>
#include <Epetra_Map.h>
#include <boost/smart_ptr.hpp>
#ifdef HAVE_MPI
#include "mpi.h"
#include <Epetra_MpiComm.h>
#else
#include <Epetra_SerialComm.h>
#endif
//#include <AztecOO.h>
//#include <AztecOO_StatusTestMaxIters.h>
//#include <AztecOO_StatusTestResNorm.h>
//#include <AztecOO_StatusTestCombo.h>

namespace dae
{
namespace solver
{
daeIDALASolver_t* daeCreateTrilinosAmesosSolver(const std::string& strSolverName);
std::vector<string> daeTrilinosAmesosSupportedSolvers(void);

class daeEpetraCSRMatrix : public daeSparseMatrix<double>
{
public:
	daeEpetraCSRMatrix(void)
	{
		indexing = CSR_C_STYLE;
	}
	
	~daeEpetraCSRMatrix(void)
	{
	}
	
public:
	void InitMatrix(size_t n, Epetra_CrsMatrix* m)
	{
		N = n;
		matrix = m;
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
		matrix->ReplaceGlobalValues((int)i, 1, &values, &indices);
	}
	
	void ClearMatrix(void)
	{
		if(!matrix)
			daeDeclareException(exInvalidPointer);
		matrix->PutScalar(0.0);
	}

	size_t GetSizeN(void) const
	{
		return N;
	}
	
	size_t GetSizeM(void) const
	{
		return N;
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
		
		matrix->InsertGlobalValues(rowCounter, n, values, indexes);
		
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
	
	void Print(void) const
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
	
	void SaveMatrixAsXPM2(const std::string& strFilename)
	{
	}

	void SaveMatrixAsPBM(const std::string& strFilename)
	{
		size_t i, j;
		std::ofstream of(strFilename.c_str(), std::ios_base::out);
		if(!of.is_open())
			return;
		
		char* row  = new char[2*N+1];
		row[2*N] = '\0';
		memset(row, ' ', 2*N);
		
		of << "P1" << std::endl;
		of << N << " " << N << " " << std::endl;
		
		int res, NumEntries;
		double* Values;
		int* Indices;

		for(i = 0; i < N; i++)
		{
			for(j = 0; j < N; j++)
				row[2*j+1] = '0';
	
			res = matrix->ExtractMyRowView(i, NumEntries, Values, Indices);
			if(res == 0)
			{
				for(j = 0; j < NumEntries; j++)
					row[ 2*Indices[j] + 1] = '1';			
			}
			
			of << row << std::endl;			
		}
		of.close();
		delete[] row;
	}

protected:
	int					N;
	size_t				rowCounter;
	Epetra_CrsMatrix*	matrix;
	bool				indexing;
};

class DAE_SOLVER_API daeTrilinosAmesosSolver : public dae::solver::daeIDALASolver_t
{
public:
	daeTrilinosAmesosSolver(const std::string& strSolverName);
	~daeTrilinosAmesosSolver(void);
	
	int Create(void* ida, size_t n, daeDAESolver_t* pDAESolver);
	int Reinitialize(void* ida);
	int SaveAsPBM(const std::string& strFileName);
	int SaveAsXPM(const std::string& strFileName);
	
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
	
protected:
	bool CheckData() const;
	void AllocateMemory(void);
	void FreeMemory(void);

public:
	daeBlock_t*	m_pBlock;
	std::string m_strSolverName;

// Amesos Solver Data
	boost::shared_ptr<Epetra_LinearProblem>	m_Problem;
	boost::shared_ptr<Epetra_Vector>		m_vecB;
	boost::shared_ptr<Epetra_Vector>		m_vecX;
	boost::shared_ptr<Epetra_Map>			m_map;
	boost::shared_ptr<Epetra_CrsMatrix>		m_matEPETRA;

	boost::shared_ptr<Amesos_BaseSolver>	m_pSolver;
//	boost::shared_ptr<AztecOO>				m_pSolver;
	
#ifdef HAVE_MPI
	Epetra_MpiComm			m_Comm;
#else
	Epetra_SerialComm		m_Comm;
#endif

	int						m_nNoEquations;
	daeDAESolver_t*			m_pDAESolver;
	daeDenseArray			m_arrValues;
	daeDenseArray			m_arrTimeDerivatives;
	daeDenseArray			m_arrResiduals;
	daeEpetraCSRMatrix		m_matJacobian;	
	size_t					m_nJacobianEvaluations;
};

}
}

#endif

