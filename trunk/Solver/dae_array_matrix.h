#ifndef DAE_ARRAY_MATRIX_H
#define DAE_ARRAY_MATRIX_H

#include <stdio.h>
#include <typeinfo>
#include <iostream>
#include <fstream>
#include "../Core/helpers.h"
#include "solver_class_factory.h"
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
using namespace boost::numeric;

#ifdef HAS_GNU_GSL
#include <gsl/gsl_linalg.h>
#endif

#ifdef HAS_TRILINOS
#include <Amesos_ConfigDefs.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_MultiVector.h>
#include <Epetra_SerialDenseVector.h>
#include <Epetra_SerialDenseMatrix.h>
#include <Epetra_CrsMatrix.h>
#endif

namespace dae
{
namespace solver
{
/*********************************************************************************************
	daeDenseArray
**********************************************************************************************/
class DAE_SOLVER_API daeDenseArray : public daeArray<real_t>
{
public:
	daeDenseArray(void)
	{
		N = 0;
		data = NULL;
	}
	virtual ~daeDenseArray(void)
	{
	}

public:
	virtual real_t GetItem(size_t i) const
	{
		if(!data || i >= N) 
			daeDeclareAndThrowException(exOutOfBounds);
		return data[i];
	}

	virtual void SetItem(size_t i, real_t value)
	{
		if(!data || i >= N) 
			daeDeclareAndThrowException(exOutOfBounds);
		data[i] = value;
	}

	virtual size_t GetSize(void) const
	{
		return N;
	}

	void InitArray(size_t n, real_t* pData)
	{
		N = n;
		data = pData;
	}
	
	void Print(bool bStructureOnly = false)
	{
		for(size_t i = 0; i < N; i++)
		{
			if(i != 0)
				std::cout << ", ";
			if(bStructureOnly)
				std::cout << (GetItem(i) == 0 ? "-" : "X");
			else
				std::cout << dae::toStringFormatted(GetItem(i), 12, 5);
		}
		std::cout << std::endl;
		std::cout.flush();
	}

protected:
	size_t 	N;
	real_t* data;
};

/*********************************************************************************************
	daeDenseMatrix
**********************************************************************************************/
class DAE_SOLVER_API daeDenseMatrix : public daeMatrix<real_t>
{
public:
	daeDenseMatrix(void)
	{
		N           = 0;
		data_access = eColumnWise;
		data        = NULL;
	}
	virtual ~daeDenseMatrix(void)
	{
	}

public:
	virtual real_t GetItem(size_t i, size_t j) const
	{
		if(!data || i >= N || j >= N) 
			daeDeclareAndThrowException(exOutOfBounds);

		if(data_access == eRowWise)
			return data[i][j];
		else
			return data[j][i];
	}

	virtual void SetItem(size_t i, size_t j, real_t value)
	{
		if(!data || i >= N || j >= N) 
			daeDeclareAndThrowException(exOutOfBounds);

		if(data_access == eRowWise)
			data[i][j] = value;
		else
			data[j][i] = value;
	}

	virtual size_t GetSizeN(void) const
	{
		return N;
	}

	virtual size_t GetSizeM(void) const
	{
		return N;
	}

	void InitMatrix(size_t n, real_t** pData, daeeMatrixAccess access)
	{
		N = n;
		data_access = access;
		data = pData;
	}
	
	void Print(bool bStructureOnly = false)
	{
		for(size_t i = 0; i < N; i++)
		{
			for(size_t k = 0; k < N; k++)
			{
				if(k != 0)
					std::cout << " ";
				if(bStructureOnly)
					std::cout << (GetItem(i, k) == 0 ? "-" : "X");
				else
					std::cout << dae::toStringFormatted(GetItem(i, k), 12, 5);
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
		std::cout.flush();
	}
	
	void ClearMatrix(void)
	{
		for(size_t i = 0; i < N; i++)
			for(size_t k = 0; k < N; k++)
				data[i][k] = 0;
	}

public:
	size_t				N;
	real_t**			data;
	daeeMatrixAccess	data_access;
};

/*********************************************************************************************
	daeUBLASMatrix
**********************************************************************************************/
//typedef boost::numeric::ublas::vector<real_t>			    ublasArray;
//typedef boost::numeric::ublas::compressed_matrix<real_t>	ublasMatrix;
//
//class DAE_SOLVER_API daeUBLASMatrix : public daeMatrix<real_t>
//{
//public:
//	daeUBLASMatrix(void)
//	{
//		m = NULL;
//		N = 0;
//	}
//	virtual ~daeUBLASMatrix(void)
//	{
//	}
//
//public:
//	virtual real_t GetItem(size_t i, size_t j) const
//	{
//		if(!m || i >= N || j >= N) 
//			daeDeclareAndThrowException(exOutOfBounds);
//
//		return (*m)(i,j);
//	}
//
//	virtual void SetItem(size_t i, size_t j, real_t value)
//	{
//		if(!m || i >= N || j >= N) 
//			daeDeclareAndThrowException(exOutOfBounds);
//
//		(*m)(i,j) = value;
//	}
//
//	virtual size_t GetSizeN(void) const
//	{
//		return N;
//	}
//
//	virtual size_t GetSizeM(void) const
//	{
//		return N;
//	}
//
//	void InitMatrix(size_t n, ublasMatrix* matrix)
//	{
//		N = n;
//		m = matrix;
//	}
//
//	void Clear(void)
//	{
//		(*m).clear();
//	}
//
//	void Print(bool bStructureOnly = false)
//	{
//		std::cout << "Jacobian; Capacity = " << (*m).nnz_capacity() << ", Filled = " << (*m).nnz() << std::endl;
//		for(size_t i = 0; i < N; i++)
//		{
//			for(size_t k = 0; k < N; k++)
//			{
//				if(k != 0)
//					std::cout << " ";
//				if(bStructureOnly)
//					std::cout << ((*m)(i, k) == 0 ? "-" : "X");
//				else
//					std::cout << dae::toStringFormatted((*m)(i, k), 12, 5);
//			}
//			std::cout << std::endl;
//		}
//		std::cout << std::endl;
//		std::cout.flush();
//	}
//
//public:
//	size_t		        N;
//	ublasMatrix*		m;
//	daeeMatrixAccess    data_access;
//};

/*********************************************************************************************
	daeGNUGSLMatrix
**********************************************************************************************/
#ifdef HAS_GNU_GSL

class DAE_SOLVER_API daeGNUGSLMatrix : public daeMatrix<real_t>
{
public:
	daeGNUGSLMatrix(void)
	{
		N = 0;
		matrix = NULL;
	}
	virtual ~daeGNUGSLMatrix(void)
	{
	}

public:
	virtual real_t GetItem(size_t i, size_t j) const
	{
		if(!matrix || i >= N || j >= N) 
			daeDeclareAndThrowException(exOutOfBounds);

		return gsl_matrix_get(matrix, i, j);
	}

	virtual void SetItem(size_t i, size_t j, real_t value)
	{
		if(!matrix || i >= N || j >= N) 
			daeDeclareAndThrowException(exOutOfBounds);

		gsl_matrix_set(matrix, i, j, value);
	}

	virtual size_t GetSizeN(void) const
	{
		return N;
	}

	virtual size_t GetSizeM(void) const
	{
		return N;
	}

	void InitMatrix(size_t n, gsl_matrix* m)
	{
		N = n;
		matrix = m;
	}
	
	void Print(bool bStructureOnly = false)
	{
		for(size_t i = 0; i < N; i++)
		{
			for(size_t k = 0; k < N; k++)
			{
				if(k != 0)
					std::cout << " ";
				if(bStructureOnly)
					std::cout << (GetItem(i, k) == 0 ? "-" : "X");
				else
					std::cout << dae::toStringFormatted(GetItem(i, k), 12, 5);
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
		std::cout.flush();
	}

public:
	size_t		N;
	gsl_matrix*	matrix;
};

#endif

inline bool CompareMatrices(daeMatrix<real_t>& left, daeMatrix<real_t>& right)
{
	size_t N = left.GetSizeN();
	
	for(size_t i = 0; i < N; i++)
		for(size_t k = 0; k < N; k++)
			if(left.GetItem(i, k) != right.GetItem(i, k))
				return false;

	return true;
}

/*********************************************************************************************
	daeLapackMatrix
**********************************************************************************************/
class DAE_SOLVER_API daeLapackMatrix : public daeMatrix<real_t>
{
public:
	daeLapackMatrix(void)
	{
		N           = 0;
		data_access = eColumnWise;
		data        = NULL;
	}
	virtual ~daeLapackMatrix(void)
	{
	}

public:
	virtual real_t GetItem(size_t i, size_t j) const
	{
		if(!data || i >= N || j >= N) 
			daeDeclareAndThrowException(exOutOfBounds);

		if(data_access == eRowWise)
			return data[i*N + j];
		else
			return data[j*N + i];
	}

	virtual void SetItem(size_t i, size_t j, real_t value)
	{
		if(!data || i >= N || j >= N) 
			daeDeclareAndThrowException(exOutOfBounds);

		if(data_access == eRowWise)
			data[i*N + j] = value;
		else
			data[j*N + i] = value;
	}

	virtual size_t GetSizeN(void) const
	{
		return N;
	}

	virtual size_t GetSizeM(void) const
	{
		return N;
	}

	void InitMatrix(size_t n, real_t* pData, daeeMatrixAccess access)
	{
		N           = n;
		data_access = access;
		data        = pData;
	}
	
	void Print(bool bStructureOnly = false)
	{
		for(size_t i = 0; i < N; i++)
		{
			for(size_t k = 0; k < N; k++)
			{
				if(k != 0)
					std::cout << " ";
				if(bStructureOnly)
					std::cout << (GetItem(i, k) == 0 ? "-" : "X");
				else
					std::cout << dae::toStringFormatted(GetItem(i, k), 12, 5);
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
		std::cout.flush();
	}
	
	void ClearMatrix(void)
	{
		::memset(data, 0, N*N*sizeof(real_t));
	}
	
public:
	size_t				N;
	real_t*				data;
	daeeMatrixAccess	data_access;
};		

/*********************************************************************************************
	daeEpetraDenseMatrix
**********************************************************************************************/
#ifdef HAS_TRILINOS

// Access is column-major
class DAE_SOLVER_API daeEpetraDenseMatrix : public daeMatrix<double>
{
public:
	daeEpetraDenseMatrix(void)
	{
		N      = 0;
		matrix = NULL;
	}
	virtual ~daeEpetraDenseMatrix(void)
	{
	}

public:
	virtual real_t GetItem(size_t i, size_t j) const
	{
		if(!matrix || i >= N || j >= N) 
			daeDeclareAndThrowException(exOutOfBounds);

		return (*matrix)(i, j);
	}

	virtual void SetItem(size_t i, size_t j, real_t value)
	{
		if(!matrix || i >= N || j >= N) 
			daeDeclareAndThrowException(exOutOfBounds);

		(*matrix)(i, j) = value;
	}

	virtual size_t GetSizeN(void) const
	{
		return N;
	}

	virtual size_t GetSizeM(void) const
	{
		return N;
	}

	void InitMatrix(size_t n, Epetra_SerialDenseMatrix* m)
	{
		N = n;
		matrix = m;
	}
	
	void ClearMatrix(void)
	{
		::memset(matrix->A(), 0, N*N*sizeof(double));
		//for(size_t i = 0; i < N; i++)
		//	for(size_t k = 0; k < N; k++)
		//		(*matrix)(i,j) = 0;
	}

	void Print(bool bStructureOnly = false)
	{
		for(size_t i = 0; i < N; i++)
		{
			for(size_t k = 0; k < N; k++)
			{
				if(k != 0)
					std::cout << " ";
				if(bStructureOnly)
					std::cout << (GetItem(i, k) == 0 ? "-" : "X");
				else
					std::cout << dae::toStringFormatted(GetItem(i, k), 12, 5);
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
		std::cout.flush();
	}

public:
	size_t						N;
	Epetra_SerialDenseMatrix*	matrix;
};

#endif

/*********************************************************************************************
	daeCSRMatrix
**********************************************************************************************/
template<typename FLOAT, typename INT>
class daeCSRMatrix : public daeSparseMatrix<FLOAT>
{
public:
	daeCSRMatrix(void)
	{
		N        = 0;
		NNZ      = 0;
		A        = NULL;
		IA       = NULL;
		JA       = NULL;
		indexing = CSR_C_STYLE;
	}
	
	daeCSRMatrix(int n, int nnz, bool ind)
	{
		N        = 0;
		NNZ      = 0;
		A        = NULL;
		IA       = NULL;
		JA       = NULL;
		indexing = ind;
	
		Reset(n, nnz, ind);
	}
	
	~daeCSRMatrix(void)
	{
		Free();
	}
	
public:
	bool GetIndexing(void)
	{
		return indexing;
	}
	
	void SetIndexing(bool ind)
	{
		indexing = ind;
	}
	
	void Reset(int n, int nnz, bool ind)
	{
		N        = n;
		NNZ      = nnz;
		indexing = ind;
		
	// If the pointers are NULL then it will allocate as in malloc
	// Otherwise it will realloc the memory to the new size
		A  = (FLOAT*)realloc(A,   NNZ  * sizeof(FLOAT));
		IA = (INT*)  realloc(IA, (N+1) * sizeof(INT));
		JA = (INT*)  realloc(JA,  NNZ  * sizeof(INT));
		
		if(!A || !IA || !JA)
		{
			Free();
			daeDeclareException(exMiscellanous);
			e << "Unable to allocate memory for daeCSRMatrix";
			throw e;
		}

		memset(A,  0,  NNZ  * sizeof(FLOAT));
		memset(IA, 0, (N+1) * sizeof(INT));
		memset(JA, 0,  NNZ  * sizeof(INT));
		
	// The last item in IA is the number of NNZ	
		if(indexing == CSR_C_STYLE)
			IA[N] = NNZ;
		else
			IA[N] = NNZ + 1;
	}
	
	void ClearValues(void)
	{
		memset(A,  0,  NNZ  * sizeof(FLOAT));
	}

	void Free(void)
	{
		if(A)
			free(A);
		if(IA)
			free(IA);
		if(JA)
			free(JA);
		
		N   = 0;
		NNZ = 0;
		A   = NULL;
		IA  = NULL;
		JA  = NULL;
	}
	
	real_t GetItem(size_t i, size_t j) const
	{
		int index = CalcIndex(i, j);
		if(index < 0)
			return 0.0;
		else
			return A[index];
	}
	
	void SetItem(size_t i, size_t j, real_t val)
	{
		temp = CalcIndex(i, j);
		if(temp < 0)
		{
			daeDeclareException(exMiscellanous);
			e << "Invalid element in CRS matrix: (" << i << ", " << j << ")";
			throw e;
		}
		A[temp] = val;
	}
	
	size_t GetSizeN(void) const
	{
		return N;
	}
	
	size_t GetSizeM(void) const
	{
		return N;
	}
		
	int CalcIndex(size_t i, size_t j) const
	{
		if(i >= N || j >= N) 
			daeDeclareException(exOutOfBounds);
	
		if(indexing == CSR_C_STYLE)
		{
			for(int k = IA[i]; k < IA[i+1]; k++)
				if(j == JA[k])
					return k;
		}
		else
		{
			j = j + 1; // FORTRAN arrays use 1 based indexing
			for(int k = IA[i]-1; k < IA[i+1]-1; k++)
				if(j == JA[k])
					return k;
		}
		return -1;
	}

	real_t CalculateRatio() const
	{
		real_t sparse = 2 * NNZ + N + 1;
		real_t dense  = N * N;
		
		return sparse/dense; 
	}
	
	void Print(bool bStructureOnly = false, bool bPrintValues = false) const
	{
		int n, i, k;
		real_t value;
		
		std::cout << "N     = " << N   << std::endl;
		std::cout << "NNZ   = " << NNZ << std::endl;
		std::cout << "Ratio = " << CalculateRatio() << std::endl;
		std::cout << "IA:" << std::endl;
		for(i = 0; i < N+1; i++)
		{
			if(k != 0)
				std::cout << ", ";
			std::cout << IA[i];
		}
		std::cout << std::endl;
	
		std::cout << "JA:" << std::endl;
		for(i = 0; i < NNZ; i++)
		{				
			if(k != 0)
				std::cout << ", ";
			std::cout << JA[i];
		}
		std::cout << std::endl;
	
		if(bPrintValues)
		{
			std::cout << "A:" << std::endl;
			for(i = 0; i < NNZ; i++)
			{
				if(k != 0)
					std::cout << ", ";
				std::cout << A[i];
			}
			std::cout << std::endl;

			for(i = 0; i < N; i++)
			{
				for(k = 0; k < N; k++)
				{				
					n = CalcIndex(i, k);
					if(n < 0)
						value = 0.0;
					else
						value = A[n];				
					
					if(k != 0)
						std::cout << " ";
					if(bStructureOnly)
						std::cout << (n < 0 ? "-" : "X");
					else
						std::cout << dae::toStringFormatted(value, 12, 5);				
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;	
		}
		std::cout.flush();
	}
	
	void AddRow(const std::map<size_t, size_t>& mapIndexes)
	{
		std::map<size_t, size_t>::const_iterator iter;
		for(iter = mapIndexes.begin(); iter != mapIndexes.end(); iter++)
			SetColumnIndex(iter->second);
		NextRow();
	}

	void ResetCounters(void)
	{
	// Reset the row counter	
		rowCounter = 0;

	// Reset column indexes counter
		counter = 0;

	// Indexes in the 1st row always start with the 1st element
		if(indexing == CSR_C_STYLE)
			IA[0] = 0; 
		else
			IA[0] = 1;
	}
	
	void NextRow(void)
	{
		rowCounter++;
		if(indexing == CSR_C_STYLE)
			IA[rowCounter] = counter;
		else
			IA[rowCounter] = counter + 1;
	}
	
	void SetColumnIndex(size_t col)
	{
		if(indexing == CSR_C_STYLE)
			JA[counter] = col;
		else
			JA[counter] = col + 1;
			
		counter++;
	}

	void Sort(void)
	{
		int i, b, e;

		for(i = 0; i < N; i++)
		{
			if(indexing == CSR_C_STYLE)
			{
				b = IA[i];
				e = IA[i+1];
			}
			else
			{
				b = IA[i]   - 1;
				e = IA[i+1] - 1;
			}
			std::sort(&JA[b], &JA[e]);
		}
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
		
		for(i = 0; i < N; i++)
		{
			memset(row, '-', N);
			for(j = 0; j < N; j++)
			{				
				if(CalcIndex(i, j) >= 0)
					row[j] = 'X';			
			}
			of << "\"" << row << (i == N-1 ? "\"" : "\",") << std::endl;
		}
		of << "};" << std::endl;
		of.close();
		delete[] row;
	}
	
	void SaveMatrixAsXPM2(const std::string& strFilename)
	{
		size_t i, j;
		std::ofstream of(strFilename.c_str(), std::ios_base::out);
		if(!of.is_open())
			return;
		
		char* row  = new char[N+1];
		row[N]  = '\0';
	
		of << "! XPM2" << std::endl;
		of << N << " " << N << " " << 2 << " " << 1 << std::endl;
		of << "- c #ffffff" << std::endl;
		of << "X c #000000" << std::endl;
		
		for(i = 0; i < N; i++)
		{
			memset(row, '-', N);
			for(j = 0; j < N; j++)
			{				
				if(CalcIndex(i, j) >= 0)
					row[j] = 'X';			
			}
			of << row << std::endl;
		}
		of.close();
		delete[] row;
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
		
		for(i = 0; i < N; i++)
		{
			for(j = 0; j < N; j++)
				row[2*j+1] = '0';
	
			for(j = 0; j < N; j++)
			{				
				if(CalcIndex(i, j) >= 0)
					row[2*j+1] = '1';			
			}
			of << row << std::endl;			
		}
		of.close();
		delete[] row;
	}
	
public:
	int     NNZ;      // no of non-zero elements
	int     N;        // matrix size
	FLOAT*  A;        // values
	INT*    IA;       // row indexes data
	INT*    JA;       // column indexes
	bool    indexing; // C style arrays start from 0, FORTRAN from 1
	
protected:
	int    temp;
	size_t rowCounter;
	size_t counter;
};

#ifdef HAS_TRILINOS

class daeEpetraCSRMatrix : public daeSparseMatrix<real_t>
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

	real_t GetItem(size_t i, size_t j) const
	{
		daeDeclareException(exNotImplemented);
		if(!matrix)
			daeDeclareException(exInvalidPointer);
		if(i >= N || j >= N) 
			daeDeclareException(exOutOfBounds);

//		int indices = j;
//		double values;
//		matrix->ExtractGlobalRowCopy((int)i, 1, &values, &indices);
//		return values;
		return 0.0;
	}
	
	void SetItem(size_t i, size_t j, real_t val)
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
			values[i]  = 1.0;
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
	
protected:
	int					N;
	size_t				rowCounter;
	Epetra_CrsMatrix*	matrix;
	bool				indexing; // C style arrays start from 0, FORTRAN from 1
};

#endif

}
}

#endif
