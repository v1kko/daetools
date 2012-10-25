#ifndef DAE_ARRAY_MATRIX_H
#define DAE_ARRAY_MATRIX_H

#include <cstring>
#include <iostream>
#include <fstream>
#include <map>
#include <stdexcept>
#include <algorithm>

/*********************************************************************************************
	daeMatrix
**********************************************************************************************/
#define CSR_FORTRAN_STYLE false
#define CSR_C_STYLE       true

enum daeeMatrixAccess
{
	eRowWise = 0,
	eColumnWise
};

template<typename REAL = real_t> 
class daeMatrix
{
public:
	virtual ~daeMatrix(void){}

public:
	virtual REAL	GetItem(size_t row, size_t col) const       = 0;
	virtual void	SetItem(size_t row, size_t col, REAL value) = 0;
	virtual size_t	GetNrows(void) const                        = 0;
	virtual size_t	GetNcols(void) const                        = 0;
};

/******************************************************************
	daeSparseMatrix
*******************************************************************/
template<typename REAL = real_t> 
class daeSparseMatrix : public daeMatrix<REAL>
{
public:
	virtual ~daeSparseMatrix(void){}
	
public:
// Auxiliary function to build a matrix
	virtual void AddRow(const std::map<size_t, size_t>& mapIndexes) = 0;

// If true then C type, otherwise FORTRAN indexing starting from 1
	virtual bool GetIndexing(void)		 = 0;
	virtual void SetIndexing(bool index) = 0;	
};


/*********************************************************************************************
	daeDenseMatrix
**********************************************************************************************/
class daeDenseMatrix : public daeMatrix<real_t>
{
public:
	daeDenseMatrix(void)
	{
		Nrow        = 0;
		Ncol        = 0;
		data_access = eColumnWise;
		data        = NULL;
	}

	daeDenseMatrix(const daeDenseMatrix& matrix)
	{
		Nrow        = matrix.Nrow;
		Ncol        = matrix.Ncol;
		data_access = matrix.data_access;
		data        = matrix.data;
	}

	virtual ~daeDenseMatrix(void)
	{
	}

public:
// Nrow is the number of rows
// Ncol is the number of columns
// GetItem/GetItem access is always (row, column)
// That internally translates to :
//      [row][col] if eRowWise
//      [col][row] if eColumnWise
	virtual real_t GetItem(size_t row, size_t col) const
	{
		if(!data) 
            throw std::runtime_error("Invalid data");
		if(row >= Nrow || col >= Ncol) 
            throw std::runtime_error("Invalid index");

		if(data_access == eRowWise)
			return data[row][col];
		else
			return data[col][row];
	}

	virtual void SetItem(size_t row, size_t col, real_t value)
	{
		if(!data) 
            throw std::runtime_error("Invalid data");
		if(row >= Nrow || col >= Ncol) 
            throw std::runtime_error("Invalid index");

		if(data_access == eRowWise)
			data[row][col] = value;
		else
			data[col][row] = value;
	}

	virtual size_t GetNrows(void) const
	{
		return Nrow;
	}

	virtual size_t GetNcols(void) const
	{
		return Ncol;
	}

	void InitMatrix(size_t nrows, size_t ncols, real_t** pData, daeeMatrixAccess access)
	{
		Nrow        = nrows;
		Ncol        = ncols;
		data_access = access;
		data        = pData;
	}
	
	void Print(bool bStructureOnly = false) const
	{
		for(size_t row = 0; row < Nrow; row++)
		{
			for(size_t col = 0; col < Ncol; col++)
			{
				if(col != 0)
					std::cout << " ";
				if(bStructureOnly)
					std::cout << (GetItem(row, col) == 0 ? "-" : "X");
				else
					std::cout << GetItem(row, col);
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
		std::cout.flush();
	}
	
	void ClearMatrix(void)
	{
		if(!data) 
            throw std::runtime_error("Invalid data");
		
		if(data_access == eRowWise)
		{
			for(size_t row = 0; row < Nrow; row++)
				for(size_t col = 0; col < Ncol; col++)
					data[row][col] = 0;
		}
		else
		{
			for(size_t row = 0; row < Nrow; row++)
				for(size_t col = 0; col < Ncol; col++)
					data[col][row] = 0;
		}
	}
	
	void SaveMatrixAsXPM(const std::string& strFilename)
	{
		std::ofstream of(strFilename.c_str(), std::ios_base::out);
		if(!of.is_open())
			return;
		
		char* rowdata  = new char[Ncol+1];
		rowdata[Ncol]  = '\0';
	
		of << "/* XPM */" << std::endl;
		of << "static char *dummy[]={" << std::endl;
		of << "\"" << Nrow << " " << Ncol << " " << 2 << " " << 1 << "\"," << std::endl;
		of << "\"- c #ffffff\"," << std::endl;
		of << "\"X c #000000\"," << std::endl;
		
		for(size_t row = 0; row < Nrow; row++)
		{
			memset(rowdata, '-', Ncol);
			for(size_t col = 0; col < Ncol; col++)
			{				
				if(data_access == eRowWise)
				{
					if(data[row][col] != 0)
						rowdata[col] = 'X';
				}
				else
				{
					if(data[col][row] != 0)
						rowdata[col] = 'X';
				}
			}
			of << "\"" << rowdata << (row == Nrow-1 ? "\"" : "\",") << std::endl;
		}
		of << "};" << std::endl;
		of.close();
		delete[] rowdata;
	}
	
public:
	size_t				Nrow;
	size_t				Ncol;
	real_t**			data;
	daeeMatrixAccess	data_access;
};

/*********************************************************************************************
	daeLapackMatrix
**********************************************************************************************/
class daeLapackMatrix : public daeMatrix<real_t>
{
public:
	daeLapackMatrix(void)
	{
		Nrow        = 0;
		Ncol        = 0;
		data_access = eColumnWise;
		data        = NULL;
	}
	virtual ~daeLapackMatrix(void)
	{
	}

public:
// Nrow is the number of rows
// Ncol is the number of columns
// GetItem/GetItem access is always (row, column)
// That internally translates to :
//      [row][col] if eRowWise
//      [col][row] if eColumnWise
	virtual real_t GetItem(size_t row, size_t col) const
	{
		if(!data) 
            throw std::runtime_error("Invalid data");
		if(row >= Nrow || col >= Ncol) 
            throw std::runtime_error("Invalid index");

		if(data_access == eRowWise)
			return data[row*Ncol + col];
		else
			return data[col*Nrow + row];
	}

	virtual void SetItem(size_t row, size_t col, real_t value)
	{
		if(!data) 
            throw std::runtime_error("Invalid data");
		if(row >= Nrow || col >= Ncol) 
            throw std::runtime_error("Invalid index");

		if(data_access == eRowWise)
			data[row*Ncol + col] = value;
		else
			data[col*Nrow + row] = value;
	}

	virtual size_t GetNrows(void) const
	{
		return Nrow;
	}

	virtual size_t GetNcols(void) const
	{
		return Ncol;
	}

	void InitMatrix(size_t nrows, size_t ncols, real_t* pData, daeeMatrixAccess access)
	{
		Nrow        = nrows;
		Ncol        = ncols;
		data_access = access;
		data        = pData;
	}
	
	void Print(bool bStructureOnly = false) const
	{
		for(size_t i = 0; i < Nrow; i++)
		{
			for(size_t k = 0; k < Ncol; k++)
			{
				if(k != 0)
					std::cout << " ";
				if(bStructureOnly)
					std::cout << (GetItem(i, k) == 0 ? "-" : "X");
				else
					std::cout << GetItem(i, k);
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
		std::cout.flush();
	}
	
	void ClearMatrix(void)
	{
		::memset(data, 0, Nrow*Ncol*sizeof(real_t));
	}

	void SaveMatrixAsXPM(const std::string& strFilename)
	{
		std::ofstream of(strFilename.c_str(), std::ios_base::out);
		if(!of.is_open())
			return;
		
		char* rowdata  = new char[Ncol+1];
		rowdata[Ncol]  = '\0';
	
		of << "/* XPM */" << std::endl;
		of << "static char *dummy[]={" << std::endl;
		of << "\"" << Nrow << " " << Ncol << " " << 2 << " " << 1 << "\"," << std::endl;
		of << "\"- c #ffffff\"," << std::endl;
		of << "\"X c #000000\"," << std::endl;
		
		for(size_t i = 0; i < Nrow; i++)
		{
            ::memset(rowdata, '-', Ncol);
			for(size_t k = 0; k < Ncol; k++)
			{				
				if(GetItem(i, k) != 0)
					rowdata[k] = 'X';
			}
			of << "\"" << rowdata << (i == Nrow-1 ? "\"" : "\",") << std::endl;
		}
		of << "};" << std::endl;
		of.close();
		delete[] rowdata;
	}
	
public:
	size_t				Nrow;
	size_t				Ncol;
	real_t*				data;
	daeeMatrixAccess	data_access;
};		

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
	
	void Reset(INT n, INT nnz, bool ind)
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
            throw std::runtime_error("Unable to allocate memory for CSR matrix");
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
		INT index = CalcIndex(i, j);
		if(index < 0)
			return 0.0;
		else
			return A[index];
	}
	
	void SetItem(size_t i, size_t j, real_t val)
	{
		INT index = CalcIndex(i, j);
		if(index < 0)
            throw std::runtime_error("Invalid element in CRS matrix");

        A[index] = val;
	}
	
	size_t GetNrows(void) const
	{
		return N;
	}
	
	size_t GetNcols(void) const
	{
		return N;
	}
		
	INT CalcIndex(size_t i, size_t j) const
	{
		if(indexing == CSR_C_STYLE)
		{
			if(i >= N || j >= N) 
                throw std::runtime_error("Invalid index");
			
		// IA contains number of column indexes in the row i: Ncol_indexes = IA[i+1] - IA[i]
		// Row indexes start at JA[ IA[i] ] and end at JA[ IA[i+1] ]
			for(INT k = IA[i]; k < IA[i+1]; k++)
				if(j == JA[k])
					return k;
		}
		else
		{
			if(i >= N+1 || j >= N+1) 
                throw std::runtime_error("Invalid index");
			
		// FORTRAN arrays use 1-based indexing, therefore we start at IA[i-1] and end at IA[i]
		// JA already contains 1-based indexes, thus no need to adjust it
			for(INT k = IA[i-1]; k < IA[i]; k++)
				if(j == JA[k])
					return k;
		}
		return -1;
	}

	real_t CalculateRatio(void) const
	{
		real_t sparse = 2 * NNZ + N + 1;
		real_t dense  = N * N;
		
		return sparse/dense; 
	}
	
	void Print(bool bStructureOnly = false) const
	{
		INT n, i, k;
		real_t value;
		
		std::cout << "N     = " << N   << std::endl;
		std::cout << "NNZ   = " << NNZ << std::endl;
		std::cout << "Ratio = " << CalculateRatio() << std::endl;
		std::cout << "IA[" << N+1 << "] = {";
		for(i = 0; i < N+1; i++)
		{
			if(i != 0)
				std::cout << ", ";
			std::cout << IA[i];
		}
		std::cout << "};" << std::endl;
	
		std::cout << "JA[" << NNZ << "] = {";
		for(i = 0; i < NNZ; i++)
		{				
			if(i != 0)
				std::cout << ", ";
			std::cout << JA[i];
		}
		std::cout << "};" << std::endl;
	
		std::cout << "A[" << NNZ << "] = {";
		for(i = 0; i < NNZ; i++)
		{
			if(i != 0)
				std::cout << ", ";
			std::cout << A[i];
		}
		std::cout << "};" << std::endl;

		for(i = 0; i < N; i++)
		{
			for(k = 0; k < N; k++)
			{				
				if(indexing == CSR_C_STYLE)
					n = CalcIndex(i, k);
				else
					n = CalcIndex(i+1, k+1);
				
				if(n < 0)
					value = 0.0;
				else
					value = A[n];				
				
				if(k != 0)
					std::cout << " ";
				if(bStructureOnly)
					std::cout << (n < 0 ? "-" : "X");
				else
					std::cout << value;				
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;	

		std::cout.flush();
	}

	void AddRow(const std::map<size_t, size_t>& mapIndexes)
	{
		std::map<size_t, size_t>::const_iterator iter;
		for(iter = mapIndexes.begin(); iter != mapIndexes.end(); iter++)
			SetColumnIndex(static_cast<INT>(iter->second));
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
	
	void SetColumnIndex(INT col)
	{
		if(indexing == CSR_C_STYLE)
			JA[counter] = col;
		else
			JA[counter] = col + 1;
			
		counter++;
	}

	void Sort(void)
	{
		INT i, b, e;

		for(i = 0; i < N; i++)
		{
			b = IA[i];
			e = IA[i+1];
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
				if(indexing == CSR_C_STYLE)
				{
					if(CalcIndex(i, j) >= 0)
						row[j] = 'X';			
				}
				else
				{
					if(CalcIndex(i+1, j+1) >= 0)
						row[j] = 'X';			
				}
			}
			of << "\"" << row << (i == N-1 ? "\"" : "\",") << std::endl;
		}
		of << "};" << std::endl;
		of.close();
		delete[] row;
	}
public:
	INT     NNZ;      // no of non-zero elements
	INT     N;        // matrix size
	FLOAT*  A;        // values
	INT*    IA;       // row indexes data
	INT*    JA;       // column indexes
	bool    indexing; // C style arrays start from 0, FORTRAN from 1
	
protected:
	size_t rowCounter;
	size_t counter;
};

#endif
