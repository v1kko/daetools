/***********************************************************************************
                 OpenCS Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License version 3 as published by the Free Software
Foundation. OpenCS is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along with
the OpenCS software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef CS_TRILINOS_SOLVER_INCLUDE_H
#define CS_TRILINOS_SOLVER_INCLUDE_H

#include <string>
#include <iostream>
#include <vector>
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
#include <ml_include.h>
#include <ml_MultiLevelPreconditioner.h>

namespace dae
{
static const std::string  exUnknown				= "Unknown";
static const std::string  exInvalidPointer		= "Invalid pointer";
static const std::string  exInvalidCall			= "Invalid call";
static const std::string  exOutOfBounds			= "Array out of bounds";
static const std::string  exNotImplemented		= "Not implemented";

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

    virtual size_t	GetNrows(void) const = 0;
    virtual size_t	GetNcols(void) const = 0;

    // Depending on matrix type these functions might or might not be implemented.
    // They exist for dense matrices.
    virtual REAL*   GetRow(size_t row)    = 0;
    virtual REAL*   GetColumn(size_t col) = 0;
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
        csThrowException(exNotImplemented);
        return 0.0;
    }

    void SetItem(size_t i, size_t j, double val)
    {
        if(!matrix)
            csThrowException(exInvalidPointer);
        if(i >= N || j >= N)
            csThrowException(exOutOfBounds);

        int indices = j;
        double values = val;
        int res = matrix->ReplaceGlobalValues((int)i, 1, &values, &indices);
        if(res != 0)
            csThrowException(exInvalidCall);
    }

    void ClearMatrix(void)
    {
        if(!matrix)
            csThrowException(exInvalidPointer);
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
        csThrowException(exNotImplemented);
        return NULL;
    }

    real_t* GetColumn(size_t col)
    {
        csThrowException(exNotImplemented);
        return NULL;
    }

    void AddRow(const std::map<size_t, size_t>& mapIndexes)
    {
        double* values;
        int i, n, *indexes;
        std::map<size_t, size_t>::const_iterator iter;

        if(!matrix)
            csThrowException(exInvalidPointer);

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
            csThrowException(exInvalidCall);

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
            csThrowException(exInvalidPointer);
        matrix->FillComplete(true);
    }

    void Print(bool bStructureOnly = false) const
    {
        if(!matrix)
            csThrowException(exInvalidPointer);
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

}
}

#endif
