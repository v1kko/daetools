#ifndef DAE_ARRAY_MATRIX_H
#define DAE_ARRAY_MATRIX_H

#include <stdio.h>
#include <typeinfo>
#include <iostream>
#include <fstream>
#include "../Core/helpers.h"
#include "solver_class_factory.h"
extern "C"
{
#include "../mmio.h"
}

namespace dae
{
namespace solver
{
template<typename REAL = real_t>
class daeRawDataArray : public daeArray<REAL>
{
public:
    daeRawDataArray(size_t n = 0, REAL* pData = NULL)
    {
        N    = n;
        data = pData;
    }

    virtual ~daeRawDataArray(void)
    {
    }

public:
    REAL GetItem(size_t i) const
    {
        if(!data || i >= N)
        {
            daeDeclareException(exOutOfBounds);
            e << "Invalid index in daeArray: " << i;
            throw e;
        }
        return data[i];
    }

    void SetItem(size_t i, REAL value)
    {
        if(!data || i >= N)
        {
            daeDeclareException(exOutOfBounds);
            e << "Invalid index in daeArray: " << i;
            throw e;
        }
        data[i] = value;
    }

    size_t GetSize(void) const
    {
        return N;
    }

    void InitArray(size_t n, REAL* pData)
    {
        N    = n;
        data = pData;
    }

    REAL* Data()
    {
        return data;
    }

    void Print(void) const
    {
        std::cout << "vector[" << N << "] = {";
        for(size_t i = 0; i < N; i++)
        {
            if(i != 0)
                std::cout << ", ";
            std::cout << data[i];
        }
        std::cout << "};" << std::endl;
        std::cout.flush();
    }

protected:
    size_t 	N;
    REAL* data;
};

/*********************************************************************************************
    daeDenseMatrix
**********************************************************************************************/
class DAE_SOLVER_API daeDenseMatrix : public daeMatrix<real_t>
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
            daeDeclareAndThrowException(exInvalidPointer);
        if(row >= Nrow || col >= Ncol)
            daeDeclareAndThrowException(exOutOfBounds);

        if(data_access == eRowWise)
            return data[row][col];
        else
            return data[col][row];
    }

    virtual void SetItem(size_t row, size_t col, real_t value)
    {
        if(!data)
            daeDeclareAndThrowException(exInvalidPointer);
        if(row >= Nrow || col >= Ncol)
            daeDeclareAndThrowException(exOutOfBounds);

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
                    std::cout << dae::toStringFormatted(GetItem(row, col), 15, 8, true);
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout.flush();
    }

    void ClearMatrix(void)
    {
        if(!data)
            daeDeclareAndThrowException(exInvalidPointer);

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

    void SaveAsMatrixMarketFile(const std::string& strFileName, const std::string& strMatrixName, const std::string& strMatrixDescription)
    {
        int n;
        size_t row, col;
        FILE* mmx = fopen(strFileName.c_str(), "w");
        if(!mmx)
        {
            daeDeclareException(exMiscellanous);
            e << "Unable to open " << strFileName << " file";
            throw e;
        }

        MM_typecode matcode;
        mm_initialize_typecode(&matcode);
        mm_set_matrix(&matcode);
        mm_set_coordinate(&matcode);
        mm_set_real(&matcode);

        mm_write_banner(mmx, matcode);
        fprintf(mmx, "%% \n");
        fprintf(mmx, "%% DAE Tools Project <www.daetools.com>\n");
        fprintf(mmx, "%% %s\n", strMatrixName.c_str());
        fprintf(mmx, "%% %s\n", strMatrixDescription.c_str());
        fprintf(mmx, "%% \n");
        mm_write_mtx_crd_size(mmx, Nrow, Ncol, Nrow * Ncol);

        /* NOTE: matrix market files use 1-based indices */
        for(row = 0; row < Nrow; row++)
        {
            for(col = 0; col < Ncol; col++)
            {
                if(data_access == eRowWise)
                    fprintf(mmx, "%d %d %17.10e\n", int(row+1), int(col+1), data[row][col]);
                else
                    fprintf(mmx, "%d %d %17.10e\n", int(row+1), int(col+1), data[col][row]);
            }
            fflush(mmx);
        }

        fclose(mmx);
    }

public:
    size_t				Nrow;
    size_t				Ncol;
    real_t**			data;
    daeeMatrixAccess	data_access;
};

inline bool CompareMatrices(daeMatrix<real_t>& left, daeMatrix<real_t>& right)
{
    size_t lNrow = left.GetNrows();
    size_t lNcol = left.GetNcols();
    size_t rNrow = right.GetNrows();
    size_t rNcol = right.GetNcols();

    if(lNrow != rNrow)
        return false;
    if(lNcol != rNcol)
        return false;

    for(size_t i = 0; i < lNrow; i++)
        for(size_t k = 0; k < lNcol; k++)
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
            daeDeclareAndThrowException(exInvalidPointer);
        if(row >= Nrow || col >= Ncol)
            daeDeclareAndThrowException(exOutOfBounds);

        if(data_access == eRowWise)
            return data[row*Ncol + col];
        else
            return data[col*Nrow + row];
    }

    virtual void SetItem(size_t row, size_t col, real_t value)
    {
        if(!data)
            daeDeclareAndThrowException(exInvalidPointer);
        if(row >= Nrow || col >= Ncol)
            daeDeclareAndThrowException(exOutOfBounds);

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
                    std::cout << dae::toStringFormatted(GetItem(i, k), 15, 8, true);
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
            memset(rowdata, '-', Ncol);
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

    void SaveAsMatrixMarketFile(const std::string& strFileName, const std::string& strMatrixName, const std::string& strMatrixDescription)
    {
        int n;
        size_t i, k;
        FILE* mmx = fopen(strFileName.c_str(), "w");
        if(!mmx)
        {
            daeDeclareException(exMiscellanous);
            e << "Unable to open " << strFileName << " file";
            throw e;
        }

        MM_typecode matcode;
        mm_initialize_typecode(&matcode);
        mm_set_matrix(&matcode);
        mm_set_coordinate(&matcode);
        mm_set_real(&matcode);

        mm_write_banner(mmx, matcode);
        fprintf(mmx, "%% \n");
        fprintf(mmx, "%% DAE Tools Project <www.daetools.com>\n");
        fprintf(mmx, "%% %s\n", strMatrixName.c_str());
        fprintf(mmx, "%% %s\n", strMatrixDescription.c_str());
        fprintf(mmx, "%% \n");
        mm_write_mtx_crd_size(mmx, Nrow, Ncol, Nrow*Ncol);

        /* NOTE: matrix market files use 1-based indices */
        for(i = 0; i < Nrow; i++)
        {
            for(k = 0; k < Ncol; k++)
            {
                fprintf(mmx, "%d %d %17.10e\n", int(i+1), int(k+1), GetItem(i, k));
            }
            fflush(mmx);
        }
        fclose(mmx);
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

    const FLOAT& operator()(size_t i, size_t j) const
    {
        INT index = CalcIndex(i, j);
        if(index < 0)
        {
            daeDeclareException(exOutOfBounds);
            e << "Invalid element in CRS matrix: (" << i << ", " << j << ")";
            throw e;
        }
        return A[index];
    }

    FLOAT& operator()(size_t i, size_t j)
    {
        INT index = CalcIndex(i, j);
        if(index < 0)
        {
            daeDeclareException(exOutOfBounds);
            e << "Invalid element in CRS matrix: (" << i << ", " << j << ")";
            throw e;
        }
        return A[index];
    }

    FLOAT GetItem(size_t i, size_t j) const
    {
        INT index = CalcIndex(i, j);
        if(index < 0)
        {
            daeDeclareException(exOutOfBounds);
            e << "Invalid element in CRS matrix: (" << i << ", " << j << ")";
            throw e;
        }
        return A[index];
    }

    void SetItem(size_t i, size_t j, FLOAT val)
    {
        INT index = CalcIndex(i, j);
        if(index < 0)
        {
            daeDeclareException(exOutOfBounds);
            e << "Invalid element in CRS matrix: (" << i << ", " << j << ")";
            throw e;
        }
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
        // i, j are always 0-based

        if(indexing == CSR_C_STYLE)
        {
            if(i >= N || j >= N)
                daeDeclareException(exOutOfBounds);

        // IA contains number of column indexes in the row i: Ncol_indexes = IA[i+1] - IA[i]
        // Row indexes start at JA[ IA[i] ] and end at JA[ IA[i+1] ]
            for(INT k = IA[i]; k < IA[i+1]; k++)
                if(j == JA[k])
                    return k;
        }
        else
        {
            if(i >= N || j >= N)
                daeDeclareException(exOutOfBounds);

        // IA and JA contain 1-based indexes
            for(INT k = IA[i]; k < IA[i+1]; k++) // k is 1-based now
                if(j+1 == JA[k-1])
                    return k-1;
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
        FLOAT value;

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
                    std::cout << dae::toStringFormatted(value, 15, 8, true);
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

//    void SavePQMatrixAsXPM(const std::string& strFilename)
//    {
//        size_t i, j;
//        std::ofstream of(strFilename.c_str(), std::ios_base::out);
//        if(!of.is_open())
//            return;

//        char* row  = new char[N+1];
//        row[N]  = '\0';

//        of << "/* XPM */" << std::endl;
//        of << "static char *dummy[]={" << std::endl;
//        of << "\"" << N << " " << N << " " << 2 << " " << 1 << "\"," << std::endl;
//        of << "\"- c #ffffff\"," << std::endl;
//        of << "\"X c #000000\"," << std::endl;

//        for(i = 0; i < N; i++)
//        {
//            memset(row, '-', N);
//            for(j = 0; j < N; j++)
//            {
//                if(CalcIndex(P[i], Q[j]) >= 0)
//                    row[j] = 'X';
//            }
//            of << "\"" << row << (i == N-1 ? "\"" : "\",") << std::endl;
//            of.flush();
//        }
//        of << "};" << std::endl;
//        of.close();
//        delete[] row;
//    }

//    void SaveBTFMatrixAsXPM(const std::string& strFilename)
//    {
//        size_t i, j;
//        std::ofstream of(strFilename.c_str(), std::ios_base::out);
//        if(!of.is_open())
//            return;

//        if(!BTF)
//            return;

//        char* row  = new char[N+1];
//        row[N]  = '\0';

//        of << "/* XPM */" << std::endl;
//        of << "static char *dummy[]={" << std::endl;
//        of << "\"" << N << " " << N << " " << 2 << " " << 1 << "\"," << std::endl;
//        of << "\"- c #ffffff\"," << std::endl;
//        of << "\"X c #000000\"," << std::endl;

//        for(i = 0; i < N; i++)
//        {
//            memset(row, '-', N);
//            for(j = 0; j < N; j++)
//            {
//                if(CalcIndex(BTF[i], j) >= 0)
//                    row[j] = 'X';
//            }
//            of << "\"" << row << (i == N-1 ? "\"" : "\",") << std::endl;
//            of.flush();
//        }
//        of << "};" << std::endl;
//        of.close();
//        delete[] row;
//    }

//    FLOAT GetBTFItem(INT i) const
//    {
//        INT index = CalcIndex(BTF[i], BTF[i]);
//        std::cout << "BTF[" << i << "] = " << BTF[i] << std::endl;
//        if(index >= 0)
//            return A[index];
//        else
//            daeDeclareAndThrowException(exInvalidCall);
//        return 0.0;
//    }

//    void SetBlockTriangularForm(const INT* btf)
//    {
//        BTF = (INT*)realloc(BTF, N*sizeof(INT));
//        memcpy(BTF, btf, N * sizeof(INT));
//    }

    void SaveAsMatrixMarketFile(const std::string& strFileName, const std::string& strMatrixName, const std::string& strMatrixDescription)
    {
        int n;
        size_t i, j;
        FILE* mmx = fopen(strFileName.c_str(), "w");
        if(!mmx)
        {
            daeDeclareException(exMiscellanous);
            e << "Unable to open " << strFileName << " file";
            throw e;
        }

        MM_typecode matcode;
        mm_initialize_typecode(&matcode);
        mm_set_matrix(&matcode);
        mm_set_coordinate(&matcode);
        mm_set_real(&matcode);

        mm_write_banner(mmx, matcode);
        fprintf(mmx, "%% \n");
        fprintf(mmx, "%% DAE Tools Project <www.daetools.com>\n");
        fprintf(mmx, "%% %s\n", strMatrixName.c_str());
        fprintf(mmx, "%% %s\n", strMatrixDescription.c_str());
        fprintf(mmx, "%% \n");
        mm_write_mtx_crd_size(mmx, N, N, NNZ);

        /* NOTE: matrix market files use 1-based indices */
        for(i = 0; i < N; i++)
        {
            for(j = 0; j < N; j++)
            {
                n = CalcIndex(i, j);

                if(n >= 0)
                    fprintf(mmx, "%d %d %17.10e\n", int(i+1), int(j+1), A[n]);
            }
            fflush(mmx);
        }

        fclose(mmx);
    }

public:
    INT     NNZ;      // no of non-zero elements
    INT     N;        // matrix size
    FLOAT*  A;        // values
    INT*    IA;       // row indexes data
    INT*    JA;       // column indexes
    bool    indexing; // C style arrays start from 0, FORTRAN from 1

    //INT*    BTF;    // block triangular form
    //INT*    P;      // row permutation form
    //INT*    Q;      // column permutation form

protected:
    size_t rowCounter;
    size_t counter;
};

}
}

#endif
