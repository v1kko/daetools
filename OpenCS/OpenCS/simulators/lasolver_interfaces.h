#ifndef LASOLVER_INTERFACES_H
#define LASOLVER_INTERFACES_H

// For real_t
#include "../cs_model.h"

namespace cs_dae_simulator
{
#define CSR_FORTRAN_STYLE false
#define CSR_C_STYLE       true

enum daeeMatrixAccess
{
    eRowWise = 0,
    eColumnWise
};

template<typename REAL>
class daeArray
{
public:
    virtual ~daeArray(void){}

public:
    virtual REAL   GetItem(size_t i) const       = 0;
    virtual void   SetItem(size_t i, REAL value) = 0;
    virtual size_t GetSize(void) const           = 0;
    virtual REAL*  Data()                        = 0;
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
            throw std::runtime_error("OutOfBounds");
        }
        return data[i];
    }

    void SetItem(size_t i, REAL value)
    {
        if(!data || i >= N)
        {
            throw std::runtime_error("OutOfBounds");
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
            std::cout << std::to_string(data[i]);
        }
        std::cout << "};" << std::endl;
        std::cout.flush();
    }

protected:
    size_t 	N;
    REAL* data;
};

class daeBlockOfEquations_t
{
public:
    virtual ~daeBlockOfEquations_t(){}

public:
    virtual int  CalcNonZeroElements() = 0;
    virtual void FillSparseMatrix(daeSparseMatrix<real_t>* pmatrix) = 0;
    virtual void CalculateJacobian(real_t				time,
                                   real_t				inverseTimeStep,
                                   daeArray<real_t>&	arrValues,
                                   daeArray<real_t>&	arrTimeDerivatives,
                                   daeMatrix<real_t>&	matJacobian) = 0;
};

/*********************************************************************************************
    daeLASolver
**********************************************************************************************/
class daeLASolver_t
{
public:
    virtual ~daeLASolver_t(void){}

public:
    virtual std::string GetName(void) const                                     = 0;
    virtual int Create(size_t n, size_t nnz, daeBlockOfEquations_t* block)      = 0;
    virtual int Reinitialize(size_t nnz)                                        = 0;
    virtual int Init(bool isODE)                                                = 0;
    virtual int Setup(real_t    time,
                      real_t    inverseTimeStep,
                      real_t    jacobianScaleFactor,
                      real_t*	values,
                      real_t*	timeDerivatives)                                = 0;
    virtual int Solve(real_t    time,
                      real_t    inverseTimeStep,
                      real_t    cjratio,
                      real_t*	b,
                      real_t*	weight,
                      real_t*	values,
                      real_t*	timeDerivatives)                                = 0;
    virtual int Free()                                                          = 0;
    virtual int SaveAsXPM(const std::string& strFileName)                       = 0;
    virtual int SaveAsMatrixMarketFile(const std::string& strFileName,
                                       const std::string& strMatrixName,
                                       const std::string& strMatrixDescription) = 0;

    virtual void        SetOption_string(const std::string& strName, const std::string& Value)  = 0;
    virtual void        SetOption_float(const std::string& strName, double Value)               = 0;
    virtual void        SetOption_int(const std::string& strName, int Value)                    = 0;
    virtual void        SetOption_bool(const std::string& strName, bool Value)                  = 0;

    virtual std::string GetOption_string(const std::string& strName)                            = 0;
    virtual double      GetOption_float(const std::string& strName)                             = 0;
    virtual int         GetOption_int(const std::string& strName)                               = 0;
    virtual bool        GetOption_bool(const std::string& strName)                              = 0;
};

}

#endif

