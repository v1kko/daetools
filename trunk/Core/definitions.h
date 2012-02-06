/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the 
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef DAE_DEFINITIONS_H
#define DAE_DEFINITIONS_H

#ifdef DAE_SINGLE_PRECISION
#define real_t float 
#else
#define real_t double 
#endif

#include <string.h>
#include <limits.h>
#include <stdio.h>
#include <stdarg.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <list>
#include <bitset>
#include <map>
#include <algorithm>
#include <memory>
#include <iomanip>
#include <typeinfo>

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
#pragma warning(disable: 4250)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#pragma warning(disable: 4100)
#pragma warning(disable: 4101)
#pragma warning(disable: 4189)

#ifdef AddPort
#undef AddPort
#endif

#ifdef GetCurrentTime
#undef GetCurrentTime
#endif

#endif

namespace dae 
{
using std::string;
using std::map;

const string daeAuthorInfo  =	"Dragan Nikolic, 2011 DAE Tools project, dnikolic at daetools.com";
const string daeLicenceInfo	=	"DAE Tools is free software: you can redistribute it and/or modify "
								"it under the terms of the GNU General Public License version 3 "
								"as published by the Free Software Foundation. \n\n"
								"This program is distributed in the hope that it will be useful, "
								"but WITHOUT ANY WARRANTY; without even the implied warranty of "
								"MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the "
								"GNU General Public License for more details. \n\n"
								"You should have received a copy of the GNU General Public License "
								"along with this program. If not, see <http://www.gnu.org/licenses/>.";
const size_t FREE = ULONG_MAX;
inline string daeVersion(bool bIncludeBuild = false)
{
	char dae__version[20];
	if(bIncludeBuild)
		::sprintf(dae__version, "%d.%d.%d", DAE_MAJOR, DAE_MINOR, DAE_BUILD);
	else
		::sprintf(dae__version, "%d.%d", DAE_MAJOR, DAE_MINOR);
	return string(dae__version);
}

inline size_t daeVersionMajor()
{
	return size_t(DAE_MAJOR);
}

inline size_t daeVersionMinor()
{
	return size_t(DAE_MINOR);
}

inline size_t daeVersionBuild()
{
	return size_t(DAE_BUILD);
}


/*********************************************************************************************
	daePtrVector
**********************************************************************************************/
template <class T> 
class daePtrVector : public std::vector<T>
{
public:
	typedef typename daePtrVector<T>::iterator _iterator;

	daePtrVector(void)
	{
		m_bOwnershipOnPointers = true;
	}

	virtual ~daePtrVector(void)
	{
		EmptyAndFreeMemory();
	}

	void SetOwnershipOnPointers(bool bOwnership)
	{
		m_bOwnershipOnPointers = bOwnership;
	}

	bool GetOwnershipOnPointers(void)
	{
		return m_bOwnershipOnPointers;
	}

	void Remove(T object)
	{
		_iterator iter;
		for(iter = this->begin(); iter != this->end(); iter++)
		{
			if(*iter == object)
			{
				if(*iter && m_bOwnershipOnPointers)
					delete *iter;
				this->erase(iter);
				return;
			}
		}
	}
	
	void EmptyAndFreeMemory(void)
	{
		if(m_bOwnershipOnPointers)
		{
			for(_iterator iter = this->begin(); iter != this->end(); iter++)
			{
				if(*iter)
					delete *iter;
			}
		}
		this->clear();
	}

protected:
	bool m_bOwnershipOnPointers;
};

template <class T> 
void clean_vector(daePtrVector<T>& ptrarrVector)
{
	ptrarrVector.EmptyAndFreeMemory();
	daePtrVector<T>().swap(ptrarrVector);
	//std::cout << "after clean_vector capacity = " << ptrarrVector.capacity() << std::endl;
}

/*********************************************************************************************
	daePtrMap
**********************************************************************************************/
template <class KEY, class VALUE>
class daePtrMap : public std::map<KEY,VALUE>
{
public:
	typedef typename daePtrMap<KEY,VALUE>::iterator _iterator;

	daePtrMap(void)
	{
		m_bOwnershipOnPointers = true;
	}

	virtual ~daePtrMap(void)
	{
		EmptyAndFreeMemory();
	}

	void SetOwnershipOnPointers(bool bOwnership)
	{
		m_bOwnershipOnPointers = bOwnership;
	}

	bool GetOwnershipOnPointers(void)
	{
		return m_bOwnershipOnPointers;
	}

	void EmptyAndFreeMemory(void)
	{
		if(m_bOwnershipOnPointers)
		{
			for(_iterator iter = this->begin(); iter != this->end(); iter++)
			{
				if((*iter).second)
					delete (*iter).second;
			}
		}
		this->clear();
	}

protected:
	bool m_bOwnershipOnPointers;
};

/*********************************************************************************************
	daeValueReference
**********************************************************************************************/
class daeValueReference
{
public:
	daeValueReference(void)
	{
		m_pdValue                = NULL;
		m_bOwnershipOnThePointer = true;
	}
	
	daeValueReference(real_t* pdValue, bool bOwnershipOnThePointer)
	{
		if(!pdValue)
			daeDeclareAndThrowException(exInvalidPointer);
		
		m_pdValue                = pdValue;
		m_bOwnershipOnThePointer = bOwnershipOnThePointer;
	}

	virtual ~daeValueReference(void)
	{
		if(m_bOwnershipOnThePointer)
			delete m_pdValue;
		m_pdValue = NULL;
	}

	real_t GetValue(void) const
	{
#ifdef DAE_DEBUG
		if(!m_pdValue)
			daeDeclareAndThrowException(exInvalidPointer);
#endif
		return *m_pdValue;
	}

	real_t* GetPointer(void) const
	{
		return m_pdValue;
	}
	
	void SetValue(real_t value)
	{
#ifdef DAE_DEBUG
		if(!m_pdValue)
			daeDeclareAndThrowException(exInvalidPointer);
#endif
		*m_pdValue = value;
	}
	
	bool GetOwnershipOnThePointer(void)
	{
		return m_bOwnershipOnThePointer;
	}

protected:
	real_t* m_pdValue;
	bool    m_bOwnershipOnThePointer;
};

/*********************************************************************************************
	        Memory/speed aware vector operations
**********************************************************************************************
a) dae_push_back 
   Since the logarithmic resize of a std::vector while items are added with push_back, many times 
   we end-up with the vector of size N and the capacity of N+M (when push_back finds no room for 
   the item to add it calls reserve(2*size) so we may end up with a lot of memory wasted).
   Therefore, we may use generally slower but more memory conservative function that does not 
   reserve memory for items that do not exist (yet).
b) dae_add_vector, dae_set_vector 
   They should be the fastest way to add items to the existing vector and to make two vectors 
   equal (currently they use the built-in std::XXX functions, but if a faster way should be 
   discovered it might be added here). 
**********************************************************************************************/
template<class Storage, class Item>
void dae_push_back(std::vector<Storage>& arrVector, Item item)
{
// Should we turn this off - since for the large vectors reallocation takes too much time? [tamba/lamba]
	arrVector.reserve(arrVector.size()+1);
	arrVector.push_back(item);
}

template<class itemSource, class itemDestination>
void dae_add_vector(const std::vector<itemSource>& arrSource, std::vector<itemDestination>& arrDestination)
{
	arrDestination.reserve(arrDestination.size() + arrSource.size());
	arrDestination.insert(arrDestination.end(), arrSource.begin(), arrSource.end());
}

template<class itemSource, class itemDestination>
void dae_set_vector(const std::vector<itemSource>& arrSource, std::vector<itemDestination>& arrDestination)
{
// resize(n) allocates enough memory to store n elements (if it does not already exist)
// std::copy(where, start, end) is an optimized function for bulk insertions
	arrDestination.resize(arrSource.size());
	std::copy(arrSource.begin(), arrSource.end(), arrDestination.begin());
}

#define dae_capacity_check(Vector) if(Vector.capacity() - Vector.size() != 0) \
    std::cout << std::string(__FILE__) << ":" << std::string(__FUNCTION__) << ":" <<  string(#Vector) << ": " << Vector.capacity() - Vector.size() << std::endl;


/*********************************************************************************************
	daeCreateObjectDelegate
**********************************************************************************************/
template <typename OBJECT>
class daeCreateObjectDelegate
{
public:
	virtual OBJECT* Create(void) = 0;
};

template <typename OBJECT, typename BASE = OBJECT>
class daeCreateObjectDelegateDerived : public daeCreateObjectDelegate<BASE>
{
public:
	virtual BASE* Create(void)
	{
		return new OBJECT;
	}
};

/*********************************************************************************************
	daeObserver
**********************************************************************************************/
template <typename SUBJECT> //SUBJECT is a class which is observed
class daeObserver
{
public:
	virtual ~daeObserver(void) {}
	virtual void Update(SUBJECT *pSubject, void* data) = 0;
};

/*********************************************************************************************
	daeSubject
**********************************************************************************************/
template <typename SUBJECT> //One can Attach only daeObservers which observe class SUBJECT
class daeSubject
{
public:
	virtual ~daeSubject(void) {}

	void Attach(daeObserver<SUBJECT>* pObserver)
	{
		dae_push_back(m_ptrarrObservers, pObserver);
	}

	void Detach(daeObserver<SUBJECT>* pObserver)
	{
		daeObserver<SUBJECT>* observer;
		for(size_t i = 0; i < m_ptrarrObservers.size(); i++)
		{
			observer = m_ptrarrObservers[i];
			if(observer == pObserver)
			{
				m_ptrarrObservers.erase(m_ptrarrObservers.begin()+i);
				return;
			}
		}
	}

	virtual void Notify(void* data)
	{
		daeObserver<SUBJECT>* observer;
		for(size_t i = 0; i < m_ptrarrObservers.size(); i++)
		{
			observer = m_ptrarrObservers[i];
			if(observer)
				observer->Update(dynamic_cast<SUBJECT*>(this), data);
		}
	}

private:
	std::vector<daeObserver<SUBJECT>*> m_ptrarrObservers;
};

/*********************************************************************************************
	daeException
**********************************************************************************************/
static const string  exUnknown				= "Unknown";
static const string  exMiscellanous			= "Miscellanous";
static const string  exInvalidPointer		= "Invalid pointer";
static const string  exInvalidCall			= "Invalid call";
static const string  exOutOfBounds			= "Array out of bounds";
static const string  exIOError				= "IO error";
static const string  exXMLIOError			= "XML IO error";
static const string  exDataReportingError	= "Data reporting error";
static const string  exRuntimeCheck			= "Runtime Check";
static const string  exNotImplemented		= "Not implemented";

#define daeDeclareException(TYPE)	          dae::daeException e(std::string(TYPE), std::string(__FUNCTION__), std::string(__FILE__), __LINE__);
#define daeDeclareAndThrowException(TYPE)	{ dae::daeException e(std::string(TYPE), std::string(__FUNCTION__), std::string(__FILE__), __LINE__); throw e; }

class daeException : public std::exception
{
public:
	daeException(const string& strExceptionType, const string& strFunction, const string& strFile, int iLine)
	{
		m_strExceptionType	= strExceptionType;
		m_strFile			= strFile;
		m_nLine				= iLine;
		m_strFunction		= strFunction;

		std::stringstream ss;
		ss << m_strExceptionType << " exception in function: ";
		ss << m_strFunction << ", source file: ";
		ss << m_strFile << ", line: ";
		ss << m_nLine << std::endl;
		m_strWhat = ss.str();
	}

	virtual ~daeException(void) throw()
	{
	}

	virtual const char* what(void) const throw()
	{
		return m_strWhat.c_str();
	}

	template<class TYPE>
	daeException& operator << (TYPE value)
	{
		std::stringstream ss;
		ss << value;
		m_strWhat += ss.str();
		return *this;
	}

public:
	string				m_strExceptionType;
	string				m_strFunction;
	string				m_strFile;
	int					m_nLine;
	string				m_strWhat;
};

/*********************************************************************************************
	daeArray
**********************************************************************************************/
template<typename REAL = real_t> 
class daeArray
{
public:
	daeArray(void)
	{
		N    = 0;
		data = NULL;
	}
	virtual ~daeArray(void)
	{
	}

public:
	REAL& operator [](size_t i)
	{
#ifdef DAE_DEBUG
		if(!data || i >= N) 
			daeDeclareAndThrowException(exOutOfBounds);
#endif
		return data[i];
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
#ifdef DAE_DEBUG
		if(!data) 
			daeDeclareAndThrowException(exInvalidPointer);
#endif
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


}

#endif
