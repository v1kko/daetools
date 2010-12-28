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
#include <bitset>
#include <map>
#include <algorithm>
#include <memory>
#include <iomanip>
#include <typeinfo>

namespace dae 
{
using std::string;

const string daeAuthorInfo  =	"Dragan Nikolic, 2009 DAE Tools project, dnikolic at daetools.com";
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
inline string daeVersion(bool bGetRevision = false)
{
	char dae__version[20];
	if(bGetRevision)
		::sprintf(dae__version, "%d.%d-%d", DAE_MAJOR, DAE_MINOR, DAE_BUILD);
	else
		::sprintf(dae__version, "%d.%d", DAE_MAJOR, DAE_MINOR);
	return string(dae__version);
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
	virtual void Update(SUBJECT *pSubject) = 0;
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
		m_ptrarrObservers.push_back(pObserver);
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

	void Notify(void)
	{
		daeObserver<SUBJECT>* observer;
		for(size_t i = 0; i < m_ptrarrObservers.size(); i++)
		{
			observer = m_ptrarrObservers[i];
			if(observer)
				observer->Update((SUBJECT*)this);
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
	virtual ~daeArray(void){}

public:
	virtual REAL	GetItem(size_t i) const			= 0;
	virtual void	SetItem(size_t i, REAL value)	= 0;
	virtual size_t	GetSize(void) const				= 0;
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
	virtual bool GetIndexing(void)				= 0;
	virtual void SetIndexing(bool index)		= 0;	
};


}

#endif
