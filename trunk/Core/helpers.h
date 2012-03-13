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
#ifndef DAE_HELPERS_H
#define DAE_HELPERS_H

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <map>
#include <stdexcept>
#include <stdlib.h>
#include <time.h>

#ifdef __MACH__
#include <mach/mach_time.h>
#endif

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)

#ifndef _MSC_VER
#define _MSC_VER
#endif

// Modify the following defines if you have to target a platform prior to the ones specified below.
// Refer to MSDN for the latest info on corresponding values for different platforms.
#ifndef WINVER				// Allow use of features specific to Windows XP or later.
#define WINVER 0x0501		// Change this to the appropriate value to target other versions of Windows.
#endif

#ifndef _WIN32_WINNT		// Allow use of features specific to Windows XP or later.
#define _WIN32_WINNT 0x0501	// Change this to the appropriate value to target other versions of Windows.
#endif

#ifndef _WIN32_IE			// Allow use of features specific to IE 6.0 or later.
#define _WIN32_IE 0x0600	// Change this to the appropriate value to target other versions of IE.
#endif

#define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers

#define _SCL_SECURE_NO_WARNINGS

#include <windows.h>

#pragma warning(disable: 4251)
#pragma warning(disable: 4275)

#endif

namespace dae 
{
#define g_cstrTrue   "true"
#define g_cstrFalse  "false"

void HexToString(const unsigned char* szData, size_t nLength, std::string& strHex);
void StringToHex(const std::string& strHex, unsigned char** szData, size_t& nLength); 

bool	 IsInteger(std::string& strValue);
bool	 IsDecimal(std::string& strValue);

std::vector<std::string>  ParseString(const std::string& strSource, char   cDelimiter = ';');
std::vector<std::string>  ParseString(const std::string& strSource, std::string strDelimiter);

void		MakeUpper(std::string& strSource);
void		MakeLower(std::string& strSource);
std::string	Replace(const std::string& strSource,    char cFind = ' ', char cReplace = '_');
std::string	ReplaceAll(const std::string& strSource, char cFind = ' ', char cReplace = '_');
void		RemoveAll(std::string& strSource, const std::string& strFind);
void		RemoveAllNonAlphaNumericCharacters(std::string& strSource);
inline void RemoveAllNonAlphaNumericCharacters(std::vector<std::string>& strarrSource);
void		LTrim(std::string& strSource, char cTrim = ' ');
void		RTrim(std::string& strSource, char cTrim = ' ');

std::string	Split(std::vector<std::string>& Array,  const char* lpszDelimiter = ", ");
std::string	SplitAndBracket(std::vector<std::string>& Array,  const char* lpszDelimiter = ", ");
void		Enclose(std::string& strToEnclose, char cLeft, char cRight);
void		Enclose(std::string& strToEnclose, const char* lpszLeft = "[", const char* lpszRight = "]");
bool		ParseSingleToken(std::string& strFullName, std::string& strShortName, std::vector<size_t>& narrDomains);

double GetTimeInSeconds(void);

		
template<class T>
T fromString(const std::string& value)
{
	T returnValue;
	std::stringstream ss(value);
	if(!(ss >> returnValue))
	{
		std::string strError = "Runtime error: cannot convert [";
		strError        += value;
		strError        += "]";
		std::invalid_argument e(strError.c_str());
		throw e;
	}
	return returnValue;
}

template<class T>
std::string toString(const T value, std::streamsize width = -1)
{
	std::stringstream ss;
	
	if(width != -1)
		ss << std::setw(width);
	
	if(!(ss << value))
	{
		std::string strError = "Runtime error: cannot convert number to string";
		std::invalid_argument e(strError.c_str());
		throw e;
	}
	return ss.str();
}

template<class T>
std::string toStringFormatted(T value, 
							  std::streamsize width = -1, 
							  std::streamsize precision = -1, 
							  bool scientific = false,
							  bool strip_if_integer = false)
{
	std::stringstream ss;
	
	if(scientific)
		ss << std::setiosflags(std::ios_base::scientific);
	else
		ss << std::setiosflags(std::ios_base::fixed);
	
	if(width != -1)
		ss << std::setw(width);
	
	if(precision != -1)
		ss << std::setprecision(precision);
	
	if(strip_if_integer && value == (T)(int)(value))
	{
		if(!(ss << (int)(value)))
		{
			std::string strError = "Runtime error: cannot convert number to formatted string";
			std::invalid_argument e(strError.c_str());
			throw e;
		}
	}
	else
	{
		if(!(ss << value))
		{
			std::string strError = "Runtime error: cannot convert number to formatted string";
			std::invalid_argument e(strError.c_str());
			throw e;
		}
	}
	
	return ss.str();
}

template<class T>
std::string getClassName(const T& Object, bool bStripNamespaces = false)
{
	size_t found;
	std::string strClass = typeid(Object).name();
	if(!bStripNamespaces)
		found = strClass.rfind(" ");
	else
		found = strClass.rfind(":");
	if(found != std::string::npos)
		return strClass.substr(found+1);
	else
		return strClass;
}

template<typename T>
std::string getClassName(bool bStripNamespaces = false)
{
	size_t found;
	std::string strClass = typeid(T).name();
	if(!bStripNamespaces)
		found = strClass.rfind(" ");
	else
		found = strClass.rfind(":");
	if(found != std::string::npos)
		return strClass.substr(found+1);
	else
		return strClass;
}

// T must be a pointer
template<class T>
std::vector<T> makeVector(T a1, T a2, T a3, T a4, T a5, T a6, T a7, T a8)
{
	std::vector<T> std_array;
	T array[8] = {a1, a2, a3, a4, a5, a6, a7, a8};
	
	for(size_t i = 0; i < 8; i++)
	{
		if(array[i])
			std_array.push_back(array[i]);
	}
	return std_array;
}

template<class T>
std::string toString(const std::vector<T>& std_array, const std::string& strDelimiter = std::string(", "))
{
	std::string result;
	for(size_t i = 0; i < std_array.size(); i++)
	{
		if(i != 0)
			result += strDelimiter;
		result += toString<T>(std_array[i]);
	}
	return result;
}

template<class T1, class T2>
std::string toString(const std::map<T1, T2>& std_map, const std::string& strDelimiter = std::string(", "))
{
	typedef typename std::map<T1,T2>::const_iterator _iterator;
	
	std::string result;
	_iterator citer;
	
	for(citer = std_map.begin(); citer != std_map.end(); citer++)
	{
		if(citer != std_map.begin())
			result += strDelimiter;
		result += "(" + toString<T1>(citer->first) + ", " + toString<T2>(citer->second) + ")";
	}
	return result;
}

// T must be a pointer-type for a daeObject-derived object
template<class T>
std::string toStringFormatted(const std::vector<T>& std_array, 
							  std::streamsize width = -1, 
							  std::streamsize precision = -1, 
							  bool scientific = false,
							  bool strip_if_integer = false,
                              const std::string& strDelimiter = std::string(", "))
{
	std::string result;
	for(size_t i = 0; i < std_array.size(); i++)
	{
		if(i != 0)
			result += strDelimiter;
		result += toStringFormatted<T>(std_array[i], width, precision, scientific, strip_if_integer);
	}
	return result;
}

// T must be a pointer-type for a daeObject-derived object
template<class T>
std::string toString_Names(const std::vector<T>& std_array, 
                           const std::string& strPrependToEachObject = std::string(""), 
                           const std::string& strDelimiter = std::string(", "))
{
	std::string result;
	for(size_t i = 0; i < std_array.size(); i++)
	{
		if(i != 0)
			result += strDelimiter;
		result += strPrependToEachObject + std_array[i]->GetName();
	}
	return result;
}

template<class T>
std::string toString_StrippedNames(const std::vector<T>& std_array, 
                                   const std::string& strPrependToEachObject = std::string(""), 
                                   const std::string& strDelimiter = std::string(", "))
{
	std::string result;
	for(size_t i = 0; i < std_array.size(); i++)
	{
		if(i != 0)
			result += strDelimiter;
		result += strPrependToEachObject + std_array[i]->GetStrippedName();
	}
	return result;
}

template<class T, class PARENT>
std::string toString_StrippedRelativeNames(const std::vector<T>& std_array,
										   const PARENT parent, 
										   const std::string& strPrependToEachObject = std::string(""),
                                           const std::string& strDelimiter = std::string(", "))
{
	std::string result;
	for(size_t i = 0; i < std_array.size(); i++)
	{
		if(i != 0)
			result += strDelimiter;
		result += strPrependToEachObject + daeGetStrippedRelativeName(parent, std_array[i]);
	}
	return result;
}

inline bool IsInteger(std::string& strValue)
{
	std::string::value_type c;
	std::string::size_type i;
	std::string::size_type iLength = strValue.size();

	if(iLength == 0)
		return false;

	for(i = 0; i < iLength; i++)
	{
		c = strValue[i];
		if(c == '-' && i > 0)
			return false;
		if(c == '+' && i > 0)
			return false;
		if(!isdigit(c))
			return false;
	}
	return true;
}

inline bool IsDecimal(std::string& strValue)
{
	std::string::value_type c;
	std::string::size_type i;
	std::string::size_type iLength = strValue.size();
	std::string::size_type iFound;

	if(iLength == 0)
		return false;

// Check ih there is '.'
// If there is, check for another; if there are more than one report error
	iFound = strValue.find('.', 0);
	if(iFound != std::string::npos)
	{
		iFound = strValue.find('.', iFound);
		if(iFound != std::string::npos)
			return false;
	}

	for(i = 0; i < iLength; i++)
	{
		c = strValue[i];
		if(c == '-' && i > 0)
			return false;
		if(c == '+' && i > 0)
			return false;
		if(c == '.' && i > 0)
			return false;
		if(!isdigit(c) || c != '.')
			return false;
	}
	return true;
}

inline void HexToString(const unsigned char* szData, size_t nLength, std::string& strHex)
{
	size_t i;
	std::stringstream ss;

	ss.setf(std::ios_base::hex, std::ios::basefield);
	ss.setf(std::ios_base::uppercase);
	ss.fill('0');

	for(i = 0; i < nLength; i++)
		ss << std::setw(2) << (unsigned short)szData[i];

	strHex = ss.str();
}

// Converts the matrix data from xml file (i.e.: 4465736372697074696f6e) into the byte array ("Description")
inline void StringToHex(const std::string& strHex, unsigned char** szData, size_t& nLength)
{
	short n;
	size_t i, j;

	nLength = strHex.size() / 2;
	if(nLength * 2 != strHex.size())
		return;

	*szData = new unsigned char[nLength+1];
	unsigned char* res = *szData;
	res[nLength] = '\0';

	for(i = 0, j = 0; i < strHex.size(); i=i+2, j++)
	{
		std::stringstream ss(std::stringstream::in | std::stringstream::out);
		ss << std::hex << strHex.substr(i, 2);
		ss >> n;
		res[j] = (unsigned char)n;
	}
}

inline std::vector<std::string> ParseString(const std::string& strSource, char cDelimiter)
{
	size_t				 nNoFounded = 0;
	std::vector<std::string>		strarrValues;
	std::string::size_type   iFounded = 0;
	std::string::size_type   iCurrent = 0;
	std::string				strItem;

// Look for the delimiter
	iFounded = strSource.find(cDelimiter);

// If not found return only one element (even if it is empty!)
	if(iFounded == std::string::npos)
	{
		strarrValues.push_back(strSource);
		return strarrValues;
	}

// Iterate through std::string and add substrings
	while(iFounded != std::string::npos)
	{
		strItem = strSource.substr(iCurrent, iFounded-iCurrent);
		LTrim(strItem, ' ');
		RTrim(strItem, ' ');
		strarrValues.push_back(strItem);
		iCurrent = iFounded + 1;
		nNoFounded++;
		iFounded = strSource.find(cDelimiter, iCurrent);
	}

// Add the remaining of the std::string (at the rear end, after the last delimiter), even if it is empty
	if(strarrValues.size() < nNoFounded+1)
	{
		strItem = strSource.substr(iCurrent, strSource.length()-1);
		LTrim(strItem, ' ');
		RTrim(strItem, ' ');
		strarrValues.push_back(strItem);
	}

	return strarrValues;
}

inline std::vector<std::string> ParseString(const std::string& strSource, std::string strDelimiter)
{
	size_t				 nNoFounded = 0;
	std::vector<std::string>		strarrValues;
	std::string::size_type   iFounded = 0;
	std::string::size_type   iCurrent = 0;
	std::string				strItem;
	std::string::size_type n = strDelimiter.length();

// Look for the delimiter
	iFounded = strSource.find(strDelimiter);

// If not found return only one element (even if it is empty!)
	if(iFounded == std::string::npos)
	{
		strarrValues.push_back(strSource);
		return strarrValues;
	}

// Iterate through std::string and add substrings
	while(iFounded != std::string::npos)
	{
		strItem = strSource.substr(iCurrent, iFounded-iCurrent);
		LTrim(strItem, ' ');
		RTrim(strItem, ' ');
		strarrValues.push_back(strItem);
		iCurrent = iFounded + n;
		nNoFounded++;
		iFounded = strSource.find(strDelimiter, iCurrent);
	}

// Add the remaining of the std::string (at the rear end, after the last delimiter), even if it is empty
	if(strarrValues.size() < nNoFounded+1)
	{
		strItem = strSource.substr(iCurrent, strSource.length()-1);
		LTrim(strItem, ' ');
		RTrim(strItem, ' ');
		strarrValues.push_back(strItem);
	}
	
	return strarrValues;
}

inline void MakeUpper(std::string& strSource)
{
	std::string strTemp;

	for(size_t i = 0; i < strSource.length(); i++)
		strTemp += ::toupper(strSource[i]);
	strSource = strTemp;
}

inline void MakeLower(std::string& strSource)
{
	std::string strTemp;

	for(size_t i = 0; i < strSource.length(); i++)
		strTemp += ::tolower(strSource[i]);
	strSource = strTemp;
}

inline std::string Replace(const std::string& strSource, char cFind /*=' '*/, char cReplace /*='_'*/)
{
	std::string strReturn = strSource;

	for(size_t i = 0; i < strReturn.length(); i++)
	{
		if(strReturn[i] == cFind)
		{
			strReturn[i] = cReplace;
			break;
		}
	}
	return strReturn;
}

inline std::string ReplaceAll(const std::string& strSource, char cFind /*=' '*/, char cReplace /*='_'*/)
{
	std::string strReturn = strSource;

	for(size_t i = 0; i < strReturn.length(); i++)
	{
		if(strReturn[i] == cFind)
			strReturn[i] = cReplace;
	}
	return strReturn;
}

inline void RemoveAll(std::string& strSource, const std::string& strFind)
{
	std::string::size_type iFounded;
	std::string::size_type n = strFind.length();

// Look for the string to find
	iFounded = strSource.find(strFind);

// Iterate through std::string and add substrings
	while(iFounded != std::string::npos)
	{
		strSource.erase(iFounded, n);
		iFounded = strSource.find(strFind, iFounded);
	}
} 

inline void RemoveAllNonAlphaNumericCharacters(std::string& strSource)
{
	static std::string strNonAlphaNumerics[10] = {"!","~","@","#","$","%","^","&","*",";"};
	for(size_t i = 0; i < 10; i++)
		RemoveAll(strSource, strNonAlphaNumerics[i]);
} 

inline void RemoveAllNonAlphaNumericCharacters(std::vector<std::string>& strarrSource)
{
	for(size_t i = 0; i < strarrSource.size(); i++)
		RemoveAllNonAlphaNumericCharacters(strarrSource[i]);
} 

inline void LTrim(std::string& strSource, char cTrim /*= ' '*/)
{
    size_t startpos = strSource.find_first_not_of(cTrim);
    if( string::npos != startpos )
        strSource = strSource.substr( startpos );

//	if(strSource.length() == 0)
//		return;
//	char cFounded = strSource.at(0);
//	while(cFounded == cTrim)
//	{
//		strSource.erase(0, 1);
//		if(strSource.empty())
//			break;
//		cFounded = strSource.at(0);
//	}
}

inline void RTrim(std::string& strSource, char cTrim /*= ' '*/)
{
	size_t endpos = strSource.find_last_not_of(cTrim);
	if( string::npos != endpos )
		strSource = strSource.substr( 0, endpos+1 );

//	if(strSource.length() == 0)
//		return;
//	char cFounded = strSource.at(strSource.length()-1);
//	while(cFounded == cTrim)
//	{
//		strSource.erase(strSource.length()-1, 1);
//		if(strSource.empty())
//			break;
//		cFounded = strSource.at(strSource.length()-1);
//	}
}

inline std::string Split(std::vector<std::string>& Array, const char* lpszDelimiter)
{
	std::string strResult;

	if(Array.size() == 1)
		strResult = Array[0];
	if(Array.size() > 1)
	{
		strResult = Array[0];
		for(size_t i = 1; i < Array.size(); i++)
		{
			strResult += lpszDelimiter;
			strResult += Array[i];
		}
	}

	return strResult;
}

inline std::string SplitAndBracket(std::vector<std::string>& Array, const char* lpszDelimiter)
{
	std::string strResult, strElement;

	if(Array.size() == 1)
	{
		strResult = Array[0];
		Enclose(strResult, "[", "]");
	}
	if(Array.size() > 1)
	{
		strResult = Array[0];
		Enclose(strResult, "[", "]");
		for(size_t i = 1; i < Array.size(); i++)
		{
			strResult += lpszDelimiter;

			strElement = Array[i];
			Enclose(strElement, "[", "]");

			strResult += strElement;
		}
	}

	return strResult;
}

inline void Enclose(std::string& strToEnclose, const char* lpszLeft, const char* lpszRight)
{
	std::string strResult;

	strResult = strToEnclose;
	if(strResult.find(' ') != std::string::npos)
	{
		strResult = lpszLeft;
		strResult += strToEnclose;
		strResult += lpszRight;

		strToEnclose = strResult;
	}
}

inline void Enclose(std::string& strToEnclose, char cLeft, char cRight)
{
	std::string strResult;

	strResult = strToEnclose;
	if(strResult.find(' ') != std::string::npos)
	{
		strResult = cLeft;
		strResult += strToEnclose;
		strResult += cRight;

		strToEnclose = strResult;
	}
}

inline bool ParseSingleToken(std::string& strFullName, std::string& strShortName, std::vector<size_t>& narrDomains)
{
	size_t					i;
	std::string				strTemp;
	std::vector<std::string>		strarrTemp;
	std::string::size_type  iLeftBracket;
	std::string::size_type  iRightBracket;

	strShortName.clear();
	narrDomains.clear();

	LTrim(strFullName, ' ');
	RTrim(strFullName, ' ');
	if(strFullName.empty())
		return false;

	iLeftBracket = strFullName.find('(');
	if(iLeftBracket == std::string::npos) // If there is no left (
	{
		iRightBracket = strFullName.find(')'); // Check if there is a right )
		if(iRightBracket != std::string::npos)
			return false; // If there is ) then is return false

		strShortName = strFullName; // Return normal name
		return true;
	}
	if(iLeftBracket == 0) // String like this "(1, 3)" without the name
		return false;

	iRightBracket = strFullName.find(')', iLeftBracket);
	if(iRightBracket == std::string::npos)
		return false;

	if(iRightBracket <= iLeftBracket+1)
		return false;

	strTemp = strFullName.substr(iLeftBracket+1, iRightBracket-iLeftBracket-1);
	strarrTemp = ParseString(strTemp, ',');
	if(strarrTemp.empty())
		return false;

	for(i = 0; i < strarrTemp.size(); i++)
	{
		strTemp = strarrTemp[i];
		try
		{
			narrDomains.push_back(fromString<size_t>(strTemp));
		}
		catch(std::exception& e)
		{
			std::string strWhat = e.what();
			narrDomains.clear();
			return false;
		}
	}
	
	strShortName = strFullName.substr(0, iLeftBracket);
	LTrim(strShortName, ' ');
	RTrim(strShortName, ' ');

	return true;
}

inline double GetTimeInSeconds(void)
{
#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
    DWORD time = GetTickCount();
    return (double)(time / 1.0E3);

#elif defined(__MACH__) || defined(__APPLE__)
// This part needs to be checked thoroughly...
	uint64_t time;
	uint64_t timeNano;
	static mach_timebase_info_data_t sTimebaseInfo;

	// Start the clock.
	time = mach_absolute_time();

	// If this is the first time we've run, get the timebase.
	// We can use denom == 0 to indicate that sTimebaseInfo is 
	// uninitialised because it makes no sense to have a zero 
	// denominator is a fraction.
	if(sTimebaseInfo.denom == 0) 
		mach_timebase_info(&sTimebaseInfo);

	timeNano = time * sTimebaseInfo.numer / sTimebaseInfo.denom;
	return (double)(timeNano / 1.0E9);

#elif __linux__ == 1
	struct timespec time;
	clock_gettime(CLOCK_MONOTONIC, &time);
	return (double)(time.tv_sec + time.tv_nsec / 1.0E9);

#else
    #error Unknown Platform!!
	return 0.0;
#endif
}


}

#endif
