#include "stdafx.h"
#include "xmlfile.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <sstream>
#include "helpers.h"

namespace dae 
{
namespace xml
{
//bool IsInteger(std::string& strValue)
//{
//	string::value_type c;
//	string::size_type i;
//	string::size_type iLength = strValue.size();
//	for(i = 0; i < iLength; i++)
//	{
//		c = strValue[i];
//		if(c == '-' && i > 0)
//			return false;
//		if(!isdigit(c))
//			return false;
//	}
//	return true;
//}
//
//bool IsDecimal(std::string& strValue)
//{
//	string::value_type c;
//	string::size_type i;
//	string::size_type iLength = strValue.size();
//	string::size_type iFound = strValue.find('.', 0);
//	if(iFound != string::npos)
//	{
//		iFound = strValue.find('.', iFound);
//		if(iFound != string::npos)
//			return false;
//	}
//	for(i = 0; i < iLength; i++)
//	{
//		c = strValue[i];
//		if(c == '-' && i > 0)
//			return false;
//		if(!isdigit(c) || c != '.')
//			return false;
//	}
//	return true;
//}
//
///*
//bool IsDate(std::string& strValue)
//{
//	return IsDate(strValue.c_str());
//}
//bool IsTime(std::string& strValue)
//{
//	return IsTime(strValue.c_str());
//}
//bool IsDateTime(std::string& strValue)
//{
//	return IsDateTime(strValue.c_str());
//}
//
//int	Int(std::string& strValue)
//{
//	return Int(strValue.c_str());
//}
//
//long Long(std::string& strValue)
//{
//	return Long(strValue.c_str());
//}
//
//size_t UnsignedInt(std::string& strValue)
//{
//	return UnsignedInt(strValue.c_str());
//}
//
//double XMLDouble(std::string& strValue)
//{
//	return XMLDouble(strValue.c_str());
//}
//
//double Double(std::string& strValue)
//{
//	return Double(strValue.c_str());
//}
//
//float Float(std::string& strValue)
//{
//	return Float(strValue.c_str());
//}
//bool Bool(std::string& strValue)
//{
//	return Bool(strValue.c_str());
//}
//*/
//
//bool CharToHexData(const char* szHex, unsigned char* rch)
//{
//	if(*szHex >= '0' && *szHex <= '9')
//		*rch = *szHex - '0';
//	else if(*szHex >= 'A' && *szHex <= 'F')
//		*rch = *szHex - 'A' + 10;
//	else if(*szHex >= 'a' && *szHex <= 'f')
//		*rch = *szHex - 'a' + 10;
//	else
//		return false; 
//
//	szHex++;
//	if(*szHex >= '0' && *szHex <= '9')
//		(*rch <<= 4) += *szHex - '0';
//	else if(*szHex >= 'A' && *szHex <= 'F')
//		(*rch <<= 4) += *szHex - 'A' + 10;
//	else if(*szHex >= 'a' && *szHex <= 'f')
//		(*rch <<= 4) += *szHex - 'a' + 10;
//	else
//		return false;
//	return true;
//}
//
//// Converts the byte array ("Description") into the hex representation (4465736372697074696f6e)
//char* HexToString(void *szData, size_t& ulSize)
//{
//	size_t         iSize, iValue;
//	char           buffer[3];
//	char*          szResult;
//	string         strHex;
//	unsigned char* szSource = (unsigned char*)szData;
//
//	for(size_t i = 0; i < ulSize; i++)
//	{
//		iValue = *szSource;
//		sprintf(buffer, "%02x", iValue);
////		_itoa(iValue, buffer, 16);
//		strHex += buffer;
//		szSource++;
//	}
//
//	iSize = strHex.size();
//	if(iSize = 0)
//		return NULL;
//	szResult = new char[strHex.size()];
//	strcpy(szResult, strHex.c_str());
//
//	return szResult;
//}
//
//// Converts the matrix data from xml file (i.e.: 4465736372697074696f6e) into the byte array ("Description")
//void* StringToHex(const char *szString, size_t& ulSize)
//{
//	const    char* szSource;
//	void*          szResult;
//	unsigned char* szDest;
//
//// Calculate size of the byte array
//	ulSize   = (size_t)strlen(szString) / 2;
//// Allocate memory for the result byte array
//	szResult = new unsigned char[ulSize];
//// Get the pointer on the source char array
//	szSource = szString;
//// Set the temporary pointer which points to the beggining of the result byte array
//	szDest   = (unsigned char*)szResult;
//// Iterate through the source char array and convert byte into the HEX representation of it
//	for(size_t i = 0; i < ulSize; i++) 
//	{
//		if(!CharToHexData(szSource, szDest))
//		{
//			delete[] (unsigned char*)szResult;
//			szResult = NULL;
//			return NULL;
//		}
//		szSource += 2;
//		szDest++;
//	}
//	return szResult;
//}
//
//std::vector<std::string> ParseString(std::string& strSource, char cDelimiter)
//{
//	std::vector<std::string> strarrValues;
//	std::string::size_type   iFounded = 0;
//	std::string::size_type   iCurrent = 0;
//	std::string				 strItem;
//
//	iFounded = strSource.find(cDelimiter);
//	if(iFounded == std::string::npos && !strSource.empty())
//	{
//		strarrValues.push_back(strSource);
//		return strarrValues;
//	}
//	while(iFounded != std::string::npos)
//	{
//		strItem = strSource.substr(iCurrent, iFounded-iCurrent);
//		LTrim(strItem, ' ');
//		RTrim(strItem, ' ');
//		strarrValues.push_back(strItem);
//		iCurrent = iFounded + 1;
//		iFounded = strSource.find(cDelimiter, iCurrent);
//	}
//	if(iCurrent < strSource.length())
//	{
//		strItem = strSource.substr(iCurrent, strSource.length()-1);
//		LTrim(strItem, ' ');
//		RTrim(strItem, ' ');
//		strarrValues.push_back(strItem);
//	}
//	return strarrValues;
//}
//
//std::vector<std::string> ParseString(std::string& strSource, string strDelimiter)
//{
//	std::vector<std::string> strarrValues;
//	std::string::size_type   iFounded = 0;
//	std::string::size_type   iCurrent = 0;
//	std::string				 strItem;
//	std::string::size_type n = strDelimiter.length();
//
//	iFounded = strSource.find(strDelimiter);
//	if(iFounded == std::string::npos && !strSource.empty())
//	{
//		strarrValues.push_back(strSource);
//		return strarrValues;
//	}
//	while(iFounded != std::string::npos)
//	{
//		strItem = strSource.substr(iCurrent, iFounded-iCurrent);
//		LTrim(strItem, ' ');
//		RTrim(strItem, ' ');
//		strarrValues.push_back(strItem);
//		iCurrent = iFounded + n;
//		iFounded = strSource.find(strDelimiter, iCurrent);
//	}
//	if(iCurrent < strSource.length())
//	{
//		strItem = strSource.substr(iCurrent, strSource.length()-1);
//		LTrim(strItem, ' ');
//		RTrim(strItem, ' ');
//		strarrValues.push_back(strItem);
//	}
//	return strarrValues;
//}
//
//void MakeUpper(std::string& strSource)
//{
//	std::string strTemp;
//
//	for(size_t i = 0; i < strSource.length(); i++)
//		strTemp += ::toupper(strSource[i]);
//	strSource = strTemp;
//}
//
//void MakeLower(std::string& strSource)
//{
//	std::string strTemp;
//
//	for(size_t i = 0; i < strSource.length(); i++)
//		strTemp += ::tolower(strSource[i]);
//	strSource = strTemp;
//}
//
//std::string Replace(std::string strSource, char cFind /*=' '*/, char cReplace /*='_'*/)
//{
//	std::string strReturn = strSource;
//
//	for(size_t i = 0; i < strReturn.length(); i++)
//	{
//		if(strReturn[i] == cFind)
//		{
//			strReturn[i] = cReplace;
//			break;
//		}
//	}
//	return strReturn;
//}
//
//std::string ReplaceAll(std::string strSource, char cFind /*=' '*/, char cReplace /*='_'*/)
//{
//	std::string strReturn = strSource;
//
//	for(size_t i = 0; i < strReturn.length(); i++)
//	{
//		if(strReturn[i] == cFind)
//			strReturn[i] = cReplace;
//	}
//	return strReturn;
//}
//
//void LTrim(std::string& strSource, char cTrim /*= ' '*/)
//{
//	std::string::size_type iFounded = 0;
//
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
//}
//
//void RTrim(std::string& strSource, char cTrim /*= ' '*/)
//{
//	std::string::size_type iFounded = 0;
//
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
//}
//
//
//std::string Split(std::vector<std::string>& Array, const char* lpszDelimiter)
//{
//	std::string strResult;
//
//	if(Array.size() == 1)
//		strResult = Array[0];
//	if(Array.size() > 1)
//	{
//		strResult = Array[0];
//		for(size_t i = 1; i < Array.size(); i++)
//		{
//			strResult += lpszDelimiter;
//			strResult += Array[i];
//		}
//	}
//
//	return strResult;
//}
//
//std::string SplitAndBracket(std::vector<std::string>& Array, const char* lpszDelimiter)
//{
//	std::string strResult, strElement;
//
//	if(Array.size() == 1)
//	{
//		strResult = Array[0];
//		Enclose(strResult, "[", "]");
//	}
//	if(Array.size() > 1)
//	{
//		strResult = Array[0];
//		Enclose(strResult, "[", "]");
//		for(size_t i = 1; i < Array.size(); i++)
//		{
//			strResult += lpszDelimiter;
//
//			strElement = Array[i];
//			Enclose(strElement, "[", "]");
//
//			strResult += strElement;
//		}
//	}
//
//	return strResult;
//}
//
//void Enclose(std::string& strToEnclose, const char* lpszLeft, const char* lpszRight)
//{
//	std::string strResult;
//
//	strResult = strToEnclose;
//	if(strResult.find(' ') != std::string::npos)
//	{
//		strResult = lpszLeft;
//		strResult += strToEnclose;
//		strResult += lpszRight;
//
//		strToEnclose = strResult;
//	}
//}
//
//void Enclose(std::string& strToEnclose, char cLeft, char cRight)
//{
//	std::string strResult;
//
//	strResult = strToEnclose;
//	if(strResult.find(' ') != std::string::npos)
//	{
//		strResult = cLeft;
//		strResult += strToEnclose;
//		strResult += cRight;
//
//		strToEnclose = strResult;
//	}
//}

}
}

