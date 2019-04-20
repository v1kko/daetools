#include "stdafx.h"
#include "xmlfile.h"
#include <stdexcept>

#include "helpers.h"

namespace daetools 
{
namespace xml
{

std::string xmlReadStream::theSeparators = " \t\n\r";
char xmlReadStream::theTerminators = '>';
char xmlReadStream::theStarters = '<';


xmlReadStream::xmlReadStream (std::istream & inStream)
			 : theStream (inStream)
{
} 


xmlReadStream::~xmlReadStream ()
{
} 


std::string xmlReadStream::readString ()
{
	string aResult;
	readString (aResult);
	return aResult;
}  

void xmlReadStream::readString (std::string & inString)
{
	inString = theBuffer;
	theBuffer.erase();
	
	char c;
	
	while (theStream.good() && !theStream.eof())
	{
		theStream.get(c);
		if (isSeparator(c))
		{
			if(inString.length() && 
				!isSeparator(inString[inString.length()-1]) && 
				!isStarter(inString[inString.length()-1]) )
			{
				// add to a string only if last was not separator
				inString += ' ';
			}
			// otherwise just ignore it
		}
		else if(isStarter(c))
		{
			// its start
			if (inString.length())
			{
				// we have something so it is the start of something new
				// save it for future and return
				theBuffer = c;
				if (inString.length())
					validateString(inString);
				return;
			}
			else
			{
				// it is a start
				inString = c;
			}
		}
		else if(isTerminator(c))
		{
			if(inString.length() && isStarter(inString[0]))
			{
				if ((inString[inString.length()-1] == ' '))
					inString[inString.length()-1] = c;
				else
					inString += c;
				
				if (inString.length())
					validateString(inString);
				return;
			}
			else
			{
				inString += c;
			}
		}
		else
		{
			// add to string and continue
			inString += c;
		}
	}

}  

void xmlReadStream::validateString (std::string & inString)
{
	if(isStarter(inString[0]))
	   if((inString.length() < 3) || !isTerminator(inString[inString.length()-1]))
		{
			string strMessage = "Invalid tag : " + inString;
			throw new xmlException(strMessage);
		}
}

bool xmlReadStream::isTerminator (const char c) const
{
	return (theTerminators == c);
}

bool xmlReadStream::isStarter (const char c) const
{
	return (theStarters == c);
}

bool xmlReadStream::isSeparator (const char c) const
{
	return (theSeparators.find(c) != std::string::npos);
}


std::string g_strMessage;

xmlException::xmlException(std::string& strDescription)
{
	m_strDescription = strDescription;
}

xmlException::xmlException(const char* lpszDescription)
{
	m_strDescription = lpszDescription;
}

xmlException::~xmlException() throw()
{

}

const char* xmlException::what() const throw()
{
	g_strMessage  = "xmlParser error: ";
	g_strMessage += m_strDescription;
	return g_strMessage.c_str();
}


}
}

