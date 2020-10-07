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
#include "logs.h"
#include "cs_logs.h"

namespace cs
{
std::shared_ptr<csLog_t> createLog_StdOut()
{
    return std::shared_ptr<csLog_t>(new csLog_StdOut());
}

std::shared_ptr<csLog_t> createLog_TextFile(const std::string& fileName)
{
    return std::shared_ptr<csLog_t>(new csLog_TextFile(fileName));
}

csLog_StdOut::csLog_StdOut()
{
    pe_rank   = 0;
    m_strName = "StdOut";
}

csLog_StdOut::~csLog_StdOut()
{
    Disconnect();
}

std::string	csLog_StdOut::GetName(void) const
{
    return m_strName;
}

bool csLog_StdOut::Connect(int rank)
{
    pe_rank = rank;
    return true;
}

void csLog_StdOut::Disconnect()
{
    std::cout.flush();
}

bool csLog_StdOut::IsConnected()
{
    return true;
}

void csLog_StdOut::Message(const std::string& strMessage)
{
    if(strMessage.empty())
        return;

    std::cout << strMessage << std::endl;
    std::cout.flush();
}

csLog_TextFile::csLog_TextFile(const std::string& strFileName)
{
    pe_rank   = 0;
    m_strName = "TextFile";

    if(strFileName.empty())
    {
        char buffer[L_tmpnam];
        tmpnam(buffer);
        m_strFilename = buffer;
    }
    else
    {
        m_strFilename = strFileName;
    }
}

csLog_TextFile::~csLog_TextFile(void)
{
    Disconnect();
}

std::string csLog_TextFile::GetName(void) const
{
    return m_strName;
}

bool csLog_TextFile::Connect(int rank)
{
    pe_rank = rank;
    file.open(m_strFilename.c_str());
    if(!file.is_open())
        return false;
    return true;
}

void csLog_TextFile::Disconnect()
{
    if(file.is_open())
    {
        file.flush();
        file.close();
    }
}

bool csLog_TextFile::IsConnected()
{
    return file.is_open();
}

void csLog_TextFile::Message(const std::string& strMessage)
{
    if(strMessage.empty())
        return;

    if(file.is_open())
        file << strMessage << std::endl;
}

std::string csLog_TextFile::GetFilename() const
{
    return m_strFilename;
}


}
